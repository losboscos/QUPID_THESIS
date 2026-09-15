import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
# Local (parameter-shift) r_pqnn.py -> the qnode uses the real quantum-native differentiation.
from r_pqnn import QUPIDModel, compute_minimum_dot_product, compute_local_dp_parameters, add_gaussian_noise

MODEL_PATH = Path("MODELS_R-QUPID/r_qupid_S1_L005.pth")
ATTACK_BOUND = 0.05
NOISE_PROBABILITY = 0.0
EPS_T, DEL_T = 20.0, 1e-5
EPS = 0.05
EOT_ITER = 5
N_SMOOTH = 100
SUBSET_SIZE = 30

class NoisyWrapper(nn.Module):
    
    def __init__(self, model, sigmas_sq):
        super().__init__()
        self.model = model
        self.sigmas_sq = sigmas_sq

    def forward(self, x):
        reals = x[:, :, :, 0] * 2.0 - 1.0
        imags = x[:, :, :, 1] * 2.0 - 1.0
        c = torch.complex(reals, imags)
        r1, r2, r3, r4 = c[:, 0, :], c[:, 1, :], c[:, 2, :], c[:, 3, :]
        p1 = add_gaussian_noise(torch.cat((r1, r2), dim=1), self.sigmas_sq[0])
        p2 = add_gaussian_noise(torch.cat((r2, r3), dim=1), self.sigmas_sq[1])
        p3 = add_gaussian_noise(torch.cat((r3, r4), dim=1), self.sigmas_sq[2])
        p4 = add_gaussian_noise(torch.cat((r4, r1), dim=1), self.sigmas_sq[3])
        return self.model(p1, p2, p3, p4)

def eot_fgsm_attack(wrapper, x, y, eps, eot_iter):
    # Average the gradient over eot_iter noisy draws. parameter-shift gives NaN each draw ->
    # nan_to_num -> 0 -> averaged 0 -> x_adv == x -> no-op (EoT cannot fix the shattered gradient).
    x0 = x.clone().detach()
    grad_sum = torch.zeros_like(x0)
    for _ in range(eot_iter):
        xr = x0.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(wrapper(xr), y)
        grad = torch.autograd.grad(loss, xr, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(xr)
        grad_sum = grad_sum + torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    avg_grad = grad_sum / eot_iter
    return (x0 + eps * avg_grad.sign()).detach(), avg_grad

class SmoothQUPID:
    def __init__(self, base_classifier: nn.Module, num_classes: int, sigmas_sq: list):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigmas_sq = sigmas_sq

    def predict(self, relays_single: list, n: int):
        # batch_size = 1: parameter-shift cannot broadcast the AmplitudeEmbedding (Mottonen) even in the forward
        self.base_classifier.eval()
        counts = np.zeros(self.num_classes, dtype=float)
        r1, r2, r3, r4 = [r.unsqueeze(0) if r.dim() == 1 else r for r in relays_single]
        with torch.no_grad():
            for _ in range(n):
                p1 = torch.cat((r1, r2), dim=1)
                p2 = torch.cat((r2, r3), dim=1)
                p3 = torch.cat((r3, r4), dim=1)
                p4 = torch.cat((r4, r1), dim=1)
                np1 = add_gaussian_noise(p1, self.sigmas_sq[0])
                np2 = add_gaussian_noise(p2, self.sigmas_sq[1])
                np3 = add_gaussian_noise(p3, self.sigmas_sq[2])
                np4 = add_gaussian_noise(p4, self.sigmas_sq[3])
                logits = self.base_classifier(np1, np2, np3, np4)
                counts[logits.argmax(dim=1).item()] += 1
        return counts.argmax(), counts / n

def main():
    complex_dataset = Path("dataset3/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()} | model={MODEL_PATH.name} | parameter-shift | EoT-FGSM iter={EOT_ITER} | subset={SUBSET_SIZE}")

    with np.load(complex_dataset, allow_pickle=True) as data:
        Xte = [torch.tensor(data[f'x_test_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
        Xtr = [torch.tensor(data[f'x_train_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
        y_test = torch.tensor([mapping_classes(y.item()) for y in data['y_test']], dtype=torch.long)

    print("Instantiating the Hybrid R-QUPID Model...")
    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{MODEL_PATH}', map_location=device))
    model.eval()

    P = [torch.cat((Xtr[a], Xtr[b]), dim=1).numpy() for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
    phis = [compute_minimum_dot_product(p) for p in P]
    sigmas_sq = [compute_local_dp_parameters(EPS_T, DEL_T, NOISE_PROBABILITY, ph, delta_L=ATTACK_BOUND)[2] ** 2 for ph in phis]
    wrapper = NoisyWrapper(model, sigmas_sq).to(device).eval()
    smoothed = SmoothQUPID(model, num_classes=6, sigmas_sq=sigmas_sq)

    n_use = min(SUBSET_SIZE, Xte[0].shape[0])
    correct, total = 0, 0
    all_preds, all_targets, all_probs = [], [], []
    grad_norm_sum, delta_sum = 0.0, 0.0

    for i in tqdm(range(n_use), desc="per-sample EoT-FGSM (parameter-shift)"):
        c = torch.stack([Xte[k][i] for k in range(4)], dim=0).unsqueeze(0)  # [1,4,16] complex
        x = torch.stack([(c.real.float() + 1.0) / 2.0, (c.imag.float() + 1.0) / 2.0], dim=-1).to(device)  # [1,4,16,2]
        y = y_test[i].unsqueeze(0).to(device)

        adv, grad = eot_fgsm_attack(wrapper, x, y, EPS / 2.0, EOT_ITER)
        grad_norm_sum += float(torch.nan_to_num(grad, nan=0.0).norm())
        delta_sum += float((adv - x).norm())

        adv_c = torch.complex(adv[0, :, :, 0] * 2.0 - 1.0, adv[0, :, :, 1] * 2.0 - 1.0)
        relays = [adv_c[0, :], adv_c[1, :], adv_c[2, :], adv_c[3, :]]
        pred, prob_dist = smoothed.predict(relays, n=N_SMOOTH)

        correct += int(pred == y.item())
        total += 1
        all_preds.append(pred)
        all_targets.append(y.item())
        all_probs.append(prob_dist)

    test_accuracy = (correct / total) * 100
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_targets, all_preds)
    g_mean = geometric_mean_score(all_targets, all_preds, average='macro')
    try:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = 0.0

    print(f"\nResults for EoT-FGSM (parameter-shift, eps={EPS}):")
    print(f"Accuracy:  {test_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"G-Mean:    {g_mean:.4f}")
    print(f"mean input-gradient norm: {grad_norm_sum / total:.6f}")
    print(f"mean ||x_adv - x||_2:     {delta_sum / total:.6f}")

    pred_counts = Counter(all_preds)
    true_counts = Counter(all_targets)
    print("\nDistribution of Predictions vs Reality:")
    for class_idx in range(6):
        p_count = pred_counts.get(class_idx, 0)
        t_count = true_counts.get(class_idx, 0)
        print(f"Class {class_idx}: Predicted {p_count} times | (In reality there were {t_count})")
    print("-" * 60)
    print("~81% with grad NaN/0 and ||x_adv - x|| = 0 -> EoT does NOT defeat the shattered gradient.")
    print("Only backprop (correct autodiff) or a transfer attack break masking A.")

if __name__ == "__main__":
    main()
