import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from r_pqnn import (QUPIDModel, compute_minimum_dot_product, compute_local_dp_parameters,
                    add_gaussian_noise, NOISE_PROBABILITY, SHOTS, MODEL_TAG)

MODEL_PATH = Path(f"MODELS_R-QUPID/{MODEL_TAG}.pth")
EPS_T, DEL_T = 20.0, 1e-5
EPS = 0.05
N_SMOOTH, SMOOTH_BATCH = 100, 128
SUBSET_SIZE = 300

class QUPIDWrapper(nn.Module):
    """[0,1] -> [-1,1] phasors -> partitions -> QUPIDModel."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        c = torch.complex(x[:, :, :, 0] * 2.0 - 1.0, x[:, :, :, 1] * 2.0 - 1.0)
        p1 = torch.cat((c[:, 0, :], c[:, 1, :]), 1)
        p2 = torch.cat((c[:, 1, :], c[:, 2, :]), 1)
        p3 = torch.cat((c[:, 2, :], c[:, 3, :]), 1)
        p4 = torch.cat((c[:, 3, :], c[:, 0, :]), 1)
        return self.model(p1, p2, p3, p4)

def fgsm_attack(wrapper, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(wrapper(x), y)
    grad = torch.autograd.grad(loss, x, allow_unused=True)[0]
    if grad is None:
        grad = torch.zeros_like(x)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return (x + eps * grad.sign()).detach(), grad

class SmoothQUPID:
    def __init__(self, base_classifier, num_classes, sigmas_sq):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigmas_sq = sigmas_sq

    def predict(self, relays_single, n, batch_size):
        self.base_classifier.eval()
        counts = np.zeros(self.num_classes, dtype=float)
        r1, r2, r3, r4 = [r.unsqueeze(0) if r.dim() == 1 else r for r in relays_single]
        remaining = n
        with torch.no_grad():
            while remaining > 0:
                bs = min(batch_size, remaining)
                remaining -= bs
                b = [r.repeat((bs, 1)) for r in (r1, r2, r3, r4)]
                np1 = add_gaussian_noise(torch.cat((b[0], b[1]), 1), self.sigmas_sq[0])
                np2 = add_gaussian_noise(torch.cat((b[1], b[2]), 1), self.sigmas_sq[1])
                np3 = add_gaussian_noise(torch.cat((b[2], b[3]), 1), self.sigmas_sq[2])
                np4 = add_gaussian_noise(torch.cat((b[3], b[0]), 1), self.sigmas_sq[3])
                for idx in self.base_classifier(np1, np2, np3, np4).argmax(1).cpu().numpy():
                    counts[idx] += 1
        return counts.argmax(), counts / n

def main():
    complex_dataset = Path("dataset3/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()} | model={MODEL_PATH.name} | default.mixed p={NOISE_PROBABILITY} | fair Linf eps={EPS}")
    if SHOTS is not None:
        print(f"NOTE: SHOTS={SHOTS} -> the attack gradient is taken over one shot-noise draw.")

    with np.load(complex_dataset, allow_pickle=True) as data:
        Xte = [torch.tensor(data[f'x_test_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
        y_test = torch.tensor([mapping_classes(y.item()) for y in data['y_test']], dtype=torch.long)

    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{MODEL_PATH}', map_location=device))
    model.eval()
    wrapper = QUPIDWrapper(model).to(device).eval()

    P = [torch.cat((Xte[a], Xte[b]), dim=1).numpy() for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
    sigmas_sq = [compute_local_dp_parameters(EPS_T, DEL_T, NOISE_PROBABILITY,
                 compute_minimum_dot_product(p), delta_L=EPS)[2] ** 2 for p in P]
    smoothed = SmoothQUPID(model, 6, sigmas_sq)

    n_use = min(SUBSET_SIZE, Xte[0].shape[0])
    correct = 0
    all_preds, all_targets, all_probs = [], [], []
    grad_norm_sum, delta_sum = 0.0, 0.0

    for i in tqdm(range(n_use), desc="per-sample fair Linf (default.mixed)"):
        c = torch.stack([Xte[k][i] for k in range(4)], dim=0).unsqueeze(0)
        x = torch.stack([(c.real.float() + 1.0) / 2.0, (c.imag.float() + 1.0) / 2.0], dim=-1).to(device)
        y = y_test[i].unsqueeze(0).to(device)

        adv, grad = fgsm_attack(wrapper, x, y, EPS / 2.0)  # /2 = fair Linf (wrapper [0,1]->[-1,1] x2)
        grad_norm_sum += float(torch.nan_to_num(grad, nan=0.0).norm())
        delta_sum += float((adv - x).norm())

        adv_c = torch.complex(adv[0, :, :, 0] * 2.0 - 1.0, adv[0, :, :, 1] * 2.0 - 1.0)
        pred, prob = smoothed.predict([adv_c[k, :] for k in range(4)], N_SMOOTH, SMOOTH_BATCH)

        correct += int(pred == y.item())
        all_preds.append(pred)
        all_targets.append(y.item())
        all_probs.append(prob)

    acc = correct / n_use * 100
    mcc = matthews_corrcoef(all_targets, all_preds)
    print(f"\nFair Linf (backprop, eps={EPS}) on the quantum-noise model:")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {precision_score(all_targets, all_preds, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_targets, all_preds, average='macro', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(all_targets, all_preds, average='macro', zero_division=0):.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"G-Mean:    {geometric_mean_score(all_targets, all_preds, average='macro'):.4f}")
    print(f"mean input-gradient norm: {grad_norm_sum / n_use:.6f}  (>0 = real gradient, not masked)")
    print(f"mean ||x_adv - x||_2:     {delta_sum / n_use:.6f}  (>0 = the attack lands)")

    pred_counts, true_counts = Counter(all_preds), Counter(all_targets)
    print("\nDistribution of Predictions vs Reality:")
    for c_idx in range(6):
        print(f"Class {c_idx}: Predicted {pred_counts.get(c_idx, 0)} times | (In reality there were {true_counts.get(c_idx, 0)})")
    print("-" * 60)

if __name__ == "__main__":
    main()
