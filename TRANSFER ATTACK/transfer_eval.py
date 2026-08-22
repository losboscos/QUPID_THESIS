import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from pathlib import Path
import math
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
# R-QUPID as-deployed (root r_pqnn.py = backprop forward pass -> batchable smoothing).
from r_pqnn import QUPIDModel, compute_minimum_dot_product, compute_local_dp_parameters, add_gaussian_noise
# The SAME encoder R-QUPID uses to turn 120 real features into complex phasors.
from to_complex import complex_partition

ADV_PATH = Path("TRANSFER/transfer_adv_FTTransformer.npz")
MODEL_PATH = Path("MODELS_R-QUPID/r_qupid_model_L005.pth")
CLEAN_COMPLEX = Path("dataset3/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
ATTACK_BOUND = 0.05
NOISE_PROBABILITY = 0.0
EPS_T, DEL_T = 20.0, 1e-5
N_SMOOTH = 100
SMOOTH_BATCH = 128

class SmoothQUPID:
    def __init__(self, base_classifier: nn.Module, num_classes: int, sigmas_sq: list):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigmas_sq = sigmas_sq

    def predict(self, relays_single: list, n: int, batch_size: int):
        self.base_classifier.eval()
        counts = np.zeros(self.num_classes, dtype=float)
        r1, r2, r3, r4 = [r.unsqueeze(0) if r.dim() == 1 else r for r in relays_single]
        remaining = n
        with torch.no_grad():
            while remaining > 0:
                bs = min(batch_size, remaining)
                remaining -= bs
                b = [r.repeat((bs, 1)) for r in (r1, r2, r3, r4)]
                p1 = add_gaussian_noise(torch.cat((b[0], b[1]), dim=1), self.sigmas_sq[0])
                p2 = add_gaussian_noise(torch.cat((b[1], b[2]), dim=1), self.sigmas_sq[1])
                p3 = add_gaussian_noise(torch.cat((b[2], b[3]), dim=1), self.sigmas_sq[2])
                p4 = add_gaussian_noise(torch.cat((b[3], b[0]), dim=1), self.sigmas_sq[3])
                logits = self.base_classifier(p1, p2, p3, p4)
                for idx in logits.argmax(dim=1).cpu().numpy():
                    counts[idx] += 1
        return counts.argmax(), counts / n

def to_relays(adv_X_120, features, device):
    """120 real MinMax features -> R-QUPID's 4 complex phasor relays (first 16 each)."""
    relays = []
    for prefix in ("R1", "R2", "R3", "R4"):
        c = complex_partition(adv_X_120, features, prefix)[:, :16]
        relays.append(torch.tensor(c, dtype=torch.complex64).to(device))
    return relays

def evaluate_variant(smoothed, relays, y, device, tag):
    correct = 0
    all_preds, all_targets, all_probs = [], [], []
    n = y.shape[0]
    for i in tqdm(range(n), desc=tag, leave=False):
        relays_single = [relays[k][i] for k in range(4)]
        pred, prob = smoothed.predict(relays_single, n=N_SMOOTH, batch_size=SMOOTH_BATCH)
        correct += int(pred == y[i].item())
        all_preds.append(pred)
        all_targets.append(y[i].item())
        all_probs.append(prob)

    acc = correct / n * 100
    mcc = matthews_corrcoef(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    g_mean = geometric_mean_score(all_targets, all_preds, average='macro')
    try:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = 0.0

    print(f"\n[{tag}]  Evaluated {n} samples:")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"MCC:       {mcc:.4f}")   
    print(f"G-Mean:    {g_mean:.4f}")

    pred_counts = Counter(all_preds)
    true_counts = Counter(all_targets)
    print("\nDistribution of Predictions vs Reality:")
    for c in range(6):
        print(f"Class {c}: Predicted {pred_counts.get(c, 0):4d} times | (In reality there were {true_counts.get(c, 0):4d})")
    print("-" * 60 + "\n")
    return acc, mcc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()} | target={MODEL_PATH.name} | transfer from surrogate {ADV_PATH.name}")

    adv = np.load(ADV_PATH, allow_pickle=True)
    features = adv["features"].tolist()
    y = torch.tensor(adv["y"], dtype=torch.long)
    # "x" = raw (unclamped) clean subset -> reproduces R-QUPID's deployed clean phasors exactly.
    variants = ["x", "fgsm_005", "pgd_005", "fgsm_010", "pgd_010"]

    print("Instantiating the Hybrid R-QUPID Model...")
    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{MODEL_PATH}', map_location=device))
    model.eval()

    # Defense sigma calibrated on the FULL clean complex test set (identical to root fgsm.py).
    with np.load(CLEAN_COMPLEX, allow_pickle=True) as data:
        Xte = [torch.tensor(data[f'x_test_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
    P = [torch.cat((Xte[a], Xte[b]), dim=1).numpy() for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
    sigmas_sq = [compute_local_dp_parameters(EPS_T, DEL_T, NOISE_PROBABILITY,
                 compute_minimum_dot_product(p), delta_L=ATTACK_BOUND)[2] ** 2 for p in P]
    smoothed = SmoothQUPID(model, num_classes=6, sigmas_sq=sigmas_sq)

    summary = []
    for key in variants:
        if key not in adv.files:
            continue
        label = "clean" if key == "x" else key
        relays = to_relays(adv[key].astype(np.float32), features, device)
        acc, mcc = evaluate_variant(smoothed, relays, y, device, label)
        summary.append((label, acc, mcc))

    print("\n" + "=" * 60)
    print("TRANSFER SUMMARY  (surrogate: FTTransformer  ->  target: R-QUPID L005)")
    print("=" * 60)
    print(f"{'variant':<12}{'accuracy':>12}{'MCC':>10}")
    for key, acc, mcc in summary:
        print(f"{key:<12}{acc:>11.2f}%{mcc:>10.4f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
