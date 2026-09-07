import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from r_pqnn import (QUPIDModel, compute_minimum_dot_product, compute_local_dp_parameters,
                    add_gaussian_noise, NOISE_PROBABILITY, SHOTS, MODEL_TAG)
from to_complex import complex_partition

# Transfer (gradient-free) evaluation of the quantum-noise model. Reuses the surrogate adversarials
# produced by the main TRANSFER ATTACK pipeline (they are model-independent) and pushes them through
# R-QUPID's own encoder -> SmoothQUPID. Unlike the analytic channel, the shot noise is stochastic
# per forward, so this is the regime where it could actually feed the randomized smoothing.
ADV_PATH = Path("TRANSFER ATTACK/transfer_adv_FTTransformer.npz")  # or MTL_LSTM / QH_DNN
MODEL_PATH = Path(f"MODELS_R-QUPID/{MODEL_TAG}.pth")
CLEAN_COMPLEX = Path("dataset3/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
EPS_T, DEL_T, ATTACK_BOUND = 20.0, 1e-5, 0.05
N_SMOOTH, SMOOTH_BATCH = 100, 128

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
                p1 = add_gaussian_noise(torch.cat((b[0], b[1]), 1), self.sigmas_sq[0])
                p2 = add_gaussian_noise(torch.cat((b[1], b[2]), 1), self.sigmas_sq[1])
                p3 = add_gaussian_noise(torch.cat((b[2], b[3]), 1), self.sigmas_sq[2])
                p4 = add_gaussian_noise(torch.cat((b[3], b[0]), 1), self.sigmas_sq[3])
                for idx in self.base_classifier(p1, p2, p3, p4).argmax(1).cpu().numpy():
                    counts[idx] += 1
        return counts.argmax(), counts / n

def to_relays(adv_X_120, features, device):
    return [torch.tensor(complex_partition(adv_X_120, features, p)[:, :16], dtype=torch.complex64).to(device)
            for p in ("R1", "R2", "R3", "R4")]

def evaluate_variant(smoothed, relays, y, tag):
    preds = []
    for i in tqdm(range(y.shape[0]), desc=tag, leave=False):
        pred, _ = smoothed.predict([relays[k][i] for k in range(4)], N_SMOOTH, SMOOTH_BATCH)
        preds.append(pred)
    preds = np.array(preds)
    acc = (preds == y.numpy()).mean() * 100
    mcc = matthews_corrcoef(y.numpy(), preds)
    print(f"[{tag}]  Accuracy: {acc:.2f}%  MCC: {mcc:.4f}")
    return acc, mcc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()} | model={MODEL_PATH.name} | default.mixed p={NOISE_PROBABILITY} shots={SHOTS}")
    print(f"Transfer source: {ADV_PATH.name}")

    adv = np.load(ADV_PATH, allow_pickle=True)
    features = adv["features"].tolist()
    y = torch.tensor(adv["y"], dtype=torch.long)
    variants = ["x", "fgsm_005", "pgd_005", "fgsm_010", "pgd_010"]  # "x" = clean reference

    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{MODEL_PATH}', map_location=device))
    model.eval()

    with np.load(CLEAN_COMPLEX, allow_pickle=True) as data:
        Xte = [torch.tensor(data[f'x_test_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
    P = [torch.cat((Xte[a], Xte[b]), dim=1).numpy() for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
    sigmas_sq = [compute_local_dp_parameters(EPS_T, DEL_T, NOISE_PROBABILITY,
                 compute_minimum_dot_product(p), delta_L=ATTACK_BOUND)[2] ** 2 for p in P]
    smoothed = SmoothQUPID(model, 6, sigmas_sq)

    summary = []
    for key in variants:
        if key not in adv.files:
            continue
        label = "clean" if key == "x" else key
        acc, mcc = evaluate_variant(smoothed, to_relays(adv[key].astype(np.float32), features, device), y, label)
        summary.append((label, acc, mcc))

    print("\n" + "=" * 56)
    print(f"QUANTUM-NOISE TRANSFER  (p={NOISE_PROBABILITY}, shots={SHOTS})")
    print("=" * 56)
    print(f"{'variant':<12}{'accuracy':>12}{'MCC':>10}")
    for label, acc, mcc in summary:
        print(f"{label:<12}{acc:>11.2f}%{mcc:>10.4f}")
    print("-" * 56)
    print("Compare against the p=0 baseline (clean 81% -> transfer MCC ~0.09).")

if __name__ == "__main__":
    main()
