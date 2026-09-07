import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
from r_pqnn import QUPIDModel, zscore_stats, zscore, add_zscore_noise, TRAIN_SIGMA, MODEL_TAG

# Evaluation of the consistency-trained model: fair L-inf FGSM applied directly on the z-scored
# features (the same convention used for the classical baselines), against randomized smoothing
# at the matched sigma. The attack is non-adaptive, so the result is an upper bound on robustness.
MODEL_PATH = Path(f"MODELS_R-QUPID/{MODEL_TAG}.pth")
EVAL_SIGMA = TRAIN_SIGMA
EPS = 0.05
N_SMOOTH = 100
SUBSET_SIZE = 500

class ZWrapper(nn.Module):
    """z-scored real/imag features [B,4,16,2] -> complex relays -> partitions -> QUPIDModel (no noise)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        c = torch.complex(x[:, :, :, 0], x[:, :, :, 1])
        p1 = torch.cat((c[:, 0, :], c[:, 1, :]), 1)
        p2 = torch.cat((c[:, 1, :], c[:, 2, :]), 1)
        p3 = torch.cat((c[:, 2, :], c[:, 3, :]), 1)
        p4 = torch.cat((c[:, 3, :], c[:, 0, :]), 1)
        return self.model(p1, p2, p3, p4)

def fgsm(wrapper, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(wrapper(x), y)
    grad = torch.autograd.grad(loss, x, allow_unused=True)[0]
    if grad is None:
        grad = torch.zeros_like(x)
    return (x + eps * torch.nan_to_num(grad, nan=0.0).sign()).detach()

def smooth_predict(model, z_relays, sigma, n, num_classes=6, batch=100):
    model.eval()
    counts = np.zeros(num_classes)
    r = [t.unsqueeze(0) for t in z_relays]
    remaining = n
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch, remaining); remaining -= bs
            rb = [t.repeat(bs, 1) for t in r]                       # fresh noise per row
            nr = [add_zscore_noise(t, sigma) for t in rb]
            p1 = torch.cat((nr[0], nr[1]), 1); p2 = torch.cat((nr[1], nr[2]), 1)
            p3 = torch.cat((nr[2], nr[3]), 1); p4 = torch.cat((nr[3], nr[0]), 1)
            for idx in model(p1, p2, p3, p4).argmax(1).cpu().numpy():
                counts[idx] += 1
    return counts.argmax()

def run(wrapper, model, zXte, y, device, eps, tag):
    preds, targets = [], []
    for i in tqdm(range(y.shape[0]), desc=tag, leave=False):
        z = [zXte[k][i] for k in range(4)]
        x = torch.stack([torch.stack([zk.real, zk.imag], -1) for zk in z], 0).unsqueeze(0).to(device)
        yi = y[i].unsqueeze(0).to(device)
        adv = fgsm(wrapper, x, yi, eps) if eps > 0 else x
        z_adv = [torch.complex(adv[0, k, :, 0], adv[0, k, :, 1]) for k in range(4)]
        preds.append(smooth_predict(model, z_adv, EVAL_SIGMA, N_SMOOTH))
        targets.append(y[i].item())
    acc = (np.array(preds) == np.array(targets)).mean() * 100
    mcc = matthews_corrcoef(targets, preds)
    print(f"[{tag}]  acc {acc:.2f}%  MCC {mcc:.4f}  F1 {f1_score(targets, preds, average='macro', zero_division=0):.4f}")
    return acc, mcc

def main():
    complex_dataset = Path("dataset3/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device.type.upper()} | model={MODEL_PATH.name} | z-scored | sigma={EVAL_SIGMA} | fair Linf eps={EPS}")

    with np.load(complex_dataset, allow_pickle=True) as data:
        Xtr = [torch.tensor(data[f'x_train_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
        Xte = [torch.tensor(data[f'x_test_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
        y_test = torch.tensor([mapping_classes(y.item()) for y in data['y_test']], dtype=torch.long)

    stats = [tuple(t.to(device) for t in st) for st in zscore_stats(Xtr)]  # same transform as training
    n_use = min(SUBSET_SIZE, Xte[0].shape[0])
    zXte = [zscore(Xte[k][:n_use].to(device), stats[k]) for k in range(4)]
    y = y_test[:n_use]

    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{MODEL_PATH}', map_location=device))
    model.eval()
    wrapper = ZWrapper(model).to(device).eval()

    print("-" * 56)
    acc_c, mcc_c = run(wrapper, model, zXte, y, device, 0.0, "clean-smoothed")
    acc_a, mcc_a = run(wrapper, model, zXte, y, device, EPS, f"fair Linf @{EPS}")
    print("-" * 56)
    print(f"clean {acc_c:.1f}%/MCC {mcc_c:.3f}  ->  attacked {acc_a:.1f}%/MCC {mcc_a:.3f}")
    # Robust only if the clean accuracy survives the sigma AND the MCC survives the attack.
    print("=> ROBUST" if acc_c >= 75 and mcc_a > 0.3 else "=> NOT ROBUST at this sigma")

if __name__ == "__main__":
    main()
