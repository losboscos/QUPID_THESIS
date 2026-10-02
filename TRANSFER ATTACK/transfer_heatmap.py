import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from rtdl import FTTransformer
from BASELINES.MTL_LSTM.MTL_LSTM import MTLLSTM
from BASELINES.QH_DNN.QH_DNN import QHDNNModel
from r_pqnn import QUPIDModel, compute_minimum_dot_product, compute_local_dp_parameters, add_gaussian_noise
from to_complex import complex_partition

SOURCES = ["FTTransformer", "MTL_LSTM", "QH_DNN"]
TARGETS = ["FTTransformer", "MTL_LSTM", "QH_DNN", "QUPID", "R-QUPID"]
VARIANTS = ["fgsm_005", "pgd_005"]  # add "fgsm_010","pgd_010" for the eps=0.1 pair
NUM_FEATURES, NUM_CLASSES = 120, 6

SCENARIO = "S12"  # S1, S3, S11, S12
ADV_DIR = Path(f"TRANSFER ATTACK/{SCENARIO}")
RESULTS = Path(f"TRANSFER ATTACK/results/{SCENARIO}")
TAG = "" if SCENARIO == "S1" else f"_{SCENARIO}"  # S1 files have no suffix

CKPT = {
    "FTTransformer": Path(f"BASELINES/FTTransformer/FTTransformer_{SCENARIO}.pth"),
    "MTL_LSTM": Path(f"BASELINES/MTL_LSTM/MTL_LSTM_{SCENARIO}.pth"),
    "QH_DNN": Path(f"BASELINES/QH_DNN/QH_DNN_{SCENARIO}.pth"),
    "QUPID": Path("MODELS_QUPID/qupid_model_S1 (94.66% acc).pth" if SCENARIO == "S1"
                  else f"MODELS_QUPID/qupid_model_{SCENARIO}.pth"),
    "R-QUPID": Path(f"MODELS_R-QUPID/r_qupid_{SCENARIO}_L005.pth"),
}
CLEAN_COMPLEX = Path(f"dataset3/scenarios_complex/data{SCENARIO[1:]} Sampled Scenarios_scaled_complex.npz")
ATTACK_BOUND, NOISE_PROBABILITY, EPS_T, DEL_T = 0.05, 0.1, 20.0, 1e-5
N_SMOOTH, SMOOTH_BATCH = 100, 128

class FTTransformerWrapper(nn.Module):
    def __init__(self, ft_model):
        super().__init__()
        self.ft_model = ft_model

    def forward(self, x):
        return self.ft_model(x, None)

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
                p1 = add_gaussian_noise(torch.cat((b[0], b[1]), dim=1), self.sigmas_sq[0])
                p2 = add_gaussian_noise(torch.cat((b[1], b[2]), dim=1), self.sigmas_sq[1])
                p3 = add_gaussian_noise(torch.cat((b[2], b[3]), dim=1), self.sigmas_sq[2])
                p4 = add_gaussian_noise(torch.cat((b[3], b[0]), dim=1), self.sigmas_sq[3])
                logits = self.base_classifier(p1, p2, p3, p4)
                for idx in logits.argmax(dim=1).cpu().numpy():
                    counts[idx] += 1
        return counts.argmax()

def load_classical(name, device):
    if name == "FTTransformer":
        raw = FTTransformer.make_default(n_num_features=NUM_FEATURES, cat_cardinalities=None, d_out=NUM_CLASSES)
        raw.load_state_dict(torch.load(CKPT[name], map_location=device))
        model = FTTransformerWrapper(raw)
    elif name == "MTL_LSTM":
        model = MTLLSTM(input_size=NUM_FEATURES, hidden_size=150, num_classes=NUM_CLASSES, dropout=0.3)
        model.load_state_dict(torch.load(CKPT[name], map_location=device))
    elif name == "QH_DNN":
        model = QHDNNModel()
        model.load_state_dict(torch.load(CKPT[name], map_location=device))
    return model.to(device).eval()

def eval_classical(model, adv_X, y, device):
    with torch.no_grad():
        preds = model(torch.tensor(adv_X, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
    return (preds == y).mean() * 100, matthews_corrcoef(y, preds)

def eval_rqupid(smoothed, adv_X, features, y, device):
    relays = [torch.tensor(complex_partition(adv_X, features, p)[:, :16], dtype=torch.complex64).to(device)
              for p in ("R1", "R2", "R3", "R4")]
    preds = np.array([smoothed.predict([relays[k][i] for k in range(4)], N_SMOOTH, SMOOTH_BATCH)
                      for i in range(y.shape[0])])
    return (preds == y).mean() * 100, matthews_corrcoef(y, preds)

def eval_qupid(model, adv_X, features, y, device, batch=128):
    """Clean QUPID target (no DP defense) -> deterministic forward, no smoothing."""
    relays = [torch.tensor(complex_partition(adv_X, features, p)[:, :16], dtype=torch.complex64).to(device)
              for p in ("R1", "R2", "R3", "R4")]
    preds = []
    with torch.no_grad():
        for s in range(0, y.shape[0], batch):
            e = s + batch
            p1 = torch.cat((relays[0][s:e], relays[1][s:e]), 1)
            p2 = torch.cat((relays[1][s:e], relays[2][s:e]), 1)
            p3 = torch.cat((relays[2][s:e], relays[3][s:e]), 1)
            p4 = torch.cat((relays[3][s:e], relays[0][s:e]), 1)
            preds.append(model(p1, p2, p3, p4).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    return (preds == y).mean() * 100, matthews_corrcoef(y, preds)

def build_qupid(device):
    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(CKPT["QUPID"], map_location=device))
    model.eval()
    return model

def build_rqupid(device):
    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(CKPT["R-QUPID"], map_location=device))
    model.eval()
    with np.load(CLEAN_COMPLEX, allow_pickle=True) as data:
        Xte = [torch.tensor(data[f'x_test_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
    P = [torch.cat((Xte[a], Xte[b]), dim=1).numpy() for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
    sigmas_sq = [compute_local_dp_parameters(EPS_T, DEL_T, NOISE_PROBABILITY,
                 compute_minimum_dot_product(p), delta_L=ATTACK_BOUND)[2] ** 2 for p in P]
    return SmoothQUPID(model, NUM_CLASSES, sigmas_sq)

def heatmap(matrix, variant, kind, fname):
    diverging = kind == "MCC"
    cmap = "RdBu" if diverging else "viridis"
    vmin, vmax = (-0.7, 0.7) if diverging else (0, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(TARGETS, rotation=30, ha="right")
    ax.set_yticks(range(len(SOURCES)))
    ax.set_yticklabels(SOURCES)
    ax.set_xlabel("target (victim)")
    ax.set_ylabel("surrogate (source)")
    ax.set_title(f"Transfer {kind}  |  {variant}")
    for i in range(len(SOURCES)):
        for j in range(len(TARGETS)):
            wb = (SOURCES[i] == TARGETS[j])  # classical diagonal = white-box
            txt = f"{matrix[i, j]:.2f}" + ("*" if wb else "")
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (diverging and abs(matrix[i, j]) > 0.35) or
                    (not diverging and matrix[i, j] < 45) else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label=kind)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  figure -> {fname}   (* = white-box, source==target)")

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()} | scenario {SCENARIO} | sources={SOURCES} | targets={TARGETS}")

    classical = {t: load_classical(t, device) for t in TARGETS if t not in ("R-QUPID", "QUPID")}
    qupid = build_qupid(device)
    smoothed = build_rqupid(device)

    # source npz files (same y/features across all -> read once)
    srcs = {s: np.load(ADV_DIR / f"transfer_adv_{s}{TAG}.npz", allow_pickle=True) for s in SOURCES}
    features = srcs[SOURCES[0]]["features"].tolist()
    y = srcs[SOURCES[0]]["y"]

    # clean reference per target (source-independent: same clean subset "x")
    print("\nclean reference (no attack):")
    x_clean = srcs[SOURCES[0]]["x"]
    for t in TARGETS:
        if t == "R-QUPID":
            acc, mcc = eval_rqupid(smoothed, x_clean, features, y, device)
        elif t == "QUPID":
            acc, mcc = eval_qupid(qupid, x_clean, features, y, device)
        else:
            acc, mcc = eval_classical(classical[t], x_clean, y, device)
        print(f"  {t:<14} acc={acc:6.2f}%  MCC={mcc:+.4f}")

    for variant in VARIANTS:
        acc_m = np.zeros((len(SOURCES), len(TARGETS)))
        mcc_m = np.zeros((len(SOURCES), len(TARGETS)))
        print(f"\n{'='*60}\nVARIANT: {variant}\n{'='*60}")
        for i, s in enumerate(SOURCES):
            adv_X = srcs[s][variant].astype(np.float32)
            for j, t in enumerate(TARGETS):
                if t == "R-QUPID":
                    acc, mcc = eval_rqupid(smoothed, adv_X, features, y, device)
                elif t == "QUPID":
                    acc, mcc = eval_qupid(qupid, adv_X, features, y, device)
                else:
                    acc, mcc = eval_classical(classical[t], adv_X, y, device)
                acc_m[i, j], mcc_m[i, j] = acc, mcc
                mark = " (white-box)" if s == t else ""
                print(f"  {s:<14} -> {t:<14} acc={acc:6.2f}%  MCC={mcc:+.4f}{mark}")

        print(f"\n  ACCURACY matrix (rows=source, cols=target):\n           " +
              "".join(f"{t[:10]:>12}" for t in TARGETS))
        for i, s in enumerate(SOURCES):
            print(f"  {s:<10}" + "".join(f"{acc_m[i, j]:>11.2f}%" for j in range(len(TARGETS))))
        print(f"\n  MCC matrix:\n           " + "".join(f"{t[:10]:>12}" for t in TARGETS))
        for i, s in enumerate(SOURCES):
            print(f"  {s:<10}" + "".join(f"{mcc_m[i, j]:>+12.4f}" for j in range(len(TARGETS))))

        heatmap(acc_m, f"{SCENARIO} {variant}", "accuracy", RESULTS / f"heatmap_{variant}_acc_{SCENARIO}.png")
        heatmap(mcc_m, f"{SCENARIO} {variant}", "MCC", RESULTS / f"heatmap_{variant}_mcc_{SCENARIO}.png")

if __name__ == "__main__":
    main()
