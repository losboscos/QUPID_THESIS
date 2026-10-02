import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from rtdl import FTTransformer
from BASELINES.MTL_LSTM.MTL_LSTM import MTLLSTM
from BASELINES.QH_DNN.QH_DNN import QHDNNModel
from r_pqnn import QUPIDModel, compute_minimum_dot_product, compute_local_dp_parameters, add_gaussian_noise
from to_complex import complex_partition

SCENARIO = "S1"  # S1, S3, S11, S12
SOURCES = ["FTTransformer", "MTL_LSTM", "QH_DNN"]
# Only eps = 0.05: the target is the L005 model
VARIANTS = ["fgsm_005", "pgd_005"]
EPS_OF = {"fgsm_005": 0.05, "pgd_005": 0.05}
EPSILONS = (0.05,)

ADV_DIR = Path(f"TRANSFER ATTACK/{SCENARIO}")
MODEL_PATH = Path(f"MODELS_R-QUPID/r_qupid_{SCENARIO}_L005.pth")
CLEAN_COMPLEX = Path(f"dataset3/scenarios_complex/data{SCENARIO[1:]} Sampled Scenarios_scaled_complex.npz")
ATTACK_BOUND, NOISE_PROBABILITY, EPS_T, DEL_T = 0.05, 0.1, 20.0, 1e-5
N_SMOOTH, SMOOTH_BATCH = 100, 128
RANDOM_SEED = 0
SUBSET = None          # None = all the samples in the npz
ONLY_FRAGILITY = False  # True = only section 5, without R-QUPID (fast)

TAG = "" if SCENARIO == "S1" else f"_{SCENARIO}"  # S1 files have no suffix
QUPID_PATH = Path("MODELS_QUPID/qupid_model_S1 (94.66% acc).pth" if SCENARIO == "S1"
                  else f"MODELS_QUPID/qupid_model_{SCENARIO}.pth")
CLASSICAL_CKPT = {s: Path(f"BASELINES/{s}/{s}_{SCENARIO}.pth") for s in SOURCES}

class FTTransformerWrapper(nn.Module):
    def __init__(self, ft_model):
        super().__init__()
        self.ft_model = ft_model

    def forward(self, x):
        return self.ft_model(x, None)

def load_classical(name, device):
    if name == "FTTransformer":
        raw = FTTransformer.make_default(n_num_features=120, cat_cardinalities=None, d_out=6)
        raw.load_state_dict(torch.load(CLASSICAL_CKPT[name], map_location=device))
        model = FTTransformerWrapper(raw)
    elif name == "MTL_LSTM":
        model = MTLLSTM(input_size=120, hidden_size=150, num_classes=6, dropout=0.3)
        model.load_state_dict(torch.load(CLASSICAL_CKPT[name], map_location=device))
    else:
        model = QHDNNModel()
        model.load_state_dict(torch.load(CLASSICAL_CKPT[name], map_location=device))
    return model.to(device).eval()

def eval_classical(model, x_120, y, device, batch=256):
    preds = []
    with torch.no_grad():
        for s in range(0, len(y), batch):
            xb = torch.tensor(x_120[s:s + batch], dtype=torch.float32).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    return (preds == y).mean() * 100, matthews_corrcoef(y, preds)

def eval_qupid(model, x_120, features, y, device, batch=128):
    """QUPID without defense: deterministic, no smoothing."""
    relays = to_relays(x_120, features, device)
    preds = []
    with torch.no_grad():
        for s in range(0, len(y), batch):
            e = s + batch
            p1 = torch.cat((relays[0][s:e], relays[1][s:e]), 1)
            p2 = torch.cat((relays[1][s:e], relays[2][s:e]), 1)
            p3 = torch.cat((relays[2][s:e], relays[3][s:e]), 1)
            p4 = torch.cat((relays[3][s:e], relays[0][s:e]), 1)
            preds.append(model(p1, p2, p3, p4).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    return (preds == y).mean() * 100, matthews_corrcoef(y, preds)

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
                for idx in self.base_classifier(p1, p2, p3, p4).argmax(dim=1).cpu().numpy():
                    counts[idx] += 1
        return counts.argmax()

def to_relays(x_120, features, device):
    """120 MinMax features -> 4 complex relays (16 values each)."""
    return [torch.tensor(complex_partition(x_120, features, p)[:, :16],
                         dtype=torch.complex64).to(device) for p in ("R1", "R2", "R3", "R4")]

def random_perturbation(x, eps, rng):
    """+-eps with random sign on every feature, then clamp to [0,1] (same L-inf budget as FGSM)."""
    signs = rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32)
    return np.clip(x + eps * signs, 0.0, 1.0)

def perturbation_stats(x_clean, x_adv, features):
    """L-inf and mean L2 norm of the perturbation on the features and on the complex values."""
    d = x_adv - x_clean
    attacker = {"linf": float(np.abs(d).max()),
                "l2": float(np.linalg.norm(d, axis=1).mean())}

    z_clean = np.concatenate([complex_partition(x_clean, features, p)[:, :16]
                              for p in ("R1", "R2", "R3", "R4")], axis=1)
    z_adv = np.concatenate([complex_partition(x_adv, features, p)[:, :16]
                            for p in ("R1", "R2", "R3", "R4")], axis=1)
    dz = z_adv - z_clean
    # L-inf over real and imaginary parts
    phasor = {"linf": float(np.maximum(np.abs(dz.real), np.abs(dz.imag)).max()),
              "l2": float(np.linalg.norm(dz.reshape(dz.shape[0], -1), axis=1).mean())}
    return attacker, phasor

def evaluate(smoothed, x_120, features, y, device, tag):
    relays = to_relays(x_120, features, device)
    preds = np.array([smoothed.predict([relays[k][i] for k in range(4)], N_SMOOTH, SMOOTH_BATCH)
                      for i in tqdm(range(y.shape[0]), desc=tag, leave=False)])
    return (preds == y).mean() * 100, matthews_corrcoef(y, preds), preds

def fragility(srcs, x_clean, randoms, features, y, device, rq_rows=None):
    """Section 5: the same random and transferred inputs on every target."""
    def adv_inputs(sources, eps):
        return [srcs[s][v].astype(np.float32)[:len(y)] for s in sources for v in VARIANTS
                if EPS_OF[v] == eps and v in srcs[s].files]

    def summarise(fn, sources):
        clean = fn(x_clean)
        out = [clean]
        for eps in EPSILONS:
            rnd = fn(randoms[eps])
            adv = [fn(xa) for xa in adv_inputs(sources, eps)]
            out += [rnd, (float(np.mean([a for a, _ in adv])), float(np.mean([m for _, m in adv])))]
        return out   # clean, then random and transfer for each eps

    table = {}
    for t in SOURCES:
        m = load_classical(t, device)
        # the white-box case (source == target) is excluded
        table[t] = summarise(lambda xx, m=m: eval_classical(m, xx, y, device),
                             [s for s in SOURCES if s != t])
        del m

    q = QUPIDModel().to(device)
    q.load_state_dict(torch.load(f'{QUPID_PATH}', map_location=device))
    q.eval()
    table["QUPID (no defense)"] = summarise(lambda xx: eval_qupid(q, xx, features, y, device), SOURCES)

    if rq_rows is not None:
        def rq_mean(eps):
            vals = [rq_rows[v][s] for v in VARIANTS if EPS_OF[v] == eps for s in SOURCES
                    if s in rq_rows[v]]
            return (float(np.mean([a for a, _ in vals])), float(np.mean([m for _, m in vals])))
        table["R-QUPID"] = [rq_rows["clean"]["-"]]
        for eps in EPSILONS:
            table["R-QUPID"] += [rq_rows[f"random {eps:.2f}"]["-"], rq_mean(eps)]

    cols = ["clean"] + [c for eps in EPSILONS for c in (f"random {eps:.2f}", f"transfer {eps:.2f}")]
    for k, kind in ((1, "MCC"), (0, "accuracy %")):
        print(f"\nFRAGILITY ACROSS TARGETS - {kind}  (transfer = mean over surrogates, FGSM and PGD)")
        print(f"{'target':<20}" + "".join(f"{c:>14}" for c in cols))
        print("-" * 90)
        for t, cells in table.items():
            fmt = (lambda v: f"{v:+.3f}") if k == 1 else (lambda v: f"{v:.2f}")
            print(f"{t:<20}" + "".join(f"{fmt(c[k]):>14}" for c in cells))

    print("\nMCC lost at eps 0.05:   to random noise   |  further, to the transferred direction")
    for t, cells in table.items():
        c, r, a = cells[0][1], cells[1][1], cells[2][1]
        print(f"  {t:<20}{c - r:>+14.3f}      |  {r - a:>+14.3f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device.type.upper()} | target R-QUPID {MODEL_PATH.name} | scenario {SCENARIO}")

    srcs = {s: np.load(ADV_DIR / f"transfer_adv_{s}{TAG}.npz", allow_pickle=True) for s in SOURCES}
    features = srcs[SOURCES[0]]["features"].tolist()
    y = srcs[SOURCES[0]]["y"]
    x_clean = srcs[SOURCES[0]]["x"].astype(np.float32)
    if SUBSET:
        y, x_clean = y[:SUBSET], x_clean[:SUBSET]
    print(f"{y.shape[0]} samples, {x_clean.shape[1]} features\n")

    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{MODEL_PATH}', map_location=device))
    model.eval()
    with np.load(CLEAN_COMPLEX, allow_pickle=True) as data:
        Xte = [torch.tensor(data[f'x_test_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
    P = [torch.cat((Xte[a], Xte[b]), dim=1).numpy() for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
    sigmas_sq = [compute_local_dp_parameters(EPS_T, DEL_T, NOISE_PROBABILITY,
                 compute_minimum_dot_product(p), delta_L=ATTACK_BOUND)[2] ** 2 for p in P]
    smoothed = SmoothQUPID(model, 6, sigmas_sq)

    # 2. perturbation size
    print("PERTURBATION ON BOTH SIDES OF THE ENCODER (mean per sample)")
    print(f"{'variant':<22}{'attacker Linf':>15}{'attacker L2':>13}{'phasor Linf':>13}{'phasor L2':>12}")
    print("-" * 75)
    rng = np.random.default_rng(RANDOM_SEED)
    randoms = {eps: random_perturbation(x_clean, eps, rng) for eps in EPSILONS}
    for eps, xr in randoms.items():
        a, p = perturbation_stats(x_clean, xr, features)
        print(f"{'random ' + f'{eps:.2f}':<22}{a['linf']:>15.4f}{a['l2']:>13.4f}"
              f"{p['linf']:>13.4f}{p['l2']:>12.4f}")
    for s in SOURCES:
        for v in VARIANTS:
            if v not in srcs[s].files:
                continue
            a, p = perturbation_stats(x_clean, srcs[s][v].astype(np.float32)[:y.shape[0]], features)
            print(f"{s + ' ' + v:<22}{a['linf']:>15.4f}{a['l2']:>13.4f}"
                  f"{p['linf']:>13.4f}{p['l2']:>12.4f}")
    print()

    if ONLY_FRAGILITY:
        fragility(srcs, x_clean, randoms, features, y, device)
        return

    # 1. and 4. transfer vs random control, FGSM vs PGD
    rows = {}
    acc, mcc, preds_clean = evaluate(smoothed, x_clean, features, y, device, "clean")
    rows["clean"] = {"-": (acc, mcc)}
    print(f"clean-smoothed: {acc:.2f}%  MCC {mcc:+.4f}")

    for eps, xr in randoms.items():
        acc, mcc, _ = evaluate(smoothed, xr, features, y, device, f"random {eps}")
        rows[f"random {eps:.2f}"] = {"-": (acc, mcc)}
        print(f"random +-{eps:.2f}:   {acc:.2f}%  MCC {mcc:+.4f}")

    preds_ref = None
    for v in VARIANTS:
        rows[v] = {}
        for s in SOURCES:
            if v not in srcs[s].files:
                continue
            acc, mcc, preds = evaluate(smoothed, srcs[s][v].astype(np.float32)[:y.shape[0]],
                                       features, y, device, f"{s} {v}")
            rows[v][s] = (acc, mcc)
            if v == "fgsm_005" and s == SOURCES[0]:
                preds_ref = preds
            print(f"  {s:<15}{v:<10}{acc:>8.2f}%  MCC {mcc:+.4f}")

    # summary
    print("\n" + "=" * 78)
    print(f"{'variant':<14}" + "".join(f"{s:>21}" for s in SOURCES))
    print("-" * 78)
    for key, per_source in rows.items():
        line = f"{key:<14}"
        for s in SOURCES:
            cell = per_source.get(s) or per_source.get("-")
            line += f"{f'{cell[0]:.2f}% / {cell[1]:+.3f}':>21}" if cell else f"{'-':>21}"
        print(line)
    print("=" * 78)

    # 3. confusion matrix
    if preds_ref is not None:
        print(f"\nPER-CLASS under transfer ({SOURCES[0]} fgsm_005), rows = real, cols = predicted")
        cm = confusion_matrix(y, preds_ref, labels=list(range(6)))
        print("         " + "".join(f"{i:>7}" for i in range(6)) + f"{'recall':>10}")
        for i in range(6):
            support = cm[i].sum()
            rec = cm[i, i] / support if support else 0.0
            print(f"  real {i}  " + "".join(f"{int(cm[i, j]):>7}" for j in range(6)) + f"{rec:>10.3f}")

    # 5. all targets
    fragility(srcs, x_clean, randoms, features, y, device, rq_rows=rows)

if __name__ == "__main__":
    main()
