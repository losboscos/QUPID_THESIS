import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
from pathlib import Path
from collections import Counter
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from r_pqnn import QUPIDModel

SCENARIO = "S1"            # S1, S3, S11, S12
EPS = 0.05                 # training budget (L-inf on the 120 MinMax features)
TRAIN_PGD_STEPS = 7        # PGD steps used to craft the training examples

EPOCHS = 200               # same as pqnn.py
LR = 0.01                  # same as pqnn.py
BATCH_SIZE = 128           # same as pqnn.py
EPS_RAMP_EPOCHS = 50       # eps grows linearly from 0 to EPS over these epochs
CLEAN_WEIGHT = 0.5         # loss = 0.5 * CE(clean) + 0.5 * CE(adversarial)
VAL_FRACTION = 0.15
VAL_SEED = 42
VAL_EVERY = 10             # validation every 10 epochs
VAL_PGD_STEPS = 10
TEST_PGD_STEPS = 20
EVAL_EPS = sorted({EPS, 0.02, 0.05})   # budgets evaluated on the test set
SEED = 0

EPS_TAG = str(EPS).replace('0.', '0')
DATASET = Path(f"dataset3/scenarios_complex/data{SCENARIO[1:]} Sampled Scenarios_scaled_complex.npz")
FEATURES_DATASET = Path(f"dataset3/scenarios/data{SCENARIO[1:]} Sampled Scenarios_scaled.npz")
if SCENARIO == "S1":
    REFERENCE_PATH = Path("MODELS_QUPID/qupid_model_S1 (94.66% acc).pth")
else:
    REFERENCE_PATH = Path(f"MODELS_QUPID/qupid_model_{SCENARIO}.pth")
MODEL_OUT = Path(f"ADVERSARIAL TRAINING/MODELS_AT/at_qupid_{SCENARIO}_E{EPS_TAG}_pgd{TRAIN_PGD_STEPS}_feat.pth")   # best validation PGD MCC
MODEL_LAST = MODEL_OUT.with_name(MODEL_OUT.stem + "_last.pth")   # end of training

def make_partitions(relays):
    """The 4 partitions of QUPID: each circuit receives two neighbouring relays."""
    r1, r2, r3, r4 = relays
    partition_1 = torch.cat((r1, r2), dim=1)
    partition_2 = torch.cat((r2, r3), dim=1)
    partition_3 = torch.cat((r3, r4), dim=1)
    partition_4 = torch.cat((r4, r1), dim=1)
    return partition_1, partition_2, partition_3, partition_4

def print_history(history, best_epoch):
    """Validation table, one row every VAL_EVERY epochs; * marks the saved checkpoint."""
    print("\n" + "=" * 72)
    print(f"{'epoch':>6}{'eps':>8}{'val clean acc':>15}{'MCC':>8}{'val PGD acc':>14}{'MCC':>8}")
    print("-" * 72)
    for ep, eps, clean_acc, clean_mcc, adv_acc, adv_mcc in history:
        mark = "  *" if ep == best_epoch else ""
        print(f"{ep:>6}{eps:>8.3f}{clean_acc:>14.2f}%{clean_mcc:>+8.3f}{adv_acc:>13.2f}%{adv_mcc:>+8.3f}{mark}")
    print("=" * 72)

def partition_plan(features, prefix):
    plan = []
    processed_features = set()
    log_name = f"relay{prefix[1:]}_log"
    indexes_pmu = [i for i, col in enumerate(features) if col.startswith(prefix) or col == log_name]

    for i in indexes_pmu:
        if i in processed_features:
            continue
        col_name = features[i]

        # Phasor (voltage or current): the magnitude is the PM column with the same name
        if 'PA' in col_name and not col_name.endswith(':Z') and not col_name.endswith(':ZH'):
            magnitude_name = col_name.replace('PA', 'PM').replace('VH', 'V').replace('IH', 'I')
            magnitude_index = features.index(magnitude_name)
            plan.append(("phasor", magnitude_index, i))
            processed_features.update((i, magnitude_index))
            continue

        # Impedance: magnitude in PA:Z, angle in PA:ZH
        if 'PA' in col_name and col_name.endswith(':Z'):
            angle_index = features.index(col_name.replace('PA:Z', 'PA:ZH'))
            plan.append(("phasor", i, angle_index))
            processed_features.update((i, angle_index))
            continue

        # Magnitudes are taken together with their phase
        if 'PM' in col_name or col_name.endswith(':ZH'):
            continue

        # Real scalar (F, DF, S or the relay log)
        plan.append(("scalar", i, None))
        processed_features.add(i)
    return plan

def encode_features(x, plans):
    """
    Differentiable version of to_complex.py: 120 MinMax features -> 4 complex relays (first 16 values).
    Phasors: z = PM * exp(i * 2pi * PA). Scalars: z = value + 0j.
    """
    relays = []
    for prefix in ("R1", "R2", "R3", "R4"):
        columns = []
        for kind, a, b in plans[prefix]:
            if kind == "phasor":
                columns.append(x[:, a] * torch.exp(1j * 2.0 * math.pi * x[:, b]))
            else:
                columns.append(torch.complex(x[:, a], torch.zeros_like(x[:, a])))
        relays.append(torch.stack(columns, dim=1).to(torch.complex64))
    return relays

def feature_logits(model, plans, x):
    return model(*make_partitions(encode_features(x, plans)))

def feature_pgd(model, plans, x, y, eps, steps, random_start):
    """L-inf PGD on the MinMax features, step 2.5*eps/steps, clamped to [0,1]."""
    alpha = 2.5 * eps / steps
    if random_start:
        delta = (torch.rand_like(x) * 2.0 - 1.0) * eps
    else:
        delta = torch.zeros_like(x)
    for _ in range(steps):
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach().requires_grad_(True)
        loss = F.cross_entropy(feature_logits(model, plans, x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        delta = torch.clamp(delta + alpha * grad.sign(), -eps, eps)
    return torch.clamp(x + delta, 0.0, 1.0).detach()

def feature_fgsm(model, plans, x, y, eps):
    """Single-step L-inf FGSM on the MinMax features, clamped to [0,1]."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(feature_logits(model, plans, x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return torch.clamp(x + eps * grad.sign(), 0.0, 1.0).detach()

def feature_predict(model, plans, x, device):
    """Predicted classes of the whole set, in batches."""
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            logits = feature_logits(model, plans, x[start:start + BATCH_SIZE].to(device))
            preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()

def feature_perturb(model, plans, x, y, kind, eps, device, gen=None):
    """Perturbed copy of the whole set: kind is "random", "FGSM" or "PGD"."""
    x_pert = []
    for start in range(0, len(x), BATCH_SIZE):
        b_x = x[start:start + BATCH_SIZE].to(device)
        b_y = y[start:start + BATCH_SIZE].to(device)
        if kind == "random":
            # +-eps with a random sign on every feature
            signs = torch.randint(0, 2, b_x.shape, generator=gen).float() * 2.0 - 1.0
            b_adv = torch.clamp(b_x + eps * signs.to(device), 0.0, 1.0)
        elif kind == "FGSM":
            b_adv = feature_fgsm(model, plans, b_x, b_y, eps)
        else:
            b_adv = feature_pgd(model, plans, b_x, b_y, eps, TEST_PGD_STEPS, random_start=False)
        x_pert.append(b_adv.cpu())
    return torch.cat(x_pert)

def feature_validate(model, plans, x, y, device):
    """Accuracy and MCC on the validation split, clean and under PGD at the full training budget."""
    model.eval()
    x_adv = []
    for start in range(0, len(x), BATCH_SIZE):
        b_x = x[start:start + BATCH_SIZE].to(device)
        b_y = y[start:start + BATCH_SIZE].to(device)
        x_adv.append(feature_pgd(model, plans, b_x, b_y, EPS, VAL_PGD_STEPS, random_start=False).cpu())
    x_adv = torch.cat(x_adv)

    y_true = y.numpy()
    preds_clean = feature_predict(model, plans, x, device)
    preds_adv = feature_predict(model, plans, x_adv, device)
    model.train()
    clean_acc = (preds_clean == y_true).mean() * 100
    adv_acc = (preds_adv == y_true).mean() * 100
    return clean_acc, matthews_corrcoef(y_true, preds_clean), adv_acc, matthews_corrcoef(y_true, preds_adv)

def main():
    if MODEL_OUT.exists():
        raise FileExistsError(f"{MODEL_OUT} already exists: delete or rename it before retraining.")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with np.load(FEATURES_DATASET, allow_pickle=True) as data:
        X_train = torch.tensor(data['x_train'], dtype=torch.float32)
        y_train = torch.tensor([mapping_classes(y.item()) for y in data['y_train']], dtype=torch.long)
        X_test = torch.tensor(data['x_test'], dtype=torch.float32)
        y_test = torch.tensor([mapping_classes(y.item()) for y in data['y_test']], dtype=torch.long)
        features = [str(c) for c in data['features'].tolist()]

    # Columns used for each relay (the first 16, as every QUPID script)
    plans = {}
    for prefix in ("R1", "R2", "R3", "R4"):
        plans[prefix] = partition_plan(features, prefix)[:16]

    # The rebuilt phasors must be the ones QUPID was trained on (stored complex npz)
    with np.load(DATASET, allow_pickle=True) as data:
        stored = [data[f'x_test_r{i}'][:, :16] for i in range(1, 5)]
        stored_labels = np.array([mapping_classes(int(y)) for y in data['y_test']])
        if not np.array_equal(stored_labels, y_test.numpy()):
            raise RuntimeError("features and complex npz files do not contain the same rows")
    rebuilt = encode_features(X_test.double(), plans)
    diff = max(float(np.abs(s - r.numpy()).max()) for s, r in zip(stored, rebuilt))
    print(f"encoding check: max |stored - rebuilt| = {diff:.2e}")
    if diff > 1e-4:
        raise RuntimeError("the differentiable encoding does not reproduce to_complex.py")

    # Validation split taken from the training set: same as at_baseline.py
    idx_fit, idx_val = train_test_split(np.arange(len(y_train)), test_size=VAL_FRACTION,
                                        stratify=y_train.numpy(), random_state=VAL_SEED)
    X_fit, y_fit = X_train[idx_fit], y_train[idx_fit]
    X_val, y_val = X_train[idx_val], y_train[idx_val]

    print(f"Adversarial training QUPID | {SCENARIO} | features | eps={EPS} (ramp {EPS_RAMP_EPOCHS} ep) | "
          f"PGD-{TRAIN_PGD_STEPS}-RS | clean weight {CLEAN_WEIGHT} | {device.type.upper()}")
    print(f"fit {len(y_fit)} / validation {len(y_val)} -> {MODEL_OUT}")

    dataset = torch.utils.data.TensorDataset(X_fit, y_fit)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = QUPIDModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_mcc = -float('inf')
    best_epoch = None
    history = []
    model.train()

    for epoch in range(EPOCHS):
        # eps grows linearly during the first EPS_RAMP_EPOCHS, then stays at EPS
        eps_t = EPS * min(1.0, epoch / EPS_RAMP_EPOCHS)
        total_loss = 0.0
        total_samples = 0
        correct_clean = 0
        correct_adv = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} eps={eps_t:.3f}", leave=False)
        for b_x, b_y in progress_bar:
            b_x, b_y = b_x.to(device), b_y.to(device)

            # Adversarial version of the batch (at epoch 0 eps is 0: plain clean training)
            if eps_t > 0:
                b_adv = feature_pgd(model, plans, b_x, b_y, eps_t, TRAIN_PGD_STEPS, random_start=True)
            else:
                b_adv = b_x

            optimizer.zero_grad()
            logits_clean = feature_logits(model, plans, b_x)
            if eps_t > 0:
                logits_adv = feature_logits(model, plans, b_adv)
            else:
                logits_adv = logits_clean
            # Half of the loss on the clean batch, half on the adversarial one
            loss = CLEAN_WEIGHT * F.cross_entropy(logits_clean, b_y) + (1.0 - CLEAN_WEIGHT) * F.cross_entropy(logits_adv, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b_y.size(0)
            total_samples += b_y.size(0)
            correct_clean += (logits_clean.argmax(dim=1) == b_y).sum().item()
            correct_adv += (logits_adv.argmax(dim=1) == b_y).sum().item()

        line = (f"Epoch {epoch + 1:>3} | eps {eps_t:.3f} | loss {total_loss / total_samples:.4f} | "
                f"train clean {correct_clean / total_samples * 100:.2f}% | train adv {correct_adv / total_samples * 100:.2f}%")

        if (epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            val_clean_acc, val_clean_mcc, val_adv_acc, val_adv_mcc = feature_validate(model, plans, X_val, y_val, device)
            history.append((epoch + 1, eps_t, val_clean_acc, val_clean_mcc, val_adv_acc, val_adv_mcc))
            line += f" || val clean {val_clean_acc:.2f}% / {val_clean_mcc:+.3f} | val PGD {val_adv_acc:.2f}% / {val_adv_mcc:+.3f}"
            # The checkpoint is chosen only once the ramp is over, on the validation PGD MCC
            if epoch >= EPS_RAMP_EPOCHS and val_adv_mcc > best_mcc:
                best_mcc = val_adv_mcc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), MODEL_OUT)
                line += "  <- saved"
        print(line)

    torch.save(model.state_dict(), MODEL_LAST)

    print_history(history, best_epoch)
    print(f"best validation PGD MCC {best_mcc:+.3f} at epoch {best_epoch}")

    # Test set: AT model (best and last) next to QUPID without adversarial training
    models = {}
    for name, path in (("AT best", MODEL_OUT), ("AT last", MODEL_LAST), ("QUPID", REFERENCE_PATH)):
        if path.exists():
            m = QUPIDModel().to(device)
            m.load_state_dict(torch.load(f'{path}', map_location=device))
            models[name] = m.eval()

    y_true = y_test.numpy()
    rows = {}
    for name, m in models.items():
        results = {"clean": feature_predict(m, plans, X_test, device)}
        for eps in EVAL_EPS:
            gen = torch.Generator().manual_seed(SEED)   # same random signs for every model
            x_random = feature_perturb(m, plans, X_test, y_test, "random", eps, device, gen)
            x_fgsm = feature_perturb(m, plans, X_test, y_test, "FGSM", eps, device)
            x_pgd = feature_perturb(m, plans, X_test, y_test, "PGD", eps, device)
            results[f"random {eps:.2f}"] = feature_predict(m, plans, x_random, device)
            results[f"FGSM {eps:.2f}"] = feature_predict(m, plans, x_fgsm, device)
            results[f"PGD-20 {eps:.2f}"] = feature_predict(m, plans, x_pgd, device)
        rows[name] = results

    names = list(rows)
    print(f"\nTEST SET ({len(y_true)} samples), attacks on the MinMax features")
    print(f"{'condition':<14}" + "".join(f"{n + ' acc':>22}{'MCC':>8}" for n in names))
    print("-" * (14 + 30 * len(names)))
    for condition in rows[names[0]]:
        line = f"{condition:<14}"
        for n in names:
            acc = (rows[n][condition] == y_true).mean() * 100
            mcc = matthews_corrcoef(y_true, rows[n][condition])
            line += f"{acc:>21.2f}%{mcc:>+8.3f}"
        print(line)

    # Verdict with criteria fixed before the runs (same as at_baseline.py)
    key = "AT best" if "AT best" in rows else "AT last"
    at = rows[key]
    clean_mcc = matthews_corrcoef(y_true, at["clean"])
    fgsm_mcc = matthews_corrcoef(y_true, at[f"FGSM {EPS:.2f}"])
    pgd_mcc = matthews_corrcoef(y_true, at[f"PGD-20 {EPS:.2f}"])
    top_share = max(Counter(at["clean"].tolist()).values()) / len(y_true)

    print(f"\nVERDICT ({key} at eps {EPS})")
    if clean_mcc < 0.2 or top_share > 0.9:
        print(f"  COLLAPSE: clean MCC {clean_mcc:+.3f}, largest class takes {top_share:.0%} of predictions")
    elif fgsm_mcc - pgd_mcc > 0.3 and pgd_mcc < 0.1:
        print(f"  MASKING SIGNATURE: FGSM MCC {fgsm_mcc:+.3f} vs PGD MCC {pgd_mcc:+.3f}")
    elif pgd_mcc > 0.3:
        print(f"  ROBUST: PGD-20 MCC {pgd_mcc:+.3f} (clean MCC {clean_mcc:+.3f})")
    else:
        print(f"  NOT ROBUST: clean MCC {clean_mcc:+.3f} | FGSM {fgsm_mcc:+.3f} | PGD-20 {pgd_mcc:+.3f}")

if __name__ == "__main__":
    main()
