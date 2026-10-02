import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # repo root on sys.path
from dataset import mapping_classes
from pathlib import Path
import numpy as np
from rtdl import FTTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from collections import Counter
import math

SCENARIO = "S1"   # S1, S3, S11, S12
SIGMAS_SQ = [0.4]
MODELS_DIR = Path("BASELINES/FTTransformer/Robust FTTransformer")

def model_path(sigma):
    if SCENARIO == "S1":
        return MODELS_DIR / f"r_FTTransformer_s{sigma} ({round(sigma ** 2, 2):g}).pth"
    return MODELS_DIR / f"r_FTTransformer_{SCENARIO}_s{sigma} ({sigma ** 2:.2f}).pth"

def add_gaussian_noise(x_tensor, sigma):
    # Z-scored features live roughly in [-4, 4], NOT [0, 1]: no clamp here.
    if sigma <= 0.0:
        return x_tensor
    return x_tensor + torch.randn_like(x_tensor) * sigma

def report(header, all_targets, all_preds, all_probs):
    """Same output block we have always used."""
    accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_targets))
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_targets, all_preds)
    g_mean = geometric_mean_score(all_targets, all_preds, average='macro')
    try:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = 0.0

    print(header)
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"G-Mean:    {g_mean:.4f}")

    pred_counts = Counter(all_preds)
    true_counts = Counter(all_targets)
    print("\nDistribution of Predictions vs Reality:")
    for class_idx in range(6):
        p_count = pred_counts.get(class_idx, 0)
        t_count = true_counts.get(class_idx, 0)
        print(f"Class {class_idx}: Predicted {p_count} times | (In reality there were {t_count})")
    print("-" * 60 + "\n")
    return mcc

@torch.no_grad()
def evaluate(model, loader, sigma, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(add_gaussian_noise(x_batch, sigma), None)
        total_loss += F.cross_entropy(logits, y_batch).item()
        probs = F.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(y_batch.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return total_loss / len(loader), all_targets, all_preds, all_probs

def main():
    if SCENARIO not in ("S1", "S3", "S11", "S12"):
        raise ValueError(f"SCENARIO must be 'S1', 'S3', 'S11' or 'S12', not {SCENARIO!r}")
    for s2 in SIGMAS_SQ:
        train(round(math.sqrt(s2), 4))

def train(TRAIN_SIGMA):
    out_name = model_path(TRAIN_SIGMA)
    if out_name.exists():
        print(f"{out_name} already exists: skipped")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {SCENARIO} | TRAIN_SIGMA={TRAIN_SIGMA} (sigma^2={TRAIN_SIGMA**2:.3f}) | z-scored features")

    dataset = Path(f"dataset3/scenarios/data{SCENARIO[1:]} Sampled Scenarios_scaled.npz")
    with np.load(dataset, allow_pickle=True) as data:
        X_all = data['x_train'].astype(np.float32)
        y_all = np.array([mapping_classes(y.item()) for y in data['y_train']], dtype=np.int64)

    # Z-score using the FULL train statistics (same convention as fig3_reproduce.py)
    mu = X_all.mean(axis=0)
    sd = X_all.std(axis=0) + 1e-8
    X_all = (X_all - mu) / sd

    # Stratified train/validation split for honest model selection
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val), batch_size=256, shuffle=False)

    model = FTTransformer.make_default(
        n_num_features=120,
        cat_cardinalities=None,
        d_out=6
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    EPOCHS = 200
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            b_x_noisy = add_gaussian_noise(x_batch, TRAIN_SIGMA)

            optimizer.zero_grad()
            logits = model(b_x_noisy, None)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(y_batch.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        report(f"\n--- End of epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.4f} ---",
               all_targets, all_preds, all_probs)

        val_loss, v_targets, v_preds, v_probs = evaluate(model, val_loader, TRAIN_SIGMA, device)
        report(f"--- VALIDATION (epoch {epoch+1}) | Avg Loss: {val_loss:.4f} ---",
               v_targets, v_preds, v_probs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_name)
            print(f"New best model saved to {out_name} (val loss {best_val_loss:.4f})\n")

if __name__ == "__main__":
    main()
