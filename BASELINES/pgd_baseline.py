import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
from pathlib import Path
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from rtdl import FTTransformer
from BASELINES.MTL_LSTM.MTL_LSTM import MTLLSTM
from BASELINES.QH_DNN.QH_DNN import QHDNNModel
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score

SCENARIO = "S1"          # S1, S3, S11, S12: selects the dataset
MODEL_PATH = Path("BASELINES/FTTransformer/Robust FTTransformer/r_FTTransformer_s0.6325 (0.4).pth")
STANDARDIZE = True      # True for z-scored models; False for [0,1] clean baselines
PGD_EPS = 0.1           
PGD_STEPS = 20           
SMOOTH_SIGMA = 0.6325    
N_SMOOTH = 100
SUBSET_SIZE = 500
NUM_FEATURES = 120
NUM_CLASSES = 6

def add_noise(x, sigma, clamp01):
    # Clamp to [0,1] ONLY on MinMax features (not on z-scored features).
    if sigma <= 0.0:
        return x
    x_noisy = x + torch.randn_like(x) * sigma
    if clamp01:
        x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
    return x_noisy

def pgd_attack(model, x, y, eps, steps, clamp01):
    # L-inf PGD, eps DIRECT (as fgsm_attack in fgsm_baseline.py), projected on the eps-ball at every step.
    alpha = 2.5 * eps / steps
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = x_orig + torch.clamp(x_adv - x_orig, -eps, eps)
        if clamp01:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()

class SmoothBaseline:
    """Randomized smoothing at test time: majority vote over N noisy forward passes."""
    def __init__(self, base_model: nn.Module, sigma: float, clamp01: bool, num_classes: int = 6):
        self.base_model = base_model
        self.sigma = sigma
        self.clamp01 = clamp01
        self.num_classes = num_classes

    def predict_batch(self, b_x: torch.Tensor, n: int = 100):
        self.base_model.eval()
        batch_size = b_x.size(0)
        counts = np.zeros((batch_size, self.num_classes), dtype=float)

        with torch.no_grad():
            for _ in range(n):
                b_x_noisy = add_noise(b_x, self.sigma, self.clamp01)
                logits = self.base_model(b_x_noisy)
                preds = logits.argmax(dim=1).cpu().numpy()
                for i, p in enumerate(preds):
                    counts[i, p] += 1.0

        best_preds = counts.argmax(axis=1)
        probs_dist = counts / n
        return best_preds, probs_dist

class FTTransformerWrapper(nn.Module):
    """Adapter so FTTransformer exposes the standard model(x) interface."""
    def __init__(self, ft_model):
        super().__init__()
        self.ft_model = ft_model

    def forward(self, x):
        return self.ft_model(x, None)

def main():
    dataset_path = Path(f"dataset3/scenarios/data{SCENARIO[1:]} Sampled Scenarios_scaled.npz")
    # the model must belong to the same scenario (S1 files have no scenario tag in the name)
    tag = re.search(r"_(S\d+)[_.]", MODEL_PATH.name)
    if tag and tag.group(1) != SCENARIO:
        raise ValueError(f"{MODEL_PATH.name} is a {tag.group(1)} model but SCENARIO = {SCENARIO}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()} | model={MODEL_PATH.name} | PGD-{PGD_STEPS} eps(Linf)={PGD_EPS} "
          f"| sigma={SMOOTH_SIGMA} | standardize={STANDARDIZE}")
    clamp01 = not STANDARDIZE

    with np.load(dataset_path, allow_pickle=True) as data:
        X_train = data['x_train'].astype(np.float32)
        X_test = torch.tensor(data['x_test'], dtype=torch.float32)
        y_test_6_classes = [mapping_classes(y.item()) for y in data['y_test']]
        y_test = torch.tensor(y_test_6_classes, dtype=torch.long)

    if STANDARDIZE:
        mu = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
        sd = torch.tensor(X_train.std(axis=0) + 1e-8, dtype=torch.float32)
        X_test = (X_test - mu) / sd
        print("Features z-scored (mean 0, std 1) using TRAIN statistics.")

    # Stratified subset of the test set to attack
    X_att, _, y_att, _ = train_test_split(
        X_test, y_test, train_size=SUBSET_SIZE, stratify=y_test, random_state=42
    )
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_att, y_att), batch_size=64, shuffle=False
    )

    if "FTTransformer" in MODEL_PATH.name:
        print("Loading FTTransformer model...")
        raw_model = FTTransformer.make_default(n_num_features=NUM_FEATURES, cat_cardinalities=None, d_out=NUM_CLASSES)
        raw_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = FTTransformerWrapper(raw_model).to(device)
    elif "MTL_LSTM" in MODEL_PATH.name:
        print(f"Loading MTL-LSTM model from {MODEL_PATH}...")
        model = MTLLSTM(input_size=NUM_FEATURES, hidden_size=150, num_classes=NUM_CLASSES, dropout=0.3).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    elif "QH_DNN" in MODEL_PATH.name:
        print(f"Loading QH-DNN model from {MODEL_PATH}...")
        model = QHDNNModel().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        raise ValueError(f"Unsupported model name: {MODEL_PATH}")

    model.eval()

    epsilons = [PGD_EPS]
    for eps in epsilons:
        print(f"\n" + "="*60)
        print(f"EVALUATING PGD-{PGD_STEPS} ATTACK | L-inf BOUND (direct) = {eps}")
        print("="*60)

        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_probs = []

        smoothed_classifier = SmoothBaseline(model, sigma=SMOOTH_SIGMA, clamp01=clamp01, num_classes=NUM_CLASSES)
        progress_bar = tqdm(dataloader, desc=f"Testing eps_Linf={eps}")
        for b_x, b_y in progress_bar:
            b_x, b_y = b_x.to(device), b_y.to(device)

            if eps > 0:
                adv_b_x = pgd_attack(model, b_x, b_y, eps, PGD_STEPS, clamp01)
            else:
                adv_b_x = b_x

            preds, probs = smoothed_classifier.predict_batch(adv_b_x, n=N_SMOOTH)

            correct_predictions += (preds == b_y.cpu().numpy()).sum()
            total_samples += b_y.size(0)

            all_preds.extend(preds)
            all_targets.extend(b_y.cpu().numpy())
            all_probs.extend(probs)

        test_accuracy = (correct_predictions / total_samples) * 100
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        mcc = matthews_corrcoef(all_targets, all_preds)
        g_mean = geometric_mean_score(all_targets, all_preds, average='macro')

        try:
            roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = 0.0

        print(f"\nResults for PGD-{PGD_STEPS} (L-inf direct eps={eps}):")
        print(f"Accuracy:  {test_accuracy:.2f}%")
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

if __name__ == "__main__":
    main()
