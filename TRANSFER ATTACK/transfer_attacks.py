import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
from pathlib import Path
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from rtdl import FTTransformer
import torchattacks
from BASELINES.MTL_LSTM.MTL_LSTM import MTLLSTM
from BASELINES.QH_DNN.QH_DNN import QHDNNModel
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
import torch.nn as nn

NUM_REAL_DIMENSIONS = 120  # Number of real-valued features in the dataset
SUBSET_SIZE = 500  # We take only a subset of the test set for generate adversarial attacks
NUM_CLASSES = 6
EPSILONS = [0.0, 0.05, 0.1] # attack bounds for the FGSM attack
PGD_STEPS = 20  # Number of steps for the PGD attack
SCENARIO = "S12"  # S1, S3, S11, S12
SURROGATES = ["MTL_LSTM", "FTTransformer", "QH_DNN"]
OUT_PATH = Path(f"TRANSFER ATTACK/{SCENARIO}")
TAG = "" if SCENARIO == "S1" else f"_{SCENARIO}"  # S1 files have no suffix

def evaluate_on_surrogate(model, adv_X, y, device, tag):
    """Accuracy of the surrogate on its own adversarial examples (white-box)."""
    with torch.no_grad():
        logits = model(adv_X.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = (preds == y.numpy()).mean() * 100
    mcc = matthews_corrcoef(y.numpy(), preds)
    print(f"  [surrogate white-box] {tag}: acc={acc:.2f}%  MCC={mcc:.4f}")
    return acc

class FTTransformerWrapper(nn.Module):
    """
    Adapter to make FTTransformer compatible with the standard model(x) interface
    required by torchattacks and PyTorch.
    """
    def __init__(self, ft_model):
        super().__init__()
        self.ft_model = ft_model

    def forward(self, x):
        # x_cat = None
        return self.ft_model(x, None)

def main():
    for surrogate in SURROGATES:
        generate(Path(f"BASELINES/{surrogate}/{surrogate}_{SCENARIO}.pth"))

def generate(model_name):
    dataset_path = Path(f"dataset3/scenarios/data{SCENARIO[1:]} Sampled Scenarios_scaled.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cuDNN does not allow the LSTM backward in eval mode, needed for the MTL-LSTM input gradients
    torch.backends.cudnn.enabled = False
    print(f"Using device: {device.type.upper()} | scenario {SCENARIO} | surrogate {model_name}")

    with np.load(dataset_path, allow_pickle=True) as data:
        X_test = torch.tensor(data['x_test'], dtype=torch.float32)
        y_test_6_classes = [mapping_classes(y.item()) for y in data['y_test']]
        y_test = torch.tensor(y_test_6_classes, dtype=torch.long)
        features = data['features'].tolist()

    # We use train_test_split to preserve class distribution
    X_att, _, y_att, _ = train_test_split(
        X_test,
        y_test,
        train_size=SUBSET_SIZE,
        stratify=y_test,
        random_state=42
    )

    dataset = torch.utils.data.TensorDataset(X_att, y_att)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    if "FTTransformer" in model_name.name:
        print("Loading FTTransformer model...")
        name = "FTTransformer"
        raw_model = FTTransformer.make_default(
            n_num_features=NUM_REAL_DIMENSIONS,
            cat_cardinalities=None,
            d_out=NUM_CLASSES,
        )
        raw_model.load_state_dict(torch.load(model_name, map_location=device))
        model = FTTransformerWrapper(raw_model).to(device)
    elif "MTL_LSTM" in model_name.name:
        print(f"Loading MTL-LSTM model from {model_name}...")
        name = "MTL_LSTM"
        model = MTLLSTM(input_size=NUM_REAL_DIMENSIONS, hidden_size=150, num_classes=NUM_CLASSES, dropout=0.3).to(device)
        model.load_state_dict(torch.load(model_name, map_location=device))
    elif "QH_DNN" in model_name.name:
        print(f"Loading QH-DNN model from {model_name}...")
        name = "QH_DNN"
        model = QHDNNModel().to(device)
        model.load_state_dict(torch.load(model_name, map_location=device))
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.eval()

    # Dictionary to save adversarial examples and related data
    saved = {"features": np.array(features, dtype=object),
             "y": y_att.numpy(),
             "x": X_att.numpy()}

    # clean test subset evaluation on the surrogate model
    evaluate_on_surrogate(model, X_att, y_att, device, tag="clean test subset")

    for eps in EPSILONS:
        print(f"\n" + "="*60)
        print(f"EVALUATING ATTACKS | TARGET BOUND = {eps}")
        print("="*60)

        # prepare attacks and fgsm/pgd structures
        fgsm = torchattacks.FGSM(model, eps=eps)
        pgd = torchattacks.PGD(model, eps=eps, alpha=2.5 * eps / PGD_STEPS, steps=PGD_STEPS, random_start=True)
        adv_fgsm = torch.empty_like(X_att)
        adv_pgd = torch.empty_like(X_att)
        ptr = 0

        progress_bar = tqdm(dataloader, desc=f"eps={eps}")
        for b_x, b_y in progress_bar:
            b_x, b_y = b_x.to(device), b_y.to(device)
            n = b_y.size(0)
            adv_fgsm[ptr:ptr + n] = fgsm(b_x, b_y).detach().cpu()
            adv_pgd[ptr:ptr + n] = pgd(b_x, b_y).detach().cpu()
            ptr += n

        # saving adversarial examples for the respective epsilon
        tag = f"{int(eps*100):03d}"
        saved[f"fgsm_{tag}"] = adv_fgsm.numpy()
        saved[f"pgd_{tag}"] = adv_pgd.numpy()

        # Evaluate the effectiveness of the adversarial examples on the surrogate model
        evaluate_on_surrogate(model, adv_fgsm, y_att, device, f"FGSM eps={eps}")
        evaluate_on_surrogate(model, adv_pgd, y_att, device, f"PGD  eps={eps}")
        print(f"  max |adv-clean| FGSM: {(adv_fgsm - X_att).abs().max():.4f} | PGD: {(adv_pgd - X_att).abs().max():.4f}")

        # Save the adversarial examples and related data to a compressed .npz file
        OUT_PATH.mkdir(parents=True, exist_ok=True)
        final_path = OUT_PATH / f"transfer_adv_{name}{TAG}.npz"
        np.savez_compressed(final_path, **saved)
        print(f"\nSaved adversarial set -> {final_path}")
        print("Keys:", ", ".join(k for k in saved if k not in ("features",)))

if __name__ == "__main__":
    main()
