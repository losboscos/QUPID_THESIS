import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from rtdl import FTTransformer
from BASELINES.MTL_LSTM.MTL_LSTM import MTLLSTM
from BASELINES.QH_DNN.QH_DNN import QHDNNModel

# mapping_classes is imported from dataset (single source of truth)

def evaluate_model(dataset, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    if "FTTransformer" in model_name:
        print("Loading FTTransformer model...")
        model = FTTransformer.make_default(
            n_num_features=120,
            cat_cardinalities=None,
            d_out=6
        ).to(device)
    elif "MTL_LSTM" in model_name:
        print(f"Loading MTL-LSTM model from {model_name}...")
        model = MTLLSTM(input_size=120, hidden_size=150, num_classes=6, dropout=0.3).to(device)
    elif "QH_DNN" in model_name:
        print(f"Loading QH-DNN model from {model_name}...")
        model = QHDNNModel().to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    

    model.load_state_dict(torch.load(f'{model_name}', map_location=device))
    model.eval()

    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []

    print("Starting evaluation...")
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x, b_y = b_x.to(device), b_y.to(device)

            if "FTTransformer" in model_name:
                logits = model(b_x, None)
            else:
                logits = model(b_x)
            _, predicted_classes = torch.max(logits, 1) # Find the index of the max log-probability (predicted class)
            probs = F.softmax(logits, dim=1)
            correct_predictions += (predicted_classes == b_y).sum().item()
            total_samples += b_y.size(0)

            all_preds.extend(predicted_classes.cpu().tolist())
            all_targets.extend(b_y.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    print("\n" + "="*50 + "\n" + "RESULTS (TEST SET)" + "\n" + "="*50)

    test_accuracy = (correct_predictions / total_samples) * 100
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_targets, all_preds)
    g_mean = geometric_mean_score(all_targets, all_preds, average='macro')
    roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')

    print(f"Accuracy:  {test_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"G-Mean:    {g_mean:.4f}")

def main():
    # STANDARDIZE = True for z-scored models (r_FTTransformer_s*.pth, trained on z-scored features);
    # False for the [0,1]-MinMax clean baselines (FTTransformer_S1.pth / MTL_LSTM_S1.pth / ...).
    STANDARDIZE = False
    dataset = Path("dataset3/scenarios/data1 Sampled Scenarios_scaled.npz")
    model_name = "BASELINES/QH_DNN/QH_DNN_S1.pth"

    with np.load(dataset, allow_pickle=True) as data:
        X_train = data['x_train'].astype(np.float32)
        X_test = torch.tensor(data['x_test'], dtype=torch.float32)
        y_test_6_classes = [mapping_classes(y.item()) for y in data['y_test']]
        y_test_tensor = torch.tensor(y_test_6_classes, dtype=torch.long)

    if STANDARDIZE:
        # Same z-scoring convention as r_FTTransformer.py and fig3_reproduce.py (TRAIN statistics)
        mu = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
        sd = torch.tensor(X_train.std(axis=0) + 1e-8, dtype=torch.float32)
        X_test = (X_test - mu) / sd
        print("Features z-scored (mean 0, std 1) using TRAIN statistics.")

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test_tensor)

    evaluate_model(test_dataset, model_name)

if __name__ == "__main__":
    main()
