import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root on sys.path
from dataset import mapping_classes
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from collections import Counter


# mapping_classes is imported from dataset (single source of truth)

SCENARIO = "S12"            # S1, S3, S11, S12
EPOCHS = 600
DATASET = Path(f"dataset3/scenarios/data{SCENARIO[1:]} Sampled Scenarios_scaled.npz")
MODEL_OUT = Path(f"BASELINES/MTL_LSTM/MTL_LSTM_{SCENARIO}.pth")


class MTLLSTM(nn.Module):
    """
    MTL-LSTM Baseline Architecture adaptated for 6-class classification.
    """
    def __init__(self, input_size=120, hidden_size=150, num_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, 120) -> Reshape for LSTM: (batch_size, seq_len=1, 120)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)  # (batch_size, 1, hidden_size)
        last_step_out = lstm_out[:, -1, :]  # Extract the output of the last time step
        logits = self.classifier(last_step_out)
        return logits


def main():
    if MODEL_OUT.exists():
        raise FileExistsError(f"{MODEL_OUT} already exists: delete or rename it before retraining.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {SCENARIO} | {EPOCHS} epochs -> {MODEL_OUT}")

    dataset_path = DATASET

    with np.load(dataset_path, allow_pickle=True) as data:
        X_train_scaled = torch.tensor(data['x_train'], dtype=torch.float32)
        y_train_6_classes = [mapping_classes(y.item()) for y in data['y_train']]
        y_train_tensor = torch.tensor(y_train_6_classes, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_scaled, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = MTLLSTM(input_size=120, hidden_size=150, num_classes=6, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_probs = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)

            all_preds.extend(predictions.cpu().tolist())
            all_targets.extend(y_batch.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        epoch_accuracy = (correct_predictions / total_samples) * 100
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        mcc = matthews_corrcoef(all_targets, all_preds)
        g_mean = geometric_mean_score(all_targets, all_preds, average='macro')

        try:
            roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = 0.0

        print(f"\n--- End of epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.4f} ---")
        print(f"Accuracy:  {epoch_accuracy:.2f}%")
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

        if epoch == 0 or avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"New best model saved with avg loss {best_loss:.4f} and accuracy {epoch_accuracy:.2f}%\n")


if __name__ == "__main__":
    main()

# https://github.com/2023ATAI/MTL-LSTM
