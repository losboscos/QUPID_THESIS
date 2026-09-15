import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root on sys.path
from dataset import mapping_classes
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pennylane as qml
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from collections import Counter

# 8 Qubit. Probably 16 qubits would be better, but the training time would be too long.
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

def init_angles(tensor):
    return torch.nn.init.uniform_(tensor, a=-math.pi, b=math.pi)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Architecture of the Quantum Circuit for the HQ-DNN model (Wang et al. / QUPID):
    1. Preparation Layer: Gate Hadamard H on all qubits + AngleEmbedding RY.
    2. Variational Ansatz: 8 layers of RY rotations + circular CNOT entanglement.
    3. Measurement Layer: Expectation value of PauliZ on each qubit in [-1, 1].
    """
    # 1. Preparation Layer
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    qml.AngleEmbedding(features=inputs, wires=range(n_qubits), rotation="Y")

    # 2. Variational Ansatz (8 Layer)
    for layer in range(8):
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)

        for i in range(n_qubits):
            target = (i + 1) % n_qubits
            qml.CNOT(wires=[i, target])

    # 3. Measurement Layer
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QHDNNModel(nn.Module):
    """
    Quantum-Classical Hybrid Deep Neural Network (HQ-DNN).
    Formulation: f(x) = B o VQC o A(x)
    - A(x): Linear(120, 64) -> ReLU -> Linear(64, 8)
    - VQC:  AngleEmbedding + 8 Layer VQC (8 Qubit)
    - B(z): Linear(8, 6) -> Logits for the 6 classes
    """
    def __init__(self):
        super().__init__()
        # Classical pre-processing A(x)
        self.pre_processing = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)  # it returns 8 latent quantum features in [-1, 1] for the VQC
        )

        weight_shapes = {"weights": (8, n_qubits)}
        custom_init = {"weights": init_angles}
        self.vqc_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)

        # Classical post-processing B(z)
        self.post_processing = nn.Linear(n_qubits, 6)

    def forward(self, x):
        # 1. Classical pre-processing
        latent_classical = self.pre_processing(x)
        # 2. VQC Core
        quantum_features = self.vqc_layer(latent_classical)
        # 3. Post-processing
        logits = self.post_processing(quantum_features)
        return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = Path("dataset3/scenarios/data1 Sampled Scenarios_scaled.npz")

    with np.load(dataset_path, allow_pickle=True) as data:
        X_train_scaled = torch.tensor(data['x_train'], dtype=torch.float32)
        y_train_6_classes = [mapping_classes(y.item()) for y in data['y_train']]
        y_train_tensor = torch.tensor(y_train_6_classes, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_scaled, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = QHDNNModel().to(device)

    total_samples = 3232
    samples_per_class = [116, 419, 171, 361, 307, 1858]
    smoothed_weights = [math.sqrt(total_samples / count) for count in samples_per_class]
    weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    EPOCHS = 200
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

        print(f"\n--- End of epoch {epoch+1} | Avg Loss: {avg_loss:.4f} ---")
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
            torch.save(model.state_dict(), "BASELINES/QH_DNN/QH_DNN_S1.pth")
            print(f"New best model saved with avg loss {best_loss:.4f} and accuracy {epoch_accuracy:.2f}%\n")


if __name__ == "__main__":
    main()