import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import pennylane as qml
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from dataset import mapping_classes

n_qubits = 5

SCENARIO = "S1"            # S1, S3, S11, S12
ATTACK_BOUND = 0.05        # [0.05, 0.1] Bound L2 of construction for DP (L_cons)
NOISE_PROBABILITY = 0.1

DATASET = Path(f"dataset3/scenarios_complex/data{SCENARIO[1:]} Sampled Scenarios_scaled_complex.npz")
MODEL_OUT = Path(f"MODELS_R-QUPID/r_qupid_{SCENARIO}_L{str(ATTACK_BOUND).replace('0.', '0')}.pth")
dev = qml.device("default.mixed" if NOISE_PROBABILITY > 0 else "default.qubit", wires=n_qubits)

def init_angles(tensor):
    return torch.nn.init.uniform_(tensor, a=-math.pi, b=math.pi)

def add_gaussian_noise(pmu_tensor, sigma_sq):
    """ Gaussian noise injection in cartesian coordinates (real and imaginary) """
    std = math.sqrt(sigma_sq)
    reals = pmu_tensor.real
    imags = pmu_tensor.imag

    reals_noisy = reals + torch.randn_like(reals) * std
    imags_noisy = imags + torch.randn_like(imags) * std

    complex_noisy = torch.complex(reals_noisy, imags_noisy)

    # Physical limitation of the magnitude (<= 1.0): the phasors live in the unit disk
    # (mag in [0,1]); without this clamp the noise pushes them outside the domain and the normalization
    # of the AmplitudeEmbedding makes them dominate the state, and it leads to unstable training.
    mags = complex_noisy.abs()
    complex_noisy = torch.where(mags > 1.0, complex_noisy / mags, complex_noisy)

    return complex_noisy.to(pmu_tensor.dtype)

def compute_minimum_dot_product(complex_features):
    """
    Computation of the minimum dot product phi (|⟨x|x'⟩|^2)
    between all pairs of complex samples in the dataset.
    """
    norms = np.linalg.norm(complex_features, axis=1, keepdims=True)
    normalized = complex_features / (norms + 1e-12)
    gram_matrix = np.abs(np.dot(normalized, normalized.conj().T)) ** 2
    np.fill_diagonal(gram_matrix, np.inf)
    phi = float(np.min(gram_matrix))
    return phi

def compute_local_dp_parameters(epsilon_target, delta_target, p, phi, delta_L=1.0, K=4):
    """
    Inverting Lemma 5 of R-QUPID to derive the standard deviation (sigma)
    of classical Gaussian noise for each partition.
    """
    local_epsilon = epsilon_target / K
    denominator = K * (1.0 - p) * math.sqrt(1.0 - phi)

    if denominator <= 0:
        raise ValueError("The denominator for the delta calculation is <= 0.")

    local_delta = delta_target / denominator
    term_log = 1.25 / local_delta

    if term_log <= 0:
        raise ValueError("The value for the logarithm in sigma is not valid.")

    sigma = math.sqrt(2.0 * math.log(term_log)) * (delta_L / local_epsilon)
    return local_epsilon, local_delta, sigma

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    QNN Encoder + Ansatz + Quantum Noise Channel (Q^(eta) in HYPER-Q).
    """
    # 1. The Encoder
    qml.AmplitudeEmbedding(
        features=inputs,
        wires=range(n_qubits),
        pad_with=0j,
        normalize=True
    )

    # 2. The Trainable Ansatz
    for layer in range(3):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

        for i in range(n_qubits):
            target = (i + 1) % n_qubits
            qml.CNOT(wires=[i, target])

    # 3. NOISE INJECTION (Depolarizing channel after the ansatz)
    if NOISE_PROBABILITY > 0:
        for i in range(n_qubits):
            qml.DepolarizingChannel(NOISE_PROBABILITY, wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QUPIDModel(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (3, n_qubits, 2)}
        custom_init = {"weights": init_angles}

        self.pqc_p1 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)
        self.pqc_p2 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)
        self.pqc_p3 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)
        self.pqc_p4 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)

        self.classical_classifier = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, p1, p2, p3, p4):
        out_1 = self.pqc_p1(p1)
        out_2 = self.pqc_p2(p2)
        out_3 = self.pqc_p3(p3)
        out_4 = self.pqc_p4(p4)

        joined_quantum_features = torch.cat((out_1, out_2, out_3, out_4), dim=1)
        joined_quantum_features = joined_quantum_features.real.float()

        target_device = next(self.classical_classifier.parameters()).device
        joined_quantum_features = joined_quantum_features.to(target_device)
        final_prediction = self.classical_classifier(joined_quantum_features)

        return final_prediction

def main():
    complex_dataset = DATASET
    if MODEL_OUT.exists():
        raise FileExistsError(f"{MODEL_OUT} already exists: delete or rename it before retraining.")
    print(f"Scenario {SCENARIO} | L_cons={ATTACK_BOUND} | p={NOISE_PROBABILITY} -> {MODEL_OUT}")

    with np.load(complex_dataset, allow_pickle=True) as data:
        X_train_r1 = torch.tensor(data['x_train_r1'], dtype=torch.complex64)
        X_train_r2 = torch.tensor(data['x_train_r2'], dtype=torch.complex64)
        X_train_r3 = torch.tensor(data['x_train_r3'], dtype=torch.complex64)
        X_train_r4 = torch.tensor(data['x_train_r4'], dtype=torch.complex64)

        y_train = torch.tensor(data['y_train'], dtype=torch.long)
        y_train_6_classes = [mapping_classes(y.item()) for y in y_train]
        y_train_tensor = torch.tensor(y_train_6_classes, dtype=torch.long)

        X_train_r1_cleaned = X_train_r1[:, :16]
        X_train_r2_cleaned = X_train_r2[:, :16]
        X_train_r3_cleaned = X_train_r3[:, :16]
        X_train_r4_cleaned = X_train_r4[:, :16]

        P1_train = torch.cat((X_train_r1_cleaned, X_train_r2_cleaned), dim=1).numpy()
        P2_train = torch.cat((X_train_r2_cleaned, X_train_r3_cleaned), dim=1).numpy()
        P3_train = torch.cat((X_train_r3_cleaned, X_train_r4_cleaned), dim=1).numpy()
        P4_train = torch.cat((X_train_r4_cleaned, X_train_r1_cleaned), dim=1).numpy()

        # Compute the minimum dot product (phi) for each partition
        phi_r1 = compute_minimum_dot_product(P1_train)
        phi_r2 = compute_minimum_dot_product(P2_train)
        phi_r3 = compute_minimum_dot_product(P3_train)
        phi_r4 = compute_minimum_dot_product(P4_train)

        print(f"Minimum dot product (phi) R1: {phi_r1}")
        print(f"Minimum dot product (phi) R2: {phi_r2}")
        print(f"Minimum dot product (phi) R3: {phi_r3}")
        print(f"Minimum dot product (phi) R4: {phi_r4}")

        eps_t, del_t = 20.0, 1e-5  # Target global DP parameters (epsilon, delta)

        # Compute local DP parameters for each partition
        eps_loc_1, del_loc_1, sigma_val_1 = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi_r1, delta_L=ATTACK_BOUND)
        eps_loc_2, del_loc_2, sigma_val_2 = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi_r2, delta_L=ATTACK_BOUND)
        eps_loc_3, del_loc_3, sigma_val_3 = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi_r3, delta_L=ATTACK_BOUND)
        eps_loc_4, del_loc_4, sigma_val_4 = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi_r4, delta_L=ATTACK_BOUND)

        print(f"R1 -> Local Epsilon: {eps_loc_1:.2f}, Local Delta: {del_loc_1:.2e}, Sigma: {sigma_val_1:.4f}")
        print(f"R2 -> Local Epsilon: {eps_loc_2:.2f}, Local Delta: {del_loc_2:.2e}, Sigma: {sigma_val_2:.4f}")
        print(f"R3 -> Local Epsilon: {eps_loc_3:.2f}, Local Delta: {del_loc_3:.2e}, Sigma: {sigma_val_3:.4f}")
        print(f"R4 -> Local Epsilon: {eps_loc_4:.2f}, Local Delta: {del_loc_4:.2e}, Sigma: {sigma_val_4:.4f}")

        # Compute sigma squared for each partition
        sigma_sq_1 = sigma_val_1 ** 2
        sigma_sq_2 = sigma_val_2 ** 2
        sigma_sq_3 = sigma_val_3 ** 2
        sigma_sq_4 = sigma_val_4 ** 2

        BATCH_SIZE = 128
        dataset = torch.utils.data.TensorDataset(X_train_r1_cleaned, X_train_r2_cleaned, X_train_r3_cleaned, X_train_r4_cleaned, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice used: {device.type.upper()}")

        print("Instantiating the Hybrid R-QUPID Model...")
        model = QUPIDModel().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        EPOCHS = 200
        best_loss = float('inf')
        model.train()

        for epoch in range(EPOCHS):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            all_preds = []
            all_targets = []
            all_probs = []

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for b_r1, b_r2, b_r3, b_r4, b_y in progress_bar:
                b_r1, b_r2, b_r3, b_r4, b_y = b_r1.to(device), b_r2.to(device), b_r3.to(device), b_r4.to(device), b_y.to(device)

                partition_1 = torch.cat((b_r1, b_r2), dim=1)
                partition_2 = torch.cat((b_r2, b_r3), dim=1)
                partition_3 = torch.cat((b_r3, b_r4), dim=1)
                partition_4 = torch.cat((b_r4, b_r1), dim=1)

                # Gaussian noise injection for Differential Privacy (DP) in each partition
                partition_1 = add_gaussian_noise(partition_1, sigma_sq_1)
                partition_2 = add_gaussian_noise(partition_2, sigma_sq_2)
                partition_3 = add_gaussian_noise(partition_3, sigma_sq_3)
                partition_4 = add_gaussian_noise(partition_4, sigma_sq_4)

                optimizer.zero_grad()

                logits = model(partition_1, partition_2, partition_3, partition_4)
                loss = criterion(logits, b_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

                probs = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == b_y).sum().item()
                total_samples += b_y.size(0)

                all_preds.extend(predictions.cpu().tolist())
                all_targets.extend(b_y.cpu().tolist())
                all_probs.extend(probs.detach().cpu().tolist())

                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
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
