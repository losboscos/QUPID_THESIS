import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data
from pathlib import Path
from tqdm import tqdm
import pennylane as qml
from sklearn.metrics import matthews_corrcoef

# Control experiment: is the depolarizing channel the missing piece of the claimed robustness?
# Standalone copy of the R-QUPID pipeline with the channel ACTIVE, optional finite-shot measurement
# noise, and noise-aware training. The root pipeline is untouched.
# The checkpoint is not committed: re-run the training to regenerate it.
n_qubits = 5
NOISE_PROBABILITY = 0.05   # Depolarizing channel ON (the paper's quantum noise)
SHOTS = 100                # None = analytic <Z>. Int K = simulated shot noise, see forward()
ATTACK_BOUND = 0.05        # L_cons, the construction bound in the DP sigma formula

MODEL_TAG = f"r_qupid_qnoise_p{NOISE_PROBABILITY}"

# The channel needs a mixed-state simulator; the shot noise is added in forward().
dev = qml.device("default.mixed", wires=n_qubits)

def init_angles(tensor):
    return torch.nn.init.uniform_(tensor, a=-math.pi, b=math.pi)

def add_gaussian_noise(pmu_tensor, sigma_sq):
    """Gaussian DP noise in cartesian coordinates (real and imaginary)."""
    std = math.sqrt(sigma_sq)
    reals = pmu_tensor.real + torch.randn_like(pmu_tensor.real) * std
    imags = pmu_tensor.imag + torch.randn_like(pmu_tensor.imag) * std
    complex_noisy = torch.complex(reals, imags)
    # Physical limitation of the magnitude (<= 1.0): the phasors live in the unit disk.
    mags = complex_noisy.abs()
    complex_noisy = torch.where(mags > 1.0, complex_noisy / mags, complex_noisy)
    return complex_noisy.to(pmu_tensor.dtype)

def compute_minimum_dot_product(complex_features):
    """Minimum dot product phi (|<x|x'>|^2) between all pairs of complex samples."""
    norms = np.linalg.norm(complex_features, axis=1, keepdims=True)
    normalized = complex_features / (norms + 1e-12)
    gram_matrix = np.abs(np.dot(normalized, normalized.conj().T)) ** 2
    np.fill_diagonal(gram_matrix, np.inf)
    return float(np.min(gram_matrix))

def compute_local_dp_parameters(epsilon_target, delta_target, p, phi, delta_L=1.0, K=4):
    """Inverting Lemma 5 of R-QUPID for the Gaussian noise sigma of each partition."""
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
    """Encoder + ansatz + quantum noise channel (depolarizing), on default.mixed."""
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), pad_with=0j, normalize=True)
    for layer in range(3):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
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
            nn.Linear(20, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, p1, p2, p3, p4):
        out = torch.cat((self.pqc_p1(p1), self.pqc_p2(p2), self.pqc_p3(p3), self.pqc_p4(p4)), dim=1)
        out = out.real.float()

        # Simulated finite-shot measurement noise (CLT), fresh every forward:
        # <Z>_est = <Z> + N(0, sqrt((1-<Z>^2)/K)). PennyLane's own sampler on default.mixed
        # raises "probabilities do not sum to 1", so it is simulated here instead.
        if SHOTS is not None:
            var = torch.clamp(1.0 - out ** 2, min=0.0) / float(SHOTS)
            out = out + torch.randn_like(out) * torch.sqrt(var)
            out = torch.clamp(out, -1.0, 1.0)

        target_device = next(self.classical_classifier.parameters()).device
        return self.classical_classifier(out.to(target_device))

def main():
    complex_dataset = Path("dataset3/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Quantum noise ACTIVE | p={NOISE_PROBABILITY} | shots={SHOTS} | device=default.mixed | {device.type.upper()}")
    if SHOTS is not None:
        print(f"Simulated shot noise ACTIVE (K={SHOTS}) on the 20 quantum features.")

    with np.load(complex_dataset, allow_pickle=True) as data:
        Xtr = [torch.tensor(data[f'x_train_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
        y_train = torch.tensor([mapping_classes(y.item()) for y in data['y_train']], dtype=torch.long)

    P = [torch.cat((Xtr[a], Xtr[b]), dim=1).numpy() for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]]
    eps_t, del_t = 20.0, 1e-5
    sigmas_sq = []
    for k, p in enumerate(P):
        phi = compute_minimum_dot_product(p)
        el, dl, sv = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi, delta_L=ATTACK_BOUND)
        print(f"R{k+1} -> phi={phi:.4f}  local eps={el:.2f}, local delta={dl:.2e}, sigma={sv:.4f}")
        sigmas_sq.append(sv ** 2)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr[0], Xtr[1], Xtr[2], Xtr[3], y_train),
        batch_size=128, shuffle=True)

    model = QUPIDModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    model.train()

    for epoch in range(200):
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        for b_r1, b_r2, b_r3, b_r4, b_y in tqdm(loader, desc=f"Epoch {epoch+1}/200"):
            b_r1, b_r2, b_r3, b_r4, b_y = [t.to(device) for t in (b_r1, b_r2, b_r3, b_r4, b_y)]
            p1 = add_gaussian_noise(torch.cat((b_r1, b_r2), dim=1), sigmas_sq[0])
            p2 = add_gaussian_noise(torch.cat((b_r2, b_r3), dim=1), sigmas_sq[1])
            p3 = add_gaussian_noise(torch.cat((b_r3, b_r4), dim=1), sigmas_sq[2])
            p4 = add_gaussian_noise(torch.cat((b_r4, b_r1), dim=1), sigmas_sq[3])

            optimizer.zero_grad()
            logits = model(p1, p2, p3, p4)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == b_y).sum().item()
            total += b_y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(b_y.cpu().tolist())

        avg_loss = total_loss / len(loader)
        acc = correct / total * 100
        mcc = matthews_corrcoef(all_targets, all_preds)
        print(f"--- Epoch {epoch+1} | loss {avg_loss:.4f} | acc {acc:.2f}% | MCC {mcc:.4f} ---")

        if epoch == 0 or avg_loss < best_loss:
            best_loss = avg_loss
            Path("MODELS_R-QUPID").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"MODELS_R-QUPID/{MODEL_TAG}.pth")
            print(f"  saved MODELS_R-QUPID/{MODEL_TAG}.pth")

if __name__ == "__main__":
    main()
