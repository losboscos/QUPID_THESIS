import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path
from dataset import mapping_classes
import math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 5

ATTACK_BOUND = 0.05        # [0.05, 0.1] Bound L2 of construction for DP (L_cons)
NOISE_PROBABILITY = 0.1    # Quantum noise
dev = qml.device("default.mixed" if NOISE_PROBABILITY > 0 else "default.qubit", wires=n_qubits)

def init_angles(tensor):
    return torch.nn.init.uniform_(tensor, a=-math.pi, b=math.pi)

# mapping_classes is imported from dataset (single source of truth)

def add_gaussian_noise(pmu_tensor, sigma_sq):
    """ Gaussian noise injection in cartesian coordinates (real and imaginary) """
    std = math.sqrt(sigma_sq)
    reals = pmu_tensor.real
    imags = pmu_tensor.imag

    reals_noisy = reals + torch.randn_like(reals) * std
    imags_noisy = imags + torch.randn_like(imags) * std

    complex_noisy = torch.complex(reals_noisy, imags_noisy)

    # Physical limitation of the magnitude (<= 1.0): the phasors live in the unit disk.
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

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
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

# parameter-shift cannot broadcast MottonenStatePreparation (from AmplitudeEmbedding);
# broadcast_expand splits a batched call into per-sample tapes so the quantum-native diff can run.
quantum_circuit = qml.transforms.broadcast_expand(quantum_circuit)

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
