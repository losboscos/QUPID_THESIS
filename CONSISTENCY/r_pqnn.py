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

# Control experiment: is the training recipe the missing piece? Consistency regularization
# (Jeong & Shin 2020) at the sigma where the classical baseline is robust: two independent noisy
# draws per batch, CE on both plus a KL term pushing them to agree.
# Regime: z-scored phasors, no magnitude clamp, sigma set directly (not via the DP formula).
# The checkpoint is not committed: re-run the training to regenerate it.
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)  # depolarizing measured inert -> fast device

TRAIN_SIGMA = 0.4472       # sigma in z-scored space (where the classical baseline IS robust)
LAMBDA = 5.0               # consistency weight (0 recovers plain noise-aware training)
EPOCHS, LR = 200, 0.01
MODEL_TAG = f"r_qupid_consistency_s{TRAIN_SIGMA}"

def init_angles(tensor):
    return torch.nn.init.uniform_(tensor, a=-math.pi, b=math.pi)

def zscore_stats(relays):
    """Per-relay, per-component (real & imag) mean/std over the training set."""
    stats = []
    for r in relays:
        stats.append((r.real.mean(0), r.real.std(0) + 1e-6, r.imag.mean(0), r.imag.std(0) + 1e-6))
    return stats

def zscore(relay, st):
    mr, sr, mi, si = st
    return torch.complex((relay.real - mr) / sr, (relay.imag - mi) / si)

def add_zscore_noise(relay, sigma):
    """Gaussian noise in z-scored space, no magnitude clamp (unlike the DP/MinMax regime)."""
    r = relay.real + torch.randn_like(relay.real) * sigma
    i = relay.imag + torch.randn_like(relay.imag) * sigma
    return torch.complex(r, i)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), pad_with=0j, normalize=True)
    for layer in range(3):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
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
        target_device = next(self.classical_classifier.parameters()).device
        return self.classical_classifier(out.to(target_device))

def noisy_partitions(zr, sigma):
    """One noisy draw: z-scored relays -> +Gaussian(sigma) -> the 4 complex partitions."""
    nr = [add_zscore_noise(r, sigma) for r in zr]
    return (torch.cat((nr[0], nr[1]), 1), torch.cat((nr[1], nr[2]), 1),
            torch.cat((nr[2], nr[3]), 1), torch.cat((nr[3], nr[0]), 1))

def kl(p, q):
    return (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(1).mean()

def main():
    complex_dataset = Path("dataset3/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CONSISTENCY test | z-scored | sigma={TRAIN_SIGMA} | lambda={LAMBDA} | default.qubit | {device.type.upper()}")

    with np.load(complex_dataset, allow_pickle=True) as data:
        Xtr = [torch.tensor(data[f'x_train_r{i}'], dtype=torch.complex64)[:, :16] for i in range(1, 5)]
        y_train = torch.tensor([mapping_classes(y.item()) for y in data['y_train']], dtype=torch.long)

    stats = [tuple(t.to(device) for t in st) for st in zscore_stats(Xtr)]

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr[0], Xtr[1], Xtr[2], Xtr[3], y_train),
        batch_size=128, shuffle=True)

    model = QUPIDModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_loss = float('inf')
    model.train()

    for epoch in range(EPOCHS):
        total_loss, ce_sum, cons_sum = 0.0, 0.0, 0.0
        all_preds, all_targets = [], []

        for b_r1, b_r2, b_r3, b_r4, b_y in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            b_r1, b_r2, b_r3, b_r4, b_y = [t.to(device) for t in (b_r1, b_r2, b_r3, b_r4, b_y)]
            zr = [zscore(r, st) for r, st in zip((b_r1, b_r2, b_r3, b_r4), stats)]

            logits1 = model(*noisy_partitions(zr, TRAIN_SIGMA))
            logits2 = model(*noisy_partitions(zr, TRAIN_SIGMA))
            ce = 0.5 * (F.cross_entropy(logits1, b_y) + F.cross_entropy(logits2, b_y))
            p1, p2 = F.softmax(logits1, 1), F.softmax(logits2, 1)
            pbar = 0.5 * (p1 + p2)
            consistency = 0.5 * (kl(p1, pbar) + kl(p2, pbar))
            loss = ce + LAMBDA * consistency

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item(); ce_sum += ce.item(); cons_sum += consistency.item()
            all_preds.extend(logits1.argmax(1).cpu().tolist())
            all_targets.extend(b_y.cpu().tolist())

        avg_loss = total_loss / len(loader)
        acc = (np.array(all_preds) == np.array(all_targets)).mean() * 100
        mcc = matthews_corrcoef(all_targets, all_preds)
        print(f"--- Epoch {epoch+1} | loss {avg_loss:.4f} (ce {ce_sum/len(loader):.4f} + {LAMBDA}*cons "
              f"{cons_sum/len(loader):.4f}) | noisy-train acc {acc:.2f}% MCC {mcc:.4f} ---")

        if epoch == 0 or avg_loss < best_loss:
            best_loss = avg_loss
            Path("MODELS_R-QUPID").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"MODELS_R-QUPID/{MODEL_TAG}.pth")
            print(f"  saved MODELS_R-QUPID/{MODEL_TAG}.pth")

if __name__ == "__main__":
    main()
