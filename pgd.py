import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
# Import of same modules in order to have consistency between the training and testing
from r_pqnn import QUPIDModel, compute_minimum_dot_product, compute_local_dp_parameters, add_gaussian_noise
from dataset import mapping_classes

SCENARIO = "S1"      # S1, S3, S11, S12
ATTACK_BOUND = 0.1  # Bound L2 of construction for DP (L_cons)
NOISE_PROBABILITY = 0.1   # Quantum noise (must match r_pqnn.py: it feeds the sigma computation)
NUM_REAL_DIMENSIONS = 128  # 4 PMUs * 16 feature * 2 (Real, Imag)
PGD_STEPS = 20

class QUPIDTorchattacksWrapper(nn.Module):
    """
    Wrapper that interacts in the real space [0, 1] by mapping the Cartesian coordinates (Real, Imag) of phasors.
    """
    def __init__(self, qupid_model):
        super().__init__()
        self.model = qupid_model

    def forward(self, x):
        # Convert the input tensor from [0, 1] to [-1, 1] for both real and imaginary parts
        reals = x[:, :, :, 0] * 2.0 - 1.0
        imags = x[:, :, :, 1] * 2.0 - 1.0

        complex_x = torch.complex(reals, imags)
        # Split the complex tensor into four separate tensors for each relay
        r1 = complex_x[:, 0, :]
        r2 = complex_x[:, 1, :]
        r3 = complex_x[:, 2, :]
        r4 = complex_x[:, 3, :]

        p1 = torch.cat((r1, r2), dim=1)
        p2 = torch.cat((r2, r3), dim=1)
        p3 = torch.cat((r3, r4), dim=1)
        p4 = torch.cat((r4, r1), dim=1)

        return self.model(p1, p2, p3, p4)

def prepare_data_for_torchattacks(b_r1, b_r2, b_r3, b_r4):
    complex_relays = torch.stack([b_r1, b_r2, b_r3, b_r4], dim=1)
    # Convert the complex tensor to real and imaginary parts in the range [0, 1]
    reals = complex_relays.real.float()
    imags = complex_relays.imag.float()
    # Normalize the real and imaginary parts to [0, 1]
    reals_norm = (reals + 1.0) / 2.0
    imags_norm = (imags + 1.0) / 2.0

    return torch.stack([reals_norm, imags_norm], dim=-1)

def pgd_attack(wrapper_model, x, y, eps, alpha, steps):
    # Honest wrapper + backprop -> the input gradient is real at every step, so the attack lands.
    # (Contrast with PARAMETER-SHIFT/pgd.py: there the quantum-native gradient is NaN -> no-op.)
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    last_grad = torch.zeros_like(x)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(wrapper_model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x_adv)
        last_grad = grad
        x_adv = (x_adv + alpha * grad.sign()).detach()
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = (x_orig + delta).detach()
    return x_adv, last_grad

class SmoothQUPID:
    def __init__(self, base_classifier: nn.Module, num_classes: int, sigmas_sq: list):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigmas_sq = sigmas_sq

    def predict(self, relays_single: list, n: int, batch_size: int):
        self.base_classifier.eval()
        counts = self._sample_noise(relays_single, n, batch_size)
        top2 = counts.argsort()[::-1][:2]

        return top2[0], counts / n

    def _sample_noise(self, relays_single: list, num: int, batch_size: int) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=float)
        r1, r2, r3, r4 = relays_single

        r1 = r1.unsqueeze(0) if r1.dim() == 1 else r1
        r2 = r2.unsqueeze(0) if r2.dim() == 1 else r2
        r3 = r3.unsqueeze(0) if r3.dim() == 1 else r3
        r4 = r4.unsqueeze(0) if r4.dim() == 1 else r4

        with torch.no_grad():
            for _ in range(math.ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                # Repeat the relays for the current batch size
                b_r1 = r1.repeat((this_batch_size, 1))
                b_r2 = r2.repeat((this_batch_size, 1))
                b_r3 = r3.repeat((this_batch_size, 1))
                b_r4 = r4.repeat((this_batch_size, 1))

                p1 = torch.cat((b_r1, b_r2), dim=1)
                p2 = torch.cat((b_r2, b_r3), dim=1)
                p3 = torch.cat((b_r3, b_r4), dim=1)
                p4 = torch.cat((b_r4, b_r1), dim=1)
                # Add Gaussian noise to each pair of relays
                noisy_p1 = add_gaussian_noise(p1, self.sigmas_sq[0])
                noisy_p2 = add_gaussian_noise(p2, self.sigmas_sq[1])
                noisy_p3 = add_gaussian_noise(p3, self.sigmas_sq[2])
                noisy_p4 = add_gaussian_noise(p4, self.sigmas_sq[3])

                logits = self.base_classifier(noisy_p1, noisy_p2, noisy_p3, noisy_p4)
                predictions = logits.argmax(dim=1)

                for idx in predictions.cpu().numpy():
                    counts[idx] += 1

        return counts

def main():
    complex_dataset = Path(f"dataset3/scenarios_complex/data{SCENARIO[1:]} Sampled Scenarios_scaled_complex.npz")
    model_name = Path(f"MODELS_R-QUPID/r_qupid_{SCENARIO}_L{str(ATTACK_BOUND).replace('0.', '0')}.pth")
    print(f"Scenario {SCENARIO} | L_cons={ATTACK_BOUND} | {model_name.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    with np.load(complex_dataset, allow_pickle=True) as data:
        X_test_r1 = torch.tensor(data['x_test_r1'], dtype=torch.complex64)
        X_test_r2 = torch.tensor(data['x_test_r2'], dtype=torch.complex64)
        X_test_r3 = torch.tensor(data['x_test_r3'], dtype=torch.complex64)
        X_test_r4 = torch.tensor(data['x_test_r4'], dtype=torch.complex64)

        y_test = torch.tensor(data['y_test'], dtype=torch.long)
        y_test_6_classes = [mapping_classes(y.item()) for y in y_test]
        y_test_tensor = torch.tensor(y_test_6_classes, dtype=torch.long)

        X_test_r1_cleaned = X_test_r1[:, :16]
        X_test_r2_cleaned = X_test_r2[:, :16]
        X_test_r3_cleaned = X_test_r3[:, :16]
        X_test_r4_cleaned = X_test_r4[:, :16]

        dataset = torch.utils.data.TensorDataset(X_test_r1_cleaned, X_test_r2_cleaned, X_test_r3_cleaned, X_test_r4_cleaned, y_test_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    print("Instantiating the Hybrid R-QUPID Model...")
    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{model_name}', map_location=device))
    model.eval()

    wrapper_model = QUPIDTorchattacksWrapper(model).to(device)
    wrapper_model.eval()

    P1_test = torch.cat((X_test_r1_cleaned, X_test_r2_cleaned), dim=1).numpy()
    P2_test = torch.cat((X_test_r2_cleaned, X_test_r3_cleaned), dim=1).numpy()
    P3_test = torch.cat((X_test_r3_cleaned, X_test_r4_cleaned), dim=1).numpy()
    P4_test = torch.cat((X_test_r4_cleaned, X_test_r1_cleaned), dim=1).numpy()
    # same as r_pqnn.py
    phi_r1 = compute_minimum_dot_product(P1_test)
    phi_r2 = compute_minimum_dot_product(P2_test)
    phi_r3 = compute_minimum_dot_product(P3_test)
    phi_r4 = compute_minimum_dot_product(P4_test)

    eps_t, del_t = 20.0, 1e-5

    _, _, sigma_val_1 = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi_r1, delta_L=ATTACK_BOUND)
    _, _, sigma_val_2 = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi_r2, delta_L=ATTACK_BOUND)
    _, _, sigma_val_3 = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi_r3, delta_L=ATTACK_BOUND)
    _, _, sigma_val_4 = compute_local_dp_parameters(eps_t, del_t, NOISE_PROBABILITY, phi_r4, delta_L=ATTACK_BOUND)

    sigmas_sq = [sigma_val_1**2, sigma_val_2**2, sigma_val_3**2, sigma_val_4**2]
    smoothed_classifier = SmoothQUPID(model, num_classes=6, sigmas_sq=sigmas_sq)

    epsilons = [0.0, 0.1]

    for eps in epsilons:
        print(f"\n" + "="*60)
        print(f"EVALUATING PGD ATTACK | FAIR L-inf BOUND = {eps}")
        print("="*60)

        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_probs = []
        grad_norm_sum, delta_sum, n_batches = 0.0, 0.0, 0

        progress_bar = tqdm(dataloader, desc=f"Testing eps_Linf={eps}")
        for b_r1, b_r2, b_r3, b_r4, b_y in progress_bar:
            b_r1, b_r2, b_r3, b_r4, b_y = b_r1.to(device), b_r2.to(device), b_r3.to(device), b_r4.to(device), b_y.to(device)

            if eps > 0:
                eps_pgd = eps / 2.0
                real_inputs = prepare_data_for_torchattacks(b_r1, b_r2, b_r3, b_r4)
                adv_real_tensor, grad = pgd_attack(wrapper_model, real_inputs, b_y, eps_pgd, 2.5 * eps_pgd / PGD_STEPS, PGD_STEPS)
                grad_norm_sum += float(grad.norm())
                delta_sum += float((adv_real_tensor - real_inputs).norm())
                n_batches += 1

                reals = adv_real_tensor[:, :, :, 0] * 2.0 - 1.0
                imags = adv_real_tensor[:, :, :, 1] * 2.0 - 1.0
                adv_complex = torch.complex(reals, imags)

                mags = adv_complex.abs()
                adv_complex = torch.where(mags > 1.0, adv_complex / mags, adv_complex)

                adv_r1 = adv_complex[:, 0, :]
                adv_r2 = adv_complex[:, 1, :]
                adv_r3 = adv_complex[:, 2, :]
                adv_r4 = adv_complex[:, 3, :]
            else:
                adv_r1, adv_r2, adv_r3, adv_r4 = b_r1, b_r2, b_r3, b_r4

            batch_size = b_y.size(0)
            for i in range(batch_size):
                relays_single = [adv_r1[i], adv_r2[i], adv_r3[i], adv_r4[i]]

                pred, prob_dist = smoothed_classifier.predict(relays_single, n=100, batch_size=128)

                correct_predictions += int(pred == b_y[i].item())
                total_samples += 1

                all_preds.append(pred)
                all_targets.append(b_y[i].item())
                all_probs.append(prob_dist)

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

        if n_batches:
            print(f"\n  mean input-gradient norm: {grad_norm_sum / n_batches:.6f}   (> 0 = real gradient)")
            print(f"  mean ||x_adv - x||_2:     {delta_sum / n_batches:.6f}   (> 0 = the attack lands)")

        print(f"\nResults for PGD (fair L-inf eps={eps}):")
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
