import torch
import numpy as np
from collections import Counter
from r_pqnn import QUPIDModel, mapping_classes
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score

def fgsm_attack(model, relays, labels, criterion, epsilon):
    # We need gradients on the physical quantities, so we rebuild the complex tensors from them
    magnitudes = [r.abs().clone().detach().requires_grad_(True) for r in relays]
    phases = [r.angle().clone().detach().requires_grad_(True) for r in relays]
    complex_relays = [magnitudes[i] * torch.exp(1j * phases[i]) for i in range(4)]

    partition_1 = torch.cat((complex_relays[0], complex_relays[1]), dim=1)
    partition_2 = torch.cat((complex_relays[1], complex_relays[2]), dim=1)
    partition_3 = torch.cat((complex_relays[2], complex_relays[3]), dim=1)
    partition_4 = torch.cat((complex_relays[3], complex_relays[0]), dim=1)

    logits = model(partition_1, partition_2, partition_3, partition_4)
    loss = criterion(logits, labels)
    model.zero_grad()
    loss.backward()

    # Build the adversarial PMUs: x_adv = x + epsilon * sign(grad), clamped to [0, 1]
    adversarial_relays = []
    for i in range(4):
        adv_magnitude = torch.clamp(magnitudes[i] + epsilon * magnitudes[i].grad.sign(), 0.0, 1.0)
        adv_phase = torch.clamp(phases[i] + epsilon * phases[i].grad.sign(), 0.0, 1.0)
        adversarial_relays.append((adv_magnitude * torch.exp(1j * adv_phase)).detach())
    return adversarial_relays

def main():
    complex_dataset = Path("dataset2/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
    model_name = "qupid_model_S1 (82.25% acc).pth"
    epsilon = 0.05 # The attack bound used in the paper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # We do NOT use class weights here because we are attacking, not training!
    criterion = nn.CrossEntropyLoss()

    with np.load(complex_dataset, allow_pickle=True) as data:
        X_test_r1 = torch.tensor(data['x_test_r1'], dtype=torch.complex64)
        X_test_r2 = torch.tensor(data['x_test_r2'], dtype=torch.complex64)
        X_test_r3 = torch.tensor(data['x_test_r3'], dtype=torch.complex64)
        X_test_r4 = torch.tensor(data['x_test_r4'], dtype=torch.complex64)
        # Labels are classical integers, so we use torch.long
        y_test = torch.tensor(data['y_test'], dtype=torch.long)
        y_test_6_classes = [mapping_classes(y.item()) for y in y_test]

        # Now we have 6 classes.
        y_test_tensor = torch.tensor(y_test_6_classes, dtype=torch.long)

        X_test_r1_cleaned = X_test_r1[:, :16]
        X_test_r2_cleaned = X_test_r2[:, :16]
        X_test_r3_cleaned = X_test_r3[:, :16]
        X_test_r4_cleaned = X_test_r4[:, :16]

        dataset = torch.utils.data.TensorDataset(X_test_r1_cleaned, X_test_r2_cleaned, X_test_r3_cleaned, X_test_r4_cleaned, y_test_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{model_name}', map_location=device))
    model.eval()

    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []

    print(f"Starting FGSM attack evaluation with epsilon = {epsilon}...")
    for b_r1, b_r2, b_r3, b_r4, b_y in dataloader:
        b_r1, b_r2, b_r3, b_r4, b_y = b_r1.to(device), b_r2.to(device), b_r3.to(device), b_r4.to(device), b_y.to(device)

        # Generate the adversarial version of the four PMUs
        relays = [b_r1, b_r2, b_r3, b_r4]
        adv_r1, adv_r2, adv_r3, adv_r4 = fgsm_attack(model, relays, b_y, criterion, epsilon)

        partition_1 = torch.cat((adv_r1, adv_r2), dim=1)
        partition_2 = torch.cat((adv_r2, adv_r3), dim=1)
        partition_3 = torch.cat((adv_r3, adv_r4), dim=1)
        partition_4 = torch.cat((adv_r4, adv_r1), dim=1)

        # Evaluate the model on the adversarial input
        with torch.no_grad():
            logits = model(partition_1, partition_2, partition_3, partition_4)
            _, predicted_classes = torch.max(logits, 1)
            probs = F.softmax(logits, dim=1)
            correct_predictions += (predicted_classes == b_y).sum().item()
            total_samples += b_y.size(0)

            all_preds.extend(predicted_classes.cpu().tolist())
            all_targets.extend(b_y.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    print("\n" + "="*50 + "\n" + f"ADVERSARIAL RESULTS (FGSM eps={epsilon})" + "\n" + "="*50)

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

    pred_counts = Counter(all_preds)
    true_counts = Counter(all_targets)
    print("\nDistribution of Adversarial Predictions vs Reality:")
    for class_idx in range(6):
        p_count = pred_counts.get(class_idx, 0)
        t_count = true_counts.get(class_idx, 0)
        print(f"Class {class_idx}: Predicted {p_count} times | (In reality there were {t_count})")

if __name__ == "__main__":
    main()
