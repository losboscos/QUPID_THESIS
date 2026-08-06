import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from r_pqnn import QUPIDModel, mapping_classes
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score

def evaluate_model(dataset, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    model = QUPIDModel().to(device)
    model.load_state_dict(torch.load(f'{model_name}', map_location=device))
    model.eval()

    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []

    print("Starting evaluation...")
    with torch.no_grad():
        for b_r1, b_r2, b_r3, b_r4, b_y in test_dataloader:
            b_r1, b_r2, b_r3, b_r4, b_y = b_r1.to(device), b_r2.to(device), b_r3.to(device), b_r4.to(device), b_y.to(device)

            partition_1 = torch.cat((b_r1, b_r2), dim=1)
            partition_2 = torch.cat((b_r2, b_r3), dim=1)
            partition_3 = torch.cat((b_r3, b_r4), dim=1)
            partition_4 = torch.cat((b_r4, b_r1), dim=1)

            logits = model(partition_1, partition_2, partition_3, partition_4)
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

    """
    # 1. Classification Report (Precision, Recall, F1-Score for each class)
    target_names = ['Normal Events', 'Short Circuits', 'Line Maintenance', 'Data Injection Attacks', 'Remote Tripping Command Injection', 'Relay Setting Change']
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    print("\nDetailed Report:")
    print(report)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)

    # 3. Generation and Saving of the Matrix Image
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix QUPID (15 Qubit)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')

    # Save the plot in high resolution for the thesis
    plt.savefig('confusion_matrix_qupid.png', dpi=300, bbox_inches='tight')
    print("\nConfusion Matrix saved as 'confusion_matrix_qupid.png'")

    # Show the image on screen
    plt.show()
    """

def main():
    complex_dataset = Path("dataset3/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
    model_name = "qupid_model.pth"

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

    evaluate_model(dataset, model_name)

if __name__ == "__main__":
    main()
