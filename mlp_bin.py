from collections import Counter
import math
import dataset
import pennylane as qml
import torch
from torch import optim
import torch.utils.data
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

def mapping_classes(old_marker):
    if old_marker in [1, 2, 3, 4, 5, 6, 13, 14, 41]:
        return 0  # Class 0: Normal
    else:
        return 1  # Class 1: Anomalous (all other markers)

class ClassicMLP(nn.Module):
    def __init__(self):
        super(ClassicMLP, self).__init__()
        
        self.extractor_p1 = nn.Sequential(nn.Linear(64, 5), nn.ReLU())
        self.extractor_p2 = nn.Sequential(nn.Linear(64, 5), nn.ReLU())
        self.extractor_p3 = nn.Sequential(nn.Linear(64, 5), nn.ReLU())
        self.extractor_p4 = nn.Sequential(nn.Linear(64, 5), nn.ReLU())

        self.classical_classifier = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, p1, p2, p3, p4):
        p1_real = torch.cat((p1.real, p1.imag), dim=1).float()
        p2_real = torch.cat((p2.real, p2.imag), dim=1).float()
        p3_real = torch.cat((p3.real, p3.imag), dim=1).float()
        p4_real = torch.cat((p4.real, p4.imag), dim=1).float()

        out_1 = self.extractor_p1(p1_real)
        out_2 = self.extractor_p2(p2_real)
        out_3 = self.extractor_p3(p3_real)
        out_4 = self.extractor_p4(p4_real)

        joined_features = torch.cat((out_1, out_2, out_3, out_4), dim=1)

        return self.classical_classifier(joined_features)

def main():
    complex_dataset = Path("dataset/scenarios_complex/data1 Sampled Scenarios_scaled_complex.npz")
    
    with np.load(complex_dataset, allow_pickle=True) as data:
        X_train_r1 = torch.tensor(data['x_train_r1'], dtype=torch.complex64)
        X_train_r2 = torch.tensor(data['x_train_r2'], dtype=torch.complex64)
        X_train_r3 = torch.tensor(data['x_train_r3'], dtype=torch.complex64)
        X_train_r4 = torch.tensor(data['x_train_r4'], dtype=torch.complex64)
        # Labels are classical integers, so we use torch.long
        y_train = torch.tensor(data['y_train'], dtype=torch.long)
        y_train_binary_classes = [mapping_classes(y.item()) for y in y_train]

        # Now we have 2 classes.
        y_train_tensor = torch.tensor(y_train_binary_classes, dtype=torch.long)

        X_train_r1_cleaned = torch.cat((X_train_r1[:, :15], X_train_r1[:, 16:]), dim=1)
        X_train_r2_cleaned = X_train_r2
        X_train_r3_cleaned = torch.cat((X_train_r3[:, :15], X_train_r3[:, 16:]), dim=1)
        X_train_r4_cleaned = torch.cat((X_train_r4[:, :15], X_train_r4[:, 16:]), dim=1)

        BATCH_SIZE = 128
        dataset = torch.utils.data.TensorDataset(X_train_r1_cleaned, X_train_r2_cleaned, X_train_r3_cleaned, X_train_r4_cleaned, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice used: {device.type.upper()}")

        print("Instantiating the MLP Model...")
        model = ClassicMLP()
        model.to(device)

        total_samples = 3232
        samples_per_class = [706, 2526]
        num_classes = 2
        balanced_weights = [total_samples / (num_classes * count) for count in samples_per_class]
        weights_tensor = torch.tensor(balanced_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        EPOCHS = 200
        best_loss = float('inf')
        model.train()

        for epoch in range(EPOCHS):
            total_loss = 0.0 # We keep track of the total loss for this epoch to calculate the average at the end
            correct_predictions = 0 
            total_samples = 0     
            all_preds = []
            all_targets = [] 
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for b_r1, b_r2, b_r3, b_r4, b_y in progress_bar:
                b_r1, b_r2, b_r3, b_r4, b_y = b_r1.to(device), b_r2.to(device), b_r3.to(device), b_r4.to(device), b_y.to(device)

                partition_1 = torch.cat((b_r1, b_r2), dim=1) 
                partition_2 = torch.cat((b_r2, b_r3), dim=1) 
                partition_3 = torch.cat((b_r3, b_r4), dim=1) 
                partition_4 = torch.cat((b_r4, b_r1), dim=1)
                
                # We reset the gradients to zero before each batch, as PyTorch accumulates them by default
                optimizer.zero_grad()
                
                # Prediction: The model takes the 4 partitions and produces logits (raw scores for each class)
                logits = model(partition_1, partition_2, partition_3, partition_4)
                
                # Loss Calculation: We compare the predicted logits with the true labels using CrossEntropyLoss, which internally applies softmax to the logits
                loss = criterion(logits, b_y)
                
                # Backward Pass: This computes the gradients of the loss with respect to all trainable parameters in the model (both quantum and classical)
                loss.backward()
                
                # Weight Update: The optimizer takes a step in the direction of the negative gradients to update the model's parameters
                optimizer.step()
                
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == b_y).sum().item()
                total_samples += b_y.size(0)

                all_preds.extend(predictions.cpu().tolist())
                all_targets.extend(b_y.cpu().tolist())
                
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            epoch_accuracy = (correct_predictions / total_samples) * 100
            print(f"\n--- End of epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Accuracy: {epoch_accuracy:.2f}% | LR used: {optimizer.param_groups[0]['lr']:.4f} ---\n")
            
            pred_counts = Counter(all_preds)
            true_counts = Counter(all_targets)
            print("\nDistribution of Predictions vs Reality:")
            for class_idx in range(2):
                p_count = pred_counts.get(class_idx, 0)
                t_count = true_counts.get(class_idx, 0)
                print(f"Class {class_idx}: Predicted {p_count} times | (In reality there were {t_count})")
            print("-" * 60 + "\n")

            if epoch == 0 or avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"mlp_model_S1_bin.pth")
                print(f"New best model saved with avg loss {best_loss:.4f} and accuracy {epoch_accuracy:.2f}%\n")

            #scheduler.step(avg_loss)
        
if __name__ == "__main__":
    main()