from collections import Counter
import math
import pennylane as qml
import torch
from torch import optim
import torch.utils.data
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

def init_angles(tensor):
    return torch.nn.init.uniform_(tensor, a=-math.pi, b=math.pi)

def mapping_classes(old_marker):
    if old_marker in [1, 2, 3, 4, 5, 6, 13, 14, 41]:
        return 0  # Class 0: Normal
    else:
        return 1  # Class 1: Anomalous (all other markers)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Takes a 1D tensor of complex features and encodes them into the 
    quantum state |psi> inside the 32-dimensional Hilbert space.
    """
    
    # 1. The Encoder (Data Loading)
    qml.AmplitudeEmbedding(
        features=inputs, 
        wires=range(n_qubits), 
        pad_with=0j,       # Fills the gap from ~18 features up to 32
        normalize=True     # Forces the vector norm to equal 1.0
    )
    
    # 2. The Trainable Ansatz (The "Brain") This applies the parameterized rotations using the weights from PyTorch
    for layer in range(3): # "3 alternating layers"
        
        # Rotations (RY e RZ gates)
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
            
        # Entanglement (Circular arrangement of CNOTs)
        for i in range(n_qubits):
            # The operator modulo (%) ensures that the last qubit is entangled back to the first one, creating a loop
            target = (i + 1) % n_qubits 
            qml.CNOT(wires=[i, target])
    
    """
    # 3. NOISE INJECTION (Depolarizing channel after the ansatz)
    noise_probability = 0.01 # 1% chance of depolarization for each qubit
    for i in range(n_qubits):
        qml.DepolarizingChannel(noise_probability, wires=i)
    """
    
    # Return 5 real expectation values instead of the 32-dim state vector
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QUPIDModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # We need to tell PyTorch the shape of the quantum weights 
        # (This will depend on how many layers of gates we add later)
        weight_shapes = {"weights": (3, n_qubits, 2)} # Example shape
        custom_init = {"weights": init_angles}
        
        # We create 3 INDEPENDENT quantum layers from the same blueprint
        self.pqc_p1 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)
        self.pqc_p2 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)
        self.pqc_p3 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)
        self.pqc_p4 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method=custom_init)
        
        # The Classical part of the network that will take the merged quantum features
        # 4 PQC * 5 qubit = 20 feature total quantum features going into the classical dense layer
        self.classical_classifier = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: 2 classes
        )

    def forward(self, p1, p2, p3, p4):
        # PARALLEL EXECUTION (No loops!) Each layer extracts 5 real numbers from its respective relay
        out_1 = self.pqc_p1(p1) 
        out_2 = self.pqc_p2(p2)
        out_3 = self.pqc_p3(p3)
        out_4 = self.pqc_p4(p4)
        
        # CONCATENATION
        # We glue the 4 outputs together into a single flat vector of 20 features
        joined_quantum_features = torch.cat((out_1, out_2, out_3, out_4), dim=1)
        joined_quantum_features = joined_quantum_features.real.float() # Ensure the data type is float for the classical layer
        target_device = next(self.classical_classifier.parameters()).device
        joined_quantum_features = joined_quantum_features.to(target_device)
        final_prediction = self.classical_classifier(joined_quantum_features)
        
        return final_prediction

def main():
    complex_dataset = Path("dataset2/scenarios_concatenated/dataset_scaled_complex.npz")
    
    with np.load(complex_dataset, allow_pickle=True) as data:
        X_train_r1 = torch.tensor(data['x_train_r1'], dtype=torch.complex64)
        X_train_r2 = torch.tensor(data['x_train_r2'], dtype=torch.complex64)
        X_train_r3 = torch.tensor(data['x_train_r3'], dtype=torch.complex64)
        X_train_r4 = torch.tensor(data['x_train_r4'], dtype=torch.complex64)
        # Labels are classical integers, so we use torch.long
        y_train = torch.tensor(data['y_train'], dtype=torch.long)
        y_train_6_classes = [mapping_classes(y.item()) for y in y_train]

        # Now we have 6 classes.
        y_train_tensor = torch.tensor(y_train_6_classes, dtype=torch.long)

        """
        print(f"Feature of R1: {X_train_r1.shape[1]}")
        print(f"Feature of R2: {X_train_r2.shape[1]}")
        print(f"Feature of R3: {X_train_r3.shape[1]}")
        print(f"Feature of R4: {X_train_r4.shape[1]}")
        """
        
        X_train_r1_cleaned = X_train_r1[:, :16]
        X_train_r2_cleaned = X_train_r2[:, :16]
        X_train_r3_cleaned = X_train_r3[:, :16]
        X_train_r4_cleaned = X_train_r4[:, :16]

        BATCH_SIZE = 128
        dataset = torch.utils.data.TensorDataset(X_train_r1_cleaned, X_train_r2_cleaned, X_train_r3_cleaned, X_train_r4_cleaned, y_train_tensor)

        """
        y_numpy = y_train_tensor.numpy()
        class_counts = np.bincount(y_numpy) # Count how many samples we have for each class (0-5)
        class_weights_for_sampler = 1.0 / class_counts # We take the inverse of the counts to give more weight to minority classes
        sample_weights = [class_weights_for_sampler[label] for label in y_numpy] # We create a weight for each sample in the dataset based on its class label, so that the sampler can use these weights to perform weighted random sampling during training.
        
        # The WeightedRandomSampler will use the sample_weights to randomly select samples for each batch, giving higher probability to samples from underrepresented classes, which helps to mitigate class imbalance during training.
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        """

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice used: {device.type.upper()}")

        print("Instantiating the Hybrid QUPID Model...")
        model = QUPIDModel()
        model.to(device)

        total_samples = 47218
        samples_per_class = [13668, 33550]
        smoothed_weights = [math.sqrt(total_samples / count) for count in samples_per_class]
        weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=7, min_lr=0.00001
        )
        """

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
            recall_c0 = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)
            f1_c0 = f1_score(all_targets, all_preds, pos_label=0, zero_division=0)
            print(f"\n--- End of epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Accuracy: {epoch_accuracy:.2f}% ---\n")
            print(f"Recall for Class 0: {recall_c0:.4f} | F1 Score for Class 0: {f1_c0:.4f}")

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
                torch.save(model.state_dict(), f"qupid_model.pth")
                print(f"New best model saved with avg loss {best_loss:.4f} and accuracy {epoch_accuracy:.2f}%\n")

            #scheduler.step(avg_loss)
        
if __name__ == "__main__":
    main()