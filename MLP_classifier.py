from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from pathlib import Path
from pqnn import mapping_classes
import torch
import numpy as np

dataset = Path("dataset2/scenarios/data1 Sampled Scenarios_scaled.npz")
    
with np.load(dataset, allow_pickle=True) as data:
    X_train_scaled = data['x_train']
    X_test_scaled = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    features = data['features'].tolist()

    y_train_classes = [mapping_classes(y.item()) for y in y_train]
    y_test_classes = [mapping_classes(y.item()) for y in y_test]

    indexes_to_keep = [i for i, col in enumerate(features) if "R" in col]
    print(f"Selected {len(indexes_to_keep)} features for MLP Classifier.")

    X_train_mlp = X_train_scaled[:, indexes_to_keep]
    X_test_mlp = X_test_scaled[:, indexes_to_keep]
    features_mlp = [features[i] for i in indexes_to_keep]

# Initialize the MLP Classifier
mlp_baseline = MLPClassifier(random_state=42, max_iter=1000) 

# Train the model
mlp_baseline.fit(X_train_mlp, y_train_classes)

# Predictions on the test set
y_pred = mlp_baseline.predict(X_test_mlp)

# Metrics
print("Accuracy:", accuracy_score(y_test_classes, y_pred))
"""
print("Recall Class 0:", recall_score(y_test_classes, y_pred, pos_label=0))
print("F1-Score Class 0:", f1_score(y_test_classes, y_pred, pos_label=0))
"""