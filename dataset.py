import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

dataset = Path("dataset3/multiclass")

# data1 is the reference scenario the whole work is built on: its .npz must not be regenerated
# (a new run would rebuild the split and invalidate every trained model). Empty this set only for
# a deliberate full regeneration from scratch.
PROTECTED = {"data1"}

def mapping_classes(old_marker):
    """Canonical 6-class label mapping, single source of truth for the whole repo.
    Imported everywhere as `from dataset import mapping_classes`."""
    if old_marker == 41:
        return 0  # Class 0: Normal
    elif 1 <= old_marker <= 6:
        return 1  # Class 1: Natural Events (Short Circuits)
    elif 13 <= old_marker <= 14:
        return 2  # Class 2: Natural Events (Line Maintenance)
    elif 7 <= old_marker <= 12:
        return 3  # Class 3: Data Injection Attacks
    elif 15 <= old_marker <= 20:
        return 4  # Class 4: Remote Tripping Command Injection
    elif 21 <= old_marker <= 40:
        return 5  # Class 5: Relay Setting Change
    else:
        raise ValueError(f"Unknown marker: {old_marker}")

def main():
    dataframe = []
    for file in dataset.glob("*.csv"):
        if any(file.name.startswith(p + " ") for p in PROTECTED):
            print(f"Skipping {file.name} (protected reference scenario)")
            continue

        df = pd.read_csv(file)
        dataframe.append(df)

        print(f"Dimension: {df.shape}")

        # Conversion of inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Counting of broken values
        corrupted_rows = df.isna().any(axis=1).sum()
        print(f"Found {corrupted_rows} rows with (NaN o Inf).")

        #Eliminate them
        df.dropna(inplace=True)
        print(f"Dataset dimension post-cleaning: {df.shape}")

        X = df.drop(columns=['marker'])
        y = df['marker']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = MinMaxScaler()
        # Fit transform on Train Set. 
        # Because the scaler is fitted on the train set, it will learn the min and max values of the train set.
        X_train_scaled = scaler.fit_transform(X_train)

        # ONLY TRANSFORM on Test Set.
        # The scaler uses old min and max of train to scale new data.
        X_test_scaled = scaler.transform(X_test)

        print("Scaling completed successfully")

        # Saving data locally
        output = Path("dataset3/scenarios")

        # Creating folder if not exists
        os.makedirs(output, exist_ok=True)

        features = X.columns.to_numpy()

        file_output = output / f"{file.stem}_scaled.npz"

        print(f"Save in progress on {file_output}...")

        # np.savez_compressed saves several array on same binary file with zip compression
        np.savez_compressed(
            file_output,
            x_train=X_train_scaled,
            x_test=X_test_scaled,
            y_train=y_train.to_numpy(), # to_numpy reset the index
            y_test=y_test.to_numpy(),
            features=features
        )

        print("Save completed")

if __name__ == "__main__":
    main()