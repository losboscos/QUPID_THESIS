import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

dataset = Path("dataset/multiclass")

def main():
    dataframe = []
    for file in dataset.glob("*.csv"):
        df_temp = pd.read_csv(file)
        dataframe.append(df_temp)

    # Only one big table containing with all csv unified
    df_total = pd.concat(dataframe, ignore_index=True)
    print(f"Total dimension: {df_total.shape}")

    # Conversion of inf values
    df_total.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Counting of broken values
    corrupted_rows = df_total.isna().any(axis=1).sum()
    print(f"Found {corrupted_rows} rows with (NaN o Inf).")

    #Eliminate them
    df_total.dropna(inplace=True)
    print(f"Dataset dimension post-cleaning: {df_total.shape}")

    X = df_total.drop(columns=['marker'])
    y = df_total['marker']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. ONLY TRANSFORM on Test Set.
    # The scaler uses old min and max of train to scale new data.
    X_test_scaled = scaler.transform(X_test)

    print("Scaling completed successfully")

    # Saving data locally
    output = Path("dataset/splitted_data")

    # Creating folder if not exists
    os.makedirs(output, exist_ok=True)

    features = X.columns.to_numpy()

    file_output = output / "dataset_scaled.npz"

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