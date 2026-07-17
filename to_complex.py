import numpy as np
from pathlib import Path

def complex_partition(dataset, features, pmu_prefix):
    """
    Extracts features of one single PMU e converts it to complex.
    """
    complex_features = []
    
    num_relay = pmu_prefix[1:] 
    log_name = f"relay{num_relay}_log"

    indexes_pmu = [i for i, col in enumerate(features) if col.startswith(pmu_prefix) or col == log_name]
    processed_features = set()
    
    for i in indexes_pmu:
        if i in processed_features:
            continue

        col_name = features[i]
            
        # CASE 1: It's a Phasor (Voltage, Current, or Z Impedance)
        # We look for the phase angle (PA)
        if 'PA' in col_name and not col_name.endswith(':Z') and not col_name.endswith(':ZH'):
            # Find the corresponding magnitude (PM)
            magnitude_name = col_name.replace('PA', 'PM').replace('VH', 'V').replace('IH', 'I')
            
            try:
                magnitude_index = features.index(magnitude_name)
                
                angle = dataset[:, i]
                magnitude_v = dataset[:, magnitude_index]
                
                # Euler's formula: z = V * exp(i * angle).
                # The PA column is an angle in DEGREES (-180..180) that MinMaxScaler squashed to
                # [0,1]. Feeding it straight in as radians confines every phasor to a ~1 rad (57 deg)
                # cone, so all encoded states end up in the same positive orthant (measured:
                # min |<x|x'>|^2 = 0.19, vs ~1/32 = 0.03 for well-spread 32-dim states) -> razor-thin
                # class margins -> catastrophic adversarial fragility. Rescale to the full circle.
                complex_vector = magnitude_v * np.exp(1j * 2.0 * np.pi * angle)
                complex_features.append(complex_vector)
                
                processed_features.add(i)
                processed_features.add(magnitude_index)
                continue
            except ValueError:
                print(f"Warning: Magnitude {magnitude_name} not found for phase {col_name}")

        elif 'PA' in col_name and col_name.endswith(':Z'):
            angle_name = col_name.replace('PA:Z', 'PA:ZH')

            try:
                angle_index = features.index(angle_name)
                
                angle = dataset[:, angle_index]
                magnitude_v = dataset[:, i]
                
                # Same as above
                complex_vector = magnitude_v * np.exp(1j * 2.0 * np.pi * angle)
                complex_features.append(complex_vector)
                
                processed_features.add(i)
                processed_features.add(angle_index)
                continue
            except ValueError:
                print(f"Warning: Magnitude {magnitude_name} not found for phase {col_name}")
        
        # Skip if it's a magnitude column but we haven't processed its phase yet
        if 'PM' in col_name or col_name.endswith(':ZH'):
            continue
            
        # CASE 2: It's a Real scalar variable (F, DF, S, or the relay_log)
        real_value = dataset[:, i]
        complex_vector = real_value + 0j
        complex_features.append(complex_vector)
        processed_features.add(i)

    # Stack all extracted features into a single 2D matrix
    return np.column_stack(complex_features)

def main():
    dataset_archive = Path("dataset3/scenarios")
    output = Path("dataset3/scenarios_complex")

    for file in dataset_archive.glob("*.npz"):
    
        with np.load(file, allow_pickle=True) as data:
            X_train_scaled = data['x_train']
            X_test_scaled = data['x_test']
            y_train = data['y_train']
            y_test = data['y_test']
            features = data['features'].tolist()

        print("Processing Training Set...")
        X_Train_R1_complex = complex_partition(X_train_scaled, features, 'R1')
        X_Train_R2_complex = complex_partition(X_train_scaled, features, 'R2')
        X_Train_R3_complex = complex_partition(X_train_scaled, features, 'R3')
        X_Train_R4_complex = complex_partition(X_train_scaled, features, 'R4')

        print("Processing Test Set...")
        X_Test_R1_complex = complex_partition(X_test_scaled, features, 'R1')
        X_Test_R2_complex = complex_partition(X_test_scaled, features, 'R2')
        X_Test_R3_complex = complex_partition(X_test_scaled, features, 'R3')
        X_Test_R4_complex = complex_partition(X_test_scaled, features, 'R4')

        file_output = output / f"{file.stem}_complex.npz"

        np.savez_compressed(
            file_output,
            x_train_r1=X_Train_R1_complex,
            x_train_r2=X_Train_R2_complex,
            x_train_r3=X_Train_R3_complex,
            x_train_r4=X_Train_R4_complex,
            x_test_r1=X_Test_R1_complex,
            x_test_r2=X_Test_R2_complex,
            x_test_r3=X_Test_R3_complex,
            x_test_r4=X_Test_R4_complex,
            y_train=y_train,
            y_test=y_test
        )
        
        print("Quantum data preparation completed successfully")

if __name__ == "__main__":
    main()