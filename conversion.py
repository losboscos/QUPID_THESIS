from pathlib import Path
import pandas as pd
from scipy.io import arff

dataset = Path("dataset2/multiclass")

def main():
    for file_arff in dataset.glob("*.arff"):
        print(f"Conversion of: {file_arff.name}")
        
        try:
            data, meta = arff.loadarff(file_arff)
            df = pd.DataFrame(data)
            
            #transforming raw data
            for col in df.select_dtypes([object]).columns:
                df[col] = df[col].str.decode('utf-8')
                
            file_csv = file_arff.with_suffix("")
            
            df.to_csv(file_csv, index=False)
            
            print(f"Saved successfully: {file_csv.name}\n")
            
        except Exception as e:
            print(f"Error in reading {file_arff.name}: {e}")


    for file in dataset.glob("*.csv"):
        df = pd.read_csv(file)
        print(f"Reading: {file.name}")
        columns_to_remove = ["snort_log1", "snort_log2", "snort_log3", "snort_log4", "control_panel_log1", "control_panel_log2", "control_panel_log3", "control_panel_log4"]
        df.drop(columns=columns_to_remove, inplace=True)
        df.to_csv(file, index=False)

if __name__ == "__main__":
    main()

