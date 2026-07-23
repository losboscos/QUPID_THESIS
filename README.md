# QUPID_THESIS

# MAIN EXECUTION:
1. conversion.py: In this file every .arff file of dataset is converted to .csv.
2. dataset.py: In this file every csv is scaled: broken values removed, train-test, MinMaxScaler applied.
3. to_complex.py: In this file the features of PMU's measurement are trasformed into complex values, in order to be given to quantum circuits.
4. pqnn.py: In this file there is the main code of QUPID: architecture and training.
5. r_pqnn.py: In this file there is the main code of R-QUPID: architecture and training.
6. evaluate.py: In this file there is the evaluation of base QUPID.
7. fgsm.py: In this file there is the evaluation of R-QUPID.

/dataset3: this folder contains the data pre-processing.
/MODELS_QUPID: this folder contains all models trained during the work. In "info.txt" all models are explained.

