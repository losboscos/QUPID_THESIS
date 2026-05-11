# QUPID_THESIS

# MAIN EXECUTION:
# 1. conversion.py: In this file every .arff file of dataset is converted to .csv.
# 2. dataset.py: In this file every csv is scaled: broken values removed, train-test, MinMaxScaler applied.
# 3. to_complex.py: In this file the features of PMU's measurement are trasformed into complex values, in order to be given to quantum circuits.
# 4. pqnn.py: In this file there is the main code of QUPID: architecture and training.
# 5. evaluate.py: In this file there is the evaluation of QUPID models.

# /dataset2: this folder contains the data pre-processing, considering the new approach that replaces the feature "relay_log" with the feature "Status Flag".
# /MODELS_QUPID: this folder contains all models trained during the work. In "info.txt" all models are explained.

# OTHER FILES (they are no longer useful, but I describe them for completeness):
# - pqnn_bin.py was the same code of pqnn.py but concerning the binary classification.
# - mlp_bin.py, MLP_classifier.py, mlp.py are files concerning the implementation of a MLP model, as the folder "/MODELS_MLP".
# - /dataset: this folder contains the older approach as told in /dataset2 description.