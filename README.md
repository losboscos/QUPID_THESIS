# QUPID_THESIS

# MAIN EXECUTION:
1. conversion.py: In this file every .arff file of dataset is converted to .csv.
2. dataset.py: In this file every csv is scaled: broken values removed, train-test, MinMaxScaler applied. It also contains mapping_classes (the single 6-class label mapping, imported everywhere).
3. to_complex.py: In this file the features of PMU's measurement are trasformed into complex values, in order to be given to quantum circuits.
4. pqnn.py: In this file there is the main code of QUPID: architecture and training.
5. r_pqnn.py: In this file there is the main code of R-QUPID: architecture and training.
6. evaluate.py: In this file there is the evaluation of base QUPID.
7. fgsm.py: In this file there is the evaluation of R-QUPID under FGSM, with an honest attack (real backprop gradient, no masking). eps=0 gives the clean-smoothed ~81% (the value the paper reports), eps>0 the real fair L-inf attack (~35%): the gap between the two rows is the result.
8. pgd.py: In this file there is the same for PGD. The masked reproduction of the paper's ~81/79 is in /PARAMETER-SHIFT, where the quantum-native gradient is NaN on its own.
9. eot.py: In this file there is the test with EoT (Expectation over Transformation) for attacks.

/dataset3: this folder contains the data pre-processing.
/MODELS_QUPID: this folder contains all clean models trained during the work. In "info.txt" all models are explained.
/MODELS_R-QUPID: this folder contains all robust models trained during the work. In "info.txt" all models are explained.
/BASELINES: this folder contains the baseline models with training files (FTTransformer.py, MTL_LSTM.py, QH_DNN.py), their FGSM (fgsm_baseline.py) and clean (evaluate_baseline.py) evaluation, and the noise-aware robust FTTransformer (Robust FTTransformer/r_FTTransformer.py).
/TRANSFER ATTACK: this folder contains the transfer (black-box) attacks: transfer_attacks.py crafts adversarials on a classical surrogate, transfer_eval.py sends one set to R-QUPID, transfer_heatmap.py builds the source-target transferability matrix.
/PARAMETER-SHIFT: this folder reproduces the paper's ~81/79 with the real quantum-native differentiation (parameter-shift), where the input gradient is NaN and the attack is a no-op. Contains copies of r_pqnn.py, fgsm.py, pgd.py, eot.py.
/QUANTUM-NOISE: Standalone R-QUPID with the channel active (p=0.05) and simulated finite-shot noise, noise-aware training. Result: no robustness gain.
/CONSISTENCY: Consistency regularization at the sigma where the classical baseline is robust. Result: stuck at the majority class already on the training set.
