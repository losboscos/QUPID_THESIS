# QUPID_THESIS

# MAIN EXECUTION:
1. conversion.py: In this file every .arff file of dataset is converted to .csv.
2. dataset.py: In this file every csv is scaled: broken values removed, train-test, MinMaxScaler applied. It also contains mapping_classes (the single 6-class label mapping, imported everywhere).
3. to_complex.py: In this file the features of PMU's measurement are trasformed into complex values, in order to be given to quantum circuits.
4. pqnn.py: In this file there is the main code of QUPID: architecture and training.
5. r_pqnn.py: In this file there is the main code of R-QUPID: architecture and training.
6. evaluate.py: In this file there is the evaluation of base QUPID.
7. fgsm.py: In this file there is the evaluation of R-QUPID under FGSM. It reproduces the paper's ~81%: the amplitude-encoder input gradient is masked, so the attack is a no-op.
8. pgd.py: In this file there is the evaluation of R-QUPID under PGD. It reproduces the paper's ~79% with the same gradient masking.
9. eot.py: In this file there is the test with EoT (Expectation over Transformation) for attacks.

/dataset3: this folder contains the data pre-processing.
/MODELS_QUPID: this folder contains all clean models trained during the work. In "info.txt" all models are explained.
/MODELS_R-QUPID: this folder contains all robust models trained during the work. In "info.txt" all models are explained.
/BASELINES: this folder contains the baseline models with training files (FTTransformer.py, MTL_LSTM.py, QH_DNN.py), their FGSM (fgsm_baseline.py) and clean (evaluate_baseline.py) evaluation, and the noise-aware robust FTTransformer (Robust FTTransformer/r_FTTransformer.py).
/TRANSFER ATTACK: this folder contains the transfer (black-box) attacks: transfer_attacks.py crafts adversarials on a classical surrogate, transfer_eval.py sends one set to R-QUPID, transfer_heatmap.py builds the source-target transferability matrix.
/PARAMETER-SHIFT: this folder reproduces the paper's ~81/79 with the real quantum-native differentiation (parameter-shift), where the input gradient is NaN and the attack is a no-op. Contains copies of r_pqnn.py, fgsm.py, pgd.py, eot.py.
