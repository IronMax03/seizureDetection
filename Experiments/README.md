# Experiments Guide

This directory contains independent experiments that support the seizure
detection project. Each experiment is kept relatively self-contained, with its
own notebook, scripts, data, figures, and README where available. The
individual READMEs are the primary reference for setup details and reported
results.

## Experiment index

| Experiment | Purpose | Documentation |
|---|---|---|
| `Artifact_Removal/` | Tests global thresholding, standard-deviation thresholding, and ATAR wavelet artifact removal on EEGdenoiseNet. | [README](Artifact_Removal/README.md) |
| `CHB-MIT_Formating/` | Downloads and windows the CHB-MIT Scalp EEG Database for seizure/non-seizure analysis. | [README](CHB-MIT_Formating/README.md) |
| `CNN_Seizure_Detection/` | Reproduces Acharya et al.'s 13-layer 1-D CNN on three Bonn EEG classes. | [README](CNN_Seizure_Detection/README.md) |
| `Entropy_Based_Fetures/` | Evaluates Kolmogorov-Sinai, spectral, approximate, and Renyi entropy features with XGBoost on Bonn EEG. | [README](Entropy_Based_Fetures/README.md) |
| `CNN_Seizure_Detection_improved/` | In-progress FiLM-conditioned CNN, CNN–RNN and CNN–GRU models on CHB-MIT, conditioned on ApEn. | [README](CNN_Seizure_Detection_improved/README.md), [Results](CNN_Seizure_Detection_improved/Results.md) |
| `Architecture_Optimizer/` | In-progress Optuna search over the FiCRNN-GRU architecture (FiLM depth, kernel sizes, channels, generator depth, GRU width, dropouts, learning rate) with k-fold CV on CHB-MIT patients. | README pending |
| `ECG_denoising/` | Separate exploratory work on signal denoising of a simulated ECG signal.  | README pending |

## Data and execution notes

- Run notebooks from their experiment directory unless the notebook specifies
  another working directory.
- Keep downloaded datasets in the locations expected by the corresponding
  README. Do not assume that datasets share sampling rates, labels, or window
  lengths.
- For classification comparisons, document the dataset split and ensure that
  windows from the same recording or patient do not leak across splits.
- Generated figures belong in the experiment's `graphs/` directory (training
  runs of `CNN_Seizure_Detection_improved/` write theirs to `results/`), or in
  `doc/visuals/` when they are used by repository-level documentation.
