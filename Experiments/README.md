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
| `CNN_Seizure_Detection_improved/` | In-progress CNN extension using entropy features for FiLM conditioning and denoising preprocessing. | README pending |
| `ECG_denoising/` | Separate exploratory work on ECG denoising and a small CNN model. | README pending |

## Suggested reading order

1. Read the [Acharya CNN reproduction](CNN_Seizure_Detection/README.md) to see
   the baseline classification task and Bonn label mapping.
2. Read [Entropy Based Features](Entropy_Based_Fetures/README.md) for the
   conditioning features used by the improved architecture.
3. Read [Artifact Removal](Artifact_Removal/README.md) for the denoising
   experiments used as preprocessing candidates.
4. Read [CHB-MIT Preprocessing](CHB-MIT_Formating/README.md) for the current
   path toward a larger clinical dataset.

## How the experiments fit together

```text
Bonn EEG
   ├── baseline 13-layer CNN
   └── entropy feature evaluation

EEGdenoiseNet
   └── artifact-removal method comparison

Entropy features + artifact removal
   └── improved CNN with FiLM conditioning

CHB-MIT EEG
   └── preprocessing and integration in progress
```

The improved architecture is an active work in progress. Its current design
uses entropy features to generate FiLM parameters that modulate intermediate
CNN feature maps, while denoising is treated as an upstream preprocessing
step. The Bonn dataset is too small and clean for its k-fold statistics to be
treated as a strong generalization result.

## Data and execution notes

- Run notebooks from their experiment directory unless the notebook specifies
  another working directory.
- Keep downloaded datasets in the locations expected by the corresponding
  README. Do not assume that datasets share sampling rates, labels, or window
  lengths.
- For classification comparisons, document the dataset split and ensure that
  windows from the same recording or patient do not leak across splits.
- Generated figures belong in the experiment's `graphs/` directory or in
  `doc/visuals/` when they are used by repository-level documentation.

## Documentation convention

When adding an experiment, include a README that states:

- the research question and referenced method;
- the dataset and download or placement instructions;
- the role of each script or notebook;
- the preprocessing, labels, and evaluation protocol;
- current results, limitations, and open questions.