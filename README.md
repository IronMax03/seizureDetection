# seizureDetection

Work in progress research repository for EEG preprocessing, feature extraction,
and seizure classification. The project collects paper reproductions and
experimental code while working toward a usable seizure-detection software
application.

> [!WARNING]
> **This project and any future software produced from it are not intended for
> clinical use, medical diagnosis, treatment decisions, or emergency response.**
> The experiments are research and engineering work in progress, and their
> results must not be treated as medical advice or as a validated clinical
> device.

## Project objective

The project has two connected objectives:

1. Experiment with EEG preprocessing, signal-derived features, and neural
    network architectures to identify a promising seizure-detection approach.
2. Implement the resulting architecture as usable software, with a clear
    interface and reproducible processing and inference pipeline.

The repository combines reproducible baselines with experiments on artifact
removal, entropy features, convolutional neural networks, and larger clinical
EEG datasets. The architecture and software interface are still evolving.

## Repository structure

```text
.
|-- Experiments/
|   |-- Artifact_Removal/              Wavelet and thresholding denoising tests
|   |-- CHB-MIT_Formating/             CHB-MIT download and windowing pipeline
|   |-- CNN_Seizure_Detection/         Acharya CNN reproduction
|   |-- ECG_denoising/                 Separate ECG denoising exploration
|   |-- Entropy_Based_Fetures/         Entropy estimators and classification
|-- src/                               Shared preprocessing and data exploration
|-- doc/visuals/                       Generated figures used by documentation
`-- README.md                          Project overview
```

The `Experiments/` directory has its own guide. Start with
[Experiments/README.md](Experiments/README.md) for experiment purposes,
datasets, entry points, and links to the individual READMEs.

## Datasets

No datasets are included in this repository. The following datasets are used by the experiments:
- **Bonn University EEG:** used by the CNN and entropy experiments.
- **EEGdenoiseNet:** used to test artifact-removal methods.
- **CHB-MIT Scalp EEG Database:** currently being integrated for a larger,
  clinically oriented seizure/non-seizure dataset.
  preprocessing work.

## TODO

- [x] Reproduce Acharya CNN on Bonn EEG.
- [x] Reproduce & evaluate entropy features on Bonn EEG.
- [x] Reproduce wavelet based artifact removal on EEGdenoiseNet.
- [x] Integrate CHB-MIT Scalp EEG Database and window it for seizure/non-seizure analysis.
- [ ] Experiment with surface Laplacian preprocessing on CHB-MIT Scalp EEG Database.
- [ ] Extend Acharya CNN with FiLM conditioning and denoising preprocessing using CHB-MIT Scalp EEG Database.
- [ ] Experiment with recurent neural networks integrated in the CNN architecture.

- [ ] implement the final architecture in C++.
- [ ] document the resulting C++ implementation and provide a command line interface for inference.