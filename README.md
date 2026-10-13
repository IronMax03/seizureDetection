# seizureDetection

Work-in-progress research repository developing an original architecture for EEG-based seizure detection. The project reproduces established methods from the literature as verified baselines, then builds on them through successive experiments toward a performant, deployable detector. The current best architecture is FiCRNN (see Latest Results); it is the leading candidate so far, not the final design the architecture is still evolving as experiments continue.

> [!WARNING]
> **This project and any future software produced from it are not intended for
> clinical use, medical diagnosis, treatment decisions, or emergency response.**
> The experiments are research and engineering work in progress, and their
> results must not be treated as medical advice or as a validated clinical
> device.

## Project objective

The project has three connected objectives:

1. **Reproduce** established seizure-detection methods from the literature (CNN classification, entropy features, wavelet artifact removal) as verified baselines, confirming reported results before building on them.
2. **Develop** an original architecture by iterating on these baselines — currently FiCRNN, which conditions a CNN–RNN on signal-derived entropy features evaluated honestly under patient-wise cross-validation and extreme class imbalance. The architecture is an active work in progress.
3. **Deliver** the resulting architecture as usable software, with a reproducible preprocessing and inference pipeline and a clear interface.

## Latest Results

The current best is **FiCRNN**: FiLM conditioning on CNN with an RNN classifier. the features used by FiLM are approximate entropy (ApEn) for each window. It was trained on raw CHB-MIT EEG
with 1 s windows, sequences of 32 windows, 50% of the data, and evaluated with
patient-wise 10-fold cross-validation.

| Metric            | FiCRNN + ApEn (CHB-MIT) |
|-------------------|-------------------------|
| Balanced accuracy | 79.47% ± 10.93          |
| AUPRC             | 0.586 ± 0.236           |

Seizure windows make up only ~0.1–0.7% of CHB-MIT, so balanced accuracy and
AUPRC (area under the precision-recall curve) are the primary metrics; plain accuracy is not informative here. 

<div align="center">
  <p><b>FiCRNN detailed architecture</b></p>
  <img width="100%" alt="fig2" src="./doc/figs/FiCRNN/ficrnn_detailed.png" />
  <p><b>FiCRNN unrolled over time</b></p>
  <img width="100%" alt="fig3" src="./doc/figs/FiCRNN/ficrnn_unrolled.png" />
</div>

### Architecture

FiCRNN classifies EEG one 1-second window at a time, using the previous
windows as context. It has two parts:

1. **FiLM-CNN encoder.** A 1D CNN of five convolutional blocks turns each
   23-channel window (256 samples) into a 40-dimensional feature vector.
   A small FiLM generator maps the window's per-channel approximate entropy
   (ApEn) to a scale γ and shift β for every channel of every block. These
   modulate the normalized convolution outputs, so the entropy features steer
   which patterns the CNN emphasises. The same encoder, with the same
   weights, is applied to every window.
2. **Recurrent classifier.** A unidirectional RNN (16 hidden units) reads the
   sequence of window features and carries a hidden state from one window to
   the next. A dropout and linear layer then outputs seizure / non-seizure
   logits for every window. Because the RNN only looks backward, each
   prediction uses only the current and past windows, which keeps the model
   usable in real time.

Regularisation includes channel dropout in the conv blocks, dropout in the
FiLM generator and before the output layer, and a zero-initialised FiLM
generator, so training starts from an unmodulated CNN.

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
- **CHB-MIT Scalp EEG Database:** used as 1 second windows for FiCRNN.

## TODO
- [x] Reproduce Acharya CNN on Bonn EEG.
- [x] Reproduce & evaluate entropy features on Bonn EEG.
- [x] Reproduce wavelet based artifact removal on EEGdenoiseNet.
- [x] Integrate CHB-MIT Scalp EEG Database and window it for seizure/non-seizure analysis.
- [x] Extend Acharya CNN with FiLM conditioning and denoising preprocessing using CHB-MIT Scalp EEG Database.
- [x] Experiment with recurrent neural networks integrated in the CNN architecture.
- [ ] Experiment with surface Laplacian preprocessing on CHB-MIT Scalp EEG Database.

- [ ] implement the final architecture in C++.
- [ ] document the resulting C++ implementation and provide a command line interface for inference.

## Sources
> [1] U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, and H. Adeli, “Deep Convolutional Neural Network for the Automated Detection and Diagnosis of Seizure Using EEG Signals,” Computers in Biology and Medicine, vol. 100, pp. 270–278, Sept. 2018, doi: 10.1016/j.compbiomed.2017.09.017.

> [2] K. N., M. L. Choo, U. R. Acharya, and Sadasivan, P.K, “Entropies for Detection of Epilepsy in EEG,” Computer Methods and Programs in Biomedicine, vol. 80, no. 3, pp. 187–194, Dec. 2005, doi: 10.1016/j.cmpb.2005.06.012.

> [3] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual Reasoning With a General Conditioning Layer,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018, doi: 10.1609/aaai.v32i1.11671.

> [4] N. Bajaj, Wavelets for EEG Analysis. IntechOpen, 2020. Accessed: Sept. 17, 2026. [Online]. Available: https://www.intechopen.com/chapters/74032

> [5] N. S. Amer and S. B. Belhaouari, “EEG Signal Processing for Medical Diagnosis, Healthcare, and Monitoring: A Comprehensive Review,” IEEE Access, vol. 11, pp. 143116–143142, 2023, doi: 10.1109/access.2023.3341419.
