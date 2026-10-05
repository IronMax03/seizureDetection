# EEG Seizure Detection — 13-Layer 1-D CNN (PyTorch)

This is a working progress experiment to create a new architecture for seizure detection using the CHB-MIT dataset. The architecture is based on the original paper [1] wich is exacly reimplemented in [CNN_Seizure_Detection](../CNN_Seizure_Detection/Notebook.ipynb). 


## What's in the repo


## Data setup

Before runing the notebook, you set-up and run [../CHB-MIT_Formating/Reformating.ipynb](Reformating.ipynb) in [../CHB-MIT_Formating/README.md](CHB-MIT_Formating) to prepare the training data.

## Results
| Metric | reimplementation results | paper results |
|--------|--------------------------|---------------|
| accuracy           | 91.00% (+/- 4.73) | 88.67% |
| sensitivity        | 95.50%            | 95.00% |
| specificity        | 91.00%            | 90.00% |

The reimplementation accuracy is slightly higher than the original paper's, but lies inside the margin of error. This and the other metrics small differences is likely due to the framework differences (2026 PyTorch vs. 2018 MATLAB).

<div align="center">

  <img width="25%" alt="radioactive shielding simulation" src="graphs/Reproduction/Confusion_Matrix.png" /> 
  <img width="60%" alt="mandelbrot_zoom1" src="graphs/Reproduction/Learning_Curve.png" />


## Citations
