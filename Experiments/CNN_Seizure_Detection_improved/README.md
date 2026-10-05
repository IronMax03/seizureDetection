# EEG Seizure Detection — New Architecture
This is a working progress experiment to create a new architecture for seizure detection using the CHB-MIT dataset. The architecture is based on the original paper [1] wich is exacly reimplemented in [CNN_Seizure_Detection](../CNN_Seizure_Detection/Notebook.ipynb). 

> [!WARNING]
> **This is a work in progress.** The architecture is not yet fully implemented and the results are not final. The notebook is not yet fully documented and the code is not yet fully tested.
## What's in the repo


## Data setup

Before runing the notebook, you set-up and run [../CHB-MIT_Formating/Reformating.ipynb](Reformating.ipynb) in [../CHB-MIT_Formating/README.md](CHB-MIT_Formating) to prepare the training data.

## Results
| Metric |Current Architecture (CHB-MIT)| paper reimplementation results (Bonn) | paper results (Bonn) |
|--------------------|--------|-------------------|---------------|
| balanced accuracy  | 92.91% +/- 3.67 |        |        |
| accuracy           | 99.56% | 91.00% (+/- 4.73) | 88.67% |
| sensitivity        | 86.2%  | 95.50%            | 95.00% |
| specificity        | 99.59% | 91.00%            | 90.00% |

Note that CHB-MIT as about 0.1% to 0.7% of seizure data, while Bonn has 50% of seizure data. Because of that the balanced accuracy is a better metric to evaluate the model performance on CHB-MIT dataset. 

<div align="center">
  <img width="100%" alt="mandelbrot_zoom1" src="graphs/First_run.png" />


## Citations
