# Project 4: Anatomically-Constrained-LDMs-for-Rare-Case-Augmentation


## Team

leparco.2285077@studenti.uniroma1.it

---

## Project Overview

Periodontitis is a severe gum infection. The severe stage of this desease can't be determined well by computer vision because of disparity between classes. We want to implement Anatomically-Constrained LDMs to help solve this problem.

---

## Repository Structure

Anatomically_Constrained_LDMs.ipynb        
presentation       
results             
README.md

The main follows the required structure:
- **Imports** — all needed packages
- **Globals** — global variables and hyperparameters
- **Utils** — helper functions
- **Data** — dataset loading and preprocessing
- **Network** — model architecture definition
- **Train** — training loop and loss functions
- **Evaluation** — metrics, visualizations, ablation results

---

## Dataset

**Name:** perio-KPT

**Source:** [\[Link to dataset — Kaggle / GitHub / PapersWithCode / academic repository\] ](https://zenodo.org/records/17272200?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImUyOGRhOTg3LTM0NmYtNGRiMC1iOGI4LWM4ODhmZGZmMWFjYyIsImRhdGEiOnt9LCJyYW5kb20iOiIzOGUwOGY1MmIxNGU4NWQ0YmVkZTQyZTY2MDI5NmU0YyJ9.wx0AXhIB7sBS0qQPxidRLha56XznHGCKCZQpvrtVNAzWDXIkszGl5IaXip3FNR_WH84r8o1DetfrAGkC1EyBTQ) 

**Description:** 

The dataset used includes bounding boxes, keypoints, and instance segmentation masks for dental structures. It provides the necessary anatomical annotations  to condition the LDM with gaussian heatmap 

---

## Environment & Requirements

**Framework:** PyTorch   
**Platform:** Kaggle 


**Key dependencies:**

torch: 2.10.0+cu128
numpy: 2.4.6
Pillow: 11.3.0
opencv: 4.13.0
sklearn: 1.6.1
torchvision: 0.25.0+cu128
seaborn: 0.13.2
matplotlib: 3.10.0
tqdm: 4.67.3


## How to Run

1. Go to Anatomically_Constrained_LDMs.ipynb
2. Click **Copy & Edit** to fork the notebook
3. Under **Session options**, enable **GPU accelerator**
4. Set the dataset path in the **Globals** cell:
   DATA_ROOT = Path("/kaggle/input/your-dataset-path/perio_KPT")
5. Run all cells sequentially (**Run All**)

##  Methodology

### Problem Statement

Create a LDM that is Anatomically-Constrained to handle the repartition of severity classes for periodontitis

### Baseline and Proposed Novelty

A BBOX LDM and an LDM with gaussian heatmap to find the best mask to guide the LLM

### References

1.  Ryan Banks, Periodontal Bone Loss Analysis via Keypoint Detection With Heuris
tic Post-Processing.
2. Ho, J., Jain, A., & Abbeel, P. ”Denoising Diffusion Probabilistic Models”. NeurIPS 2020.
3. Kazerouni, A., et al. ”Diffusion models in medical imaging: A comprehensive survey”, Medical Image Analysis
4. Pinaya, W. H. L., et al. ”Brain Imaging Generation with Latent Diffusion Models”. MICCAI 2022.
5. Konz, N., et al. ”Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion
Models”. MICCAI 2024


## Reproducibility

To guarantee exact reproducibility of all reported results:

- Random seeds are fixed throughout the codebase (`SEED = 42` by default).
- All hyperparameters are centralized in the **Globals** section.
- Training is deterministic (`torch.backends.cudnn.deterministic = True`).

```python
import torch, random, numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

---
