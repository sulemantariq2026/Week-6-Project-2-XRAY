# Chest X-ray Pneumonia Classification

A PyTorch pipeline that classifies chest X-rays as **NORMAL** or **PNEUMONIA**, built with a focus on trustworthy evaluation: patient-level splits, leakage and duplicate checks, a recall-targeted decision threshold, prevalence-aware PPV, and Grad-CAM explanations.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for clinical decisions.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Results](#results)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Accuracy alone is a poor way to judge a medical imaging model. This project goes further by:

1. Indexing every image with a **patient group ID** so that no patient appears in both train and validation.
2. Handling **class imbalance** with a weighted loss.
3. Choosing a decision **threshold on the validation set** to reach a target recall (sensitivity), then reporting test performance at that threshold.
4. Showing how **PPV collapses as disease prevalence falls**, for example in community screening at 5% prevalence.
5. Auditing the data for **exact and near duplicates** and **group overlap** between the official train and test splits.
6. Explaining predictions with **Grad-CAM** heatmaps for both correct and incorrect cases.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al.), available on Kaggle.

Expected layout:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/        # not used (only 16 images); validation is split from train instead
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset is **not included** in this repository. Download it from Kaggle and place it anywhere under the working directory, `/kaggle/input`, `/content` or `/mnt`. The code finds it automatically.

## Pipeline

| Step | Task | Description |
|------|------|-------------|
| 1 | Indexing | One row per image: path, label, file name, patient group |
| 2 | Split | Group-aware train/validation split (`GroupShuffleSplit`, 85/15) |
| 3 | Training | ResNet18 (ImageNet weights), class-weighted cross-entropy, Adam, best checkpoint by validation AUC |
| 4 | Test report | Classification report, macro-F1, confusion matrix at the default argmax threshold |
| 5 | Threshold | Threshold chosen on validation for a target recall, then evaluated on test |
| 6 | Prevalence | PPV curve across prevalence values, with test and 5% screening points |
| 7 | Grad-CAM | Heatmaps on `layer4[-1]` for correct and incorrect predictions |
| 8 | Audit | Exact duplicates (file hash), near duplicates (thumbnail distance), group overlap |
| 9 | Export | Model checkpoint, `metrics.json`, per-image test predictions |

## Project Structure

```
.
├── chest_xray_complete.py   # Full pipeline (cell-marked with `# %%`)
├── figures/                 # Generated plots
├── outputs/                 # Model, metrics and predictions
├── requirements.txt
└── README.md
```

Rename `chest_xray_complete.py` to match your repository. If you work in a notebook, each `# %%` marker is one cell.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
scipy
Pillow
```

A GPU is strongly recommended. The code falls back to CPU, but training will be slow.

## Usage

**Script:**

```bash
python chest_xray_complete.py
```

**Jupyter / Colab / Kaggle:**

```python
%run chest_xray_complete.py
```

On Kaggle, turn **Internet ON** so the pretrained ResNet18 weights can download. Without internet, the model trains from scratch and accuracy will be lower.

### Configuration

Edit the config block at the top of the script:

| Variable | Default | Meaning |
|----------|---------|---------|
| `POSITIVE` | `'PNEUMONIA'` | Positive class (must match a folder name) |
| `TARGET_RECALL` | `0.95` | Recall the threshold must reach on validation |
| `GROUP_PATTERN` | `r'(person\d+)'` | Regex that extracts the patient ID from file names (`None` to disable) |
| `IMG_SIZE` | `224` | Input resolution |
| `BATCH` | `32` | Batch size |
| `EPOCHS` | `5` | Training epochs |
| `LR` | `1e-4` | Learning rate |
| `FIG_DIR` | `'figures'` | Where plots are saved |
| `OUT_DIR` | `'outputs'` | Where the model and metrics are saved |

## Outputs

**Figures** (`figures/`)

| File | Content |
|------|---------|
| `01_class_balance.png` | Class distribution by split |
| `06_confusion_matrix.png` | Test confusion matrix at the default threshold |
| `07_roc_threshold.png` | Validation ROC with the calibrated test operating point |
| `08_prevalence_ppv.png` | PPV versus disease prevalence |
| `09_gradcam.png` | Grad-CAM for correct predictions (top) and errors (bottom) |

**Artifacts** (`outputs/`)

| File | Content |
|------|---------|
| `week6_model.pt` | State dict plus classes, backbone, image size and grayscale flag |
| `metrics.json` | Environment, config, dataset stats, duplicate audit, test metrics |
| `test_predictions.csv` | Per-image true label, pneumonia probability and prediction |

### Loading the saved model

```python
import torch
from torchvision import models

ckpt = torch.load('outputs/week6_model.pt', map_location='cpu')
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
model.load_state_dict(ckpt['state_dict'])
model.eval()
```

## Results

Fill this table from `outputs/metrics.json` after your run.

| Metric | Default threshold | Calibrated threshold |
|--------|-------------------|----------------------|
| Accuracy | _ | _ |
| Macro-F1 | _ | _ |
| Sensitivity (recall) | _ | _ |
| Specificity | _ | _ |
| PPV (test prevalence) | _ | _ |
| PPV (5% prevalence) | _ | _ |
| AUC (test) | _ | _ |

Results vary with the random seed, number of epochs and whether pretrained weights were used.

## Design Decisions

- **Patient-level split.** Several X-rays can come from the same patient. A random image-level split would leak patients across train and validation and inflate scores.
- **Validation split from train.** The official `val` folder has only 16 images, which is too few to tune a threshold reliably.
- **Threshold chosen on validation, not test.** Picking it on the test set would be test-set tuning. The test set is used only to report the final operating point.
- **PPV at multiple prevalences.** The test set is heavily enriched with pneumonia cases. PPV in a low-prevalence screening setting is much lower, and the plot makes that visible.
- **Duplicate and overlap audit.** Exact or near-duplicate images across train and test can silently inflate results, so they are measured and logged.

## Limitations

- Single dataset from one source. No external validation, so generalisation to other hospitals or scanners is unknown.
- Pediatric images only, so results do not transfer to adults.
- Grad-CAM shows where the model looked, not why it decided, and can highlight non-clinical artifacts such as markers and borders.
- Near-duplicate detection uses a simple 16×16 thumbnail distance, which is an approximation.
- Not clinically validated.

## License

Add your license here (for example MIT). The dataset has its own license and terms on Kaggle.

## Acknowledgements

- Kermany, D. et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018.
- Selvaraju, R. et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- He, K. et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
