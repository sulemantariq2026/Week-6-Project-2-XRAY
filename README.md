# 🩻 Week 6 – Project 2: X-Ray Image Classification

A deep learning project that analyzes X-ray images and classifies them using a Convolutional Neural Network (CNN). The full workflow, from data loading to evaluation, is contained in a single Jupyter Notebook.

> **Note:** This project is for educational purposes only and is **not** a medical diagnostic tool.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model](#model)
- [Results](#results)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

Chest X-rays are one of the most common medical imaging tools. This project explores how deep learning can be used to automatically classify X-ray images (for example, **Normal** vs. **Pneumonia**).

**Goals:**
- Load and preprocess an X-ray image dataset
- Explore the data and check class balance
- Build and train a CNN classifier
- Evaluate performance with appropriate metrics

## Dataset

- **Source:** `<add dataset name and link, e.g. Chest X-Ray Images (Pneumonia) on Kaggle>`
- **Classes:** `<e.g. NORMAL, PNEUMONIA>`
- **Split:** `<e.g. train / validation / test>`
- **Image size used:** `<e.g. 150 x 150>`

> The dataset is not included in this repository because of its size. Download it from the link above and update the path in the notebook.

## Project Workflow

1. **Import libraries**
2. **Load the data** from the train / validation / test folders
3. **Exploratory data analysis** – sample images and class distribution
4. **Preprocessing** – resizing, normalization, and data augmentation
5. **Build the model** – CNN architecture
6. **Train the model** – with validation monitoring
7. **Evaluate** – accuracy, loss curves, confusion matrix, classification report
8. **Predict** on unseen images

## Model

| Setting | Value |
|---|---|
| Architecture | `<e.g. Custom CNN / Transfer Learning (VGG16, ResNet50…)>` |
| Optimizer | `<e.g. Adam>` |
| Loss | `<e.g. Binary Crossentropy>` |
| Epochs | `<e.g. 10>` |
| Batch size | `<e.g. 32>` |

## Results

| Metric | Score |
|---|---|
| Test Accuracy | `<xx%>` |
| Precision | `<xx%>` |
| Recall | `<xx%>` |
| F1-Score | `<xx%>` |

You can add training curves and the confusion matrix here:

```markdown
![Training Curves](images/training_curves.png)
![Confusion Matrix](images/confusion_matrix.png)
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook, JupyterLab, or Google Colab / Kaggle

### Installation

```bash
# Clone the repository
git clone https://github.com/sulemantariq2026/Week-6-Project-2-XRAY.git
cd Week-6-Project-2-XRAY

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python jupyter
```

### Run the notebook

```bash
jupyter notebook week-6-project-2.ipynb
```

Then update the dataset path in the notebook and run all cells.

## Repository Structure

```
Week-6-Project-2-XRAY/
├── README.md
└── week-6-project-2.ipynb    # Main notebook (data, model, evaluation)
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- scikit-learn
- OpenCV
- Jupyter Notebook

## Future Improvements

- Try transfer learning with pretrained models
- Handle class imbalance (class weights, oversampling)
- Add Grad-CAM visualizations to show what the model focuses on
- Deploy the model as a simple web app (Streamlit / Flask)

## Author

**Suleman Tariq**
GitHub: [@sulemantariq2026](https://github.com/sulemantariq2026)

---

⭐ If you found this project useful, consider giving it a star!
