# DenseNet-121 Based Classification of Myocardial Perfusion Images

A deep learning project using a fine-tuned DenseNet-121 model to classify Myocardial Perfusion Imaging (MPI) scans into three diagnostic categories for coronary artery disease detection.

**Authors:** Omar Abuhadhoud, Alaa Odat, Anas Joudeh, Ayham Moneer  
**Institution:** University of Jordan  
**Date:** May 2025

---

## Overview

This project proposes a supervised learning approach using a fine-tuned DenseNet-121 model trained on RGB heart images derived from three MPI modalities: Attenuation-Corrected Quantitative Perfusion SPECT (AC QPS), Stress Quantitative Gated SPECT (Stress QGS), and Rest Quantitative Gated SPECT (Rest QGS). The model classifies each patient into one of three cardiac conditions: Normal, Ischemia, or Infarction, achieving an overall test accuracy of 90%.

---

## Repository Structure

```
Medical/
├── Medical.ipynb
├── MPI Images Selected/
│   ├── Normal/
│   ├── Ischemia/
│   └── Infarction/
├── docs/
│   └── DenseNet_121_Based_Classification_of_Myocardial_Perfusion_Images.pdf
└── README.md
```

---

## Dataset

- **Patients:** 97 labeled by expert clinicians
- **Classes:** Normal, Ischemia, Infarction
- **Imaging modalities:** AC QPS, Stress QGS, Rest QGS
- **Total labeled slices:** 620
- **Split (per-patient basis):**

| Split | Ratio |
|-------|-------|
| Train | 60% |
| Validation | 15% |
| Test | 25% |

> The dataset was split on a per-patient basis to prevent data leakage across subsets.

---

## Preprocessing

- Embedded textual annotations were manually cropped from images to avoid bias
- All images resized to **224 × 224** pixels
- Pixel values normalized to the range **[0, 1]**
- Images from the `display`, `nac`, and `segments` modalities were excluded during the split

---

## Data Augmentation

To address class imbalance (underrepresentation of Ischemia and Infarction), augmentation was applied selectively to minority classes to match the Normal class count.

| Transformation | Parameter |
|----------------|-----------|
| Rescaling | 1/255 (normalization to [0, 1]) |
| Rotation | ±25° |
| Zoom | 0.75 to 1.25 |
| Width Shift | Up to 20% |
| Height Shift | Up to 20% |
| Horizontal Flip | Enabled |
| Brightness Adjustment | 0.7 to 1.3 |
| Shear | Up to 10 degrees |
| Fill Mode | Nearest |

---

## Model Architecture

The model is built on **DenseNet-121** pretrained on ImageNet, used as a fixed feature extractor with the last 10 layers fine-tuned. The custom head consists of:

- Global Average Pooling
- Fully connected layer (256 units, ReLU activation, L2 regularization)
- Dropout (rate = 0.5)
- Softmax output layer (3 classes)

---

## Training Configuration

| Component | Setting |
|-----------|---------|
| Optimizer | Adam (lr = 1e-4) |
| Loss | Custom Ischemia-Aware Categorical Cross-Entropy (Ischemia weight = 3.0) |
| Epochs | Max 50 (early stopping, patience = 4) |
| Batch Size | 16 |
| Regularization | L2 (0.001), Dropout (0.5) |
| LR Scheduler | ReduceLROnPlateau (factor = 0.5, min_lr = 1e-6) |
| Frameworks | TensorFlow, Keras, scikit-learn, Matplotlib |

### Custom Loss Function

A custom ischemia-aware loss function was implemented to penalize misclassification of ischemic cases as normal, given the clinical risk of such errors:

```python
def custom_ischemia_loss(weight=2.0):
    def loss(y_true, y_pred):
        base_loss = categorical_crossentropy(y_true, y_pred)
        true_ischemia = (argmax(y_true) == 1)  # ischemia index
        pred_normal   = (argmax(y_pred) == 0)  # normal index
        penalty = weight * (true_ischemia * pred_normal)
        return base_loss + penalty
    return loss
```

---

## Results

### Classification Report (Test Set — 96 samples)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Infarction | 0.86 | 1.00 | 0.92 | 6 |
| Ischemia | 0.89 | 0.67 | 0.76 | 24 |
| Normal | 0.90 | 0.97 | 0.93 | 66 |
| **Overall Accuracy** | | | **0.90** | **96** |
| Macro Avg | 0.88 | 0.88 | 0.87 | 96 |
| Weighted Avg | 0.90 | 0.90 | 0.89 | 96 |

### Confusion Matrix

| | Predicted Infarction | Predicted Ischemia | Predicted Normal |
|---|---|---|---|
| True Infarction | 6 | 0 | 0 |
| True Ischemia | 1 | 16 | 7 |
| True Normal | 0 | 2 | 64 |

### Key Observations

- Infarction and Normal classes achieved strong precision and recall across all evaluations
- Ischemia classification was the most challenging due to class imbalance and the subtle visual similarity of ischemic patterns to normal scans
- Mild overfitting observed: training accuracy reached ~99% while validation accuracy plateaued at ~85–89%

---

## How to Run

1. Install dependencies:
```bash
pip install tensorflow keras scikit-learn matplotlib seaborn pillow numpy
```

2. Organize your dataset under `MPI Images Selected/` with one subfolder per class (Normal, Ischemia, Infarction).

3. Open and run the notebook:
```bash
jupyter notebook Medical.ipynb
```

The notebook will handle dataset splitting, augmentation, model training, and evaluation automatically.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | Model building and training |
| DenseNet-121 (ImageNet) | Pretrained backbone |
| scikit-learn | Metrics and evaluation |
| Matplotlib / Seaborn | Visualization |
| Pillow | Image cropping and preprocessing |
| NumPy | Numerical operations |

---

## References

- Kaplan Berkaya S. et al. Classification models for SPECT myocardial perfusion imaging. *Comput Biol Med*, 2020.
- Huang G. et al. Densely Connected Convolutional Networks. *CVPR*, 2017.
- Slart R.H.J.A. et al. Position paper on AI applications in multimodality cardiovascular imaging. *Eur J Nucl Med Mol Imaging*, 2021.
- Papandrianos N. & Papageorgiou E. Automatic diagnosis of coronary artery disease in SPECT MPI employing deep learning. *Applied Sciences*, 2021.
