# Brain Tumor Detection Using YOLO Variants

A deep learning project comparing YOLOv5, YOLOv8, and YOLOv11 for detecting brain tumors from MRI images.

**Authors:** Ayham Obeidat, Omar Jabour, Layan Nofal  
**Institution:** University of Jordan  
**Date:** May 2025

---

## Overview

This project evaluates three YOLO object detection models for classifying and localizing brain tumors in MRI scans across three tumor types:

- Glioma
- Meningioma
- Pituitary

---

## Repository Structure

```
BrainTumor/
├── YOLOv5/
│   └── YOLOv5.ipynb
├── YOLOv8/
│   └── YOLOv8.ipynb
├── YOLOv11/
│   └── YOLOv11.ipynb
├── docs/
│   └── brain_tumor_detection_report.pdf
└── README.md
```

---

## Dataset

- **Source:** [Brain Tumor Detection Dataset — Kaggle](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection/data)
- **Total images:** 3,064 MRI scans
- **Classes:** Glioma, Meningioma, Pituitary
- **Split:**
  - Train: 2,144 images
  - Validation: 612 images
  - Test: 308 images
- **Format:** YOLO-compatible (images + `.txt` label files)
- **Image size:** Resized from 260×260 to 640×640

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | Auto |
| Device | GPU |
| Models | YOLOv5s, YOLOv8n, YOLOv11n |

---

## Results

| Model | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|-------|-----------|--------|---------|---------------|
| YOLOv5 | 0.924 | 0.880 | 0.919 | 0.693 |
| **YOLOv8** | 0.891 | 0.883 | **0.924** | 0.711 |
| YOLOv11 | 0.917 | 0.856 | 0.922 | 0.715 |

> YOLOv8 achieved the highest mAP@0.5 of 0.924.

### Class-wise Highlights
- **Meningioma & Pituitary:** Consistently high precision and recall (above 0.96) across all models
- **Glioma:** The most challenging class across all models due to ambiguous imaging characteristics

---

## Methodology

1. **Dataset Preparation** — Organized into YOLO format with stratified train/val/test split
2. **Preprocessing** — Resized images to 640×640, verified label files
3. **Training** — Each model trained independently for 50 epochs on GPU
4. **Evaluation** — Precision, Recall, mAP, F1-Confidence Curve, Confusion Matrix
5. **Experiment** — Tested undersampling + augmentation on YOLOv11 to address class imbalance

---

## How to Run

### YOLOv5
```bash
cd YOLOv5
pip install -r requirements.txt
python train.py --data data.yaml --weights yolov5s.pt --epochs 50 --batch 16
```

### YOLOv8
```bash
pip install ultralytics
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 batch=16
```

### YOLOv11
```bash
pip install ultralytics
yolo detect train data=data.yaml model=yolo11n.pt epochs=50 batch=16
```

---

## References

- Shannon, C. E. (1948). A mathematical theory of communication.
- Lapointe, S., Perry, A., & Butowski, N. A. (2018). Primary brain tumours in adults. *Lancet*.
- [Application of MRI image segmentation based on improved YOLO](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1510175/full)
- [A Computer-Aided Diagnosis of Brain Tumors Using Fine-Tuned YOLO](https://koreascience.kr/article/JAKO202009135419641.page)
