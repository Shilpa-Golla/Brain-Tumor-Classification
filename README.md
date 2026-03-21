# Brain Tumor MRI Classification

A deep learning system that classifies brain MRI scans into four categories — **Glioma, Meningioma, No Tumor, and Pituitary** — using PyTorch. Built as part of the Machine Learning and Deep Learning portfolio module.

---

## Live Demo

 **[Try the app on Streamlit Cloud](https://brain-tumor-classification-avxyduf6zcapatddcfjpqr.streamlit.app/)**

Upload any brain MRI scan and get an instant prediction with Grad-CAM heatmap showing exactly where the model looked.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MLP Baseline | 64.50% | 0.6394 | 0.6450 | 0.6296 |
| Custom CNN | 79.31% | 0.8135 | 0.7931 | 0.7877 |
| ResNet-18 | **95.62%** | **0.9591** | **0.9563** | **0.9553** |
| EfficientNet-B0 | 95.56% | 0.9584 | 0.9556 | 0.9546 |

---

## Repository Structure

```
Brain-Tumor-Classification/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── notebooks/
│   ├── train.ipynb                 # Task 1 — Data preparation & augmentation
│   ├── model_development.ipynb     # Task 2 & 3 — Model training & comparison
│   ├── evaluate.ipynb              # Task 3 — Full evaluation on test set
│   └── gradcam.ipynb               # Task 4 — Grad-CAM visualisations
│
└── saved_models/
    ├── MLP_Baseline_best.pth
    ├── Custom_CNN_best.pth
    ├── ResNet18_best.pth
    └── EfficientNet_v2_Phase1_best.pth
```

---

##  Dataset

**Brain Tumor MRI Dataset** — Kaggle (Masoud Nickparvar, 2021)

- 7,023 MRI images across 4 classes
- Pre-split into Training and Testing folders
- Images in JPEG format at varying resolutions

| Class | Label | Train | Test |
|-------|-------|-------|------|
| Glioma | 0 | 1321 | 300 |
| Meningioma | 1 | 827 | 306 |
| No Tumor | 2 | 1595 | 405 |
| Pituitary | 3 | 1457 | 300 |

**Download:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Model Architectures

### 1. MLP Baseline
Flattens the 224×224 image into 150,528 values and passes through fully connected layers. Serves as the lower-bound reference — no spatial awareness.

### 2. Custom CNN
Built from scratch with 4 convolutional blocks (Conv → BN → ReLU → MaxPool), each doubling the filters while halving the spatial resolution. Uses Global Average Pooling to reduce overfitting.

### 3. ResNet-18 — Transfer Learning
Pretrained on ImageNet (1.2M images). Fine-tuned in 2 phases — head warm-up first, then full end-to-end fine-tuning with a lower learning rate.

### 4. EfficientNet-B0 — Transfer Learning
Compound scaling across depth, width, and resolution. Layer-wise learning rates in Phase 2. Label smoothing and CosineAnnealingWarmRestarts for better generalisation.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Shilpa-Golla/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from Kaggle and place it in the following structure:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 4. Run the notebooks in order
```
1. notebooks/train.ipynb
2. notebooks/model_development.ipynb
3. notebooks/evaluate.ipynb
4. notebooks/gradcam.ipynb
```

### 5. Launch the Streamlit app locally
```bash
streamlit run app.py
```

---

## Streamlit App Features

- Upload any brain MRI scan (JPG, PNG)
- Instant prediction with confidence score
- Per-class probability bars
- Grad-CAM heatmap showing model focus region
- Toggle for Test-Time Augmentation (TTA)
- Entropy and prediction margin metrics

---

## Key Techniques

| Technique | Purpose |
|-----------|---------|
| Data Augmentation | Flips, rotations, colour jitter, affine transforms |
| WeightedRandomSampler | Handles class imbalance during training |
| 2-Phase Fine-tuning | Protects pretrained features during transfer learning |
| Layer-wise Learning Rates | Different LR per backbone layer group |
| Label Smoothing | Prevents overconfident wrong predictions |
| Test-Time Augmentation | Reduces prediction variance on borderline cases |
| Grad-CAM | Explains which MRI regions drove each prediction |

---

## Citations

- **Dataset** — Masoud Nickparvar (2021). Brain Tumor MRI Dataset. Kaggle.
- **ResNet** — He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- **EfficientNet** — Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
- **Grad-CAM** — Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
- **Pretrained Weights** — torchvision.models, ImageNet1K_V1
- **Framework** — PyTorch

---

## Disclaimer

This project is built for **academic and research purposes only**. It must not be used for clinical diagnosis or any medical decision-making.

---

**Author:** Shilpa Golla, Divi Teja Dimmiti

