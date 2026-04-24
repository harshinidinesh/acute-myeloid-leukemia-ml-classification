# Acute Myeloid Leukemia (AML) Cancer Cell Classification

**Course:** DS 4420 | **Instructor:** Dr. Eric Gerber | **Date:** April 2026  
**Authors:** Harshini Dinesh & Samyutha Srinivasan

A machine learning pipeline for classifying Acute Myeloid Leukemia (AML) subtypes using two complementary approaches: a Convolutional Neural Network (CNN) applied to single-cell blood smear images and a Gaussian Naive Bayes classifier applied to patient-level clinical metadata.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Models](#models)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [Naive Bayes Classifier](#naive-bayes-classifier)
- [Results](#results)
- [Future Work](#future-work)
- [Repository Structure](#repository-structure)
- [References](#references)

---

## Background

AML is a rapid-onset blood cancer with significant diagnostic complexity. Traditional clinical and statistical approaches struggle to reliably distinguish AML subtypes. This project fills a gap in the literature by:

1. Manually implementing a CNN to analyze single-cell blood smear images
2. Applying a Gaussian Naive Bayes classifier to patient-level clinical metadata
3. Applying both approaches to the **AML-Cytomorphology MLL Helmholtz** dataset — a novel application not found in prior literature

---

## Dataset

**Source:** [AML-Cytomorphology MLL Helmholtz](https://doi.org/10.7937/6PPE-4020) via The Cancer Imaging Archive

- **81,214** single-cell images from **189** peripheral blood smears (2009–2020)
- Collected by the Munich Leukemia Laboratory (MLL)
- Scanned at 40x magnification via oil immersion microscopy in `.TIF` format

**Classes (5 total):**

| Label | Description |
|---|---|
| Control | Healthy stem cell donors |
| APL / PML-RARA | APL with PML::RARA fusion |
| NPM1 | AML with NPM1 mutation |
| CBFB-MYH11 | AML with CBFB::MYH11 fusion (no NPM1 mutation) |
| RUNX1-RUNX1T1 | AML with RUNX1::RUNX1T1 fusion |

---

## Models

### Convolutional Neural Network (CNN)

**Language:** Python

**Preprocessing:**
- Images resized from 144×144 → 96×96 pixels
- Converted to grayscale
- Pixel values normalized to [0, 1]
- Patient-level 75/25 train-test split (prevents data leakage from same-patient images)
- Limited to 15,000 training / 4,000 test images
- Class-weighted loss to handle class imbalance

**Architecture:**
- 3 convolutional layers (32, 64, 128 filters) with ReLU activation
- Batch normalization after each convolution
- Global Average Pooling (reduces overfitting vs. flattening)
- Dropout rate: 0.4
- Softmax output layer (5-class)
- Optimizer: Adam (lr = 1e-4), loss: categorical cross-entropy
- Trained for 10 epochs

---

### Naive Bayes Classifier

**Language:** R

**Preprocessing:**
- 189 patient observations with 13 peripheral blood differential features (e.g., myeloblast %, neutrophil %, monocyte %, leucocyte count, age, sex)
- Stratified 80/20 train-test split on `bag_label`
- Missing values imputed with per-feature training-set medians

**Model:**
- Gaussian Naive Bayes: `P(class | x) ∝ P(class) × ∏ P(xᵢ | class)`
- Likelihoods modeled as Gaussians parameterized by per-class mean and standard deviation
- Categorical `sex` feature handled with Laplace-smoothed class-conditional proportions
- Predictions assigned by highest log-posterior score

**Feature Importance (Mutual Information):**

| Feature | MI Score |
|---|---|
| pb_myeloblast | 0.652 |
| pb_neutrophil_segmented | 0.580 |
| pb_monocyte | 0.429 |
| pb_neutrophil_band | 0.072 |
| pb_metamyelocyte | 0.042 |
| pb_other | 0.056 |

---

## Results

### CNN

| Metric | Value |
|---|---|
| Test Accuracy | 43.9% |
| Baseline (random) | 20.0% |
| Best Validation Accuracy | 47.08% (epoch 7) |

Notable findings:
- Class 4 (RUNX1-RUNX1T1): precision 0.91, recall 0.80 — strong performance
- Class 0 (Control): recall 0.93 but precision 0.25 — model over-predicts this class
- Systematic misclassification between morphologically similar subtypes

### Naive Bayes

| Metric | Value |
|---|---|
| Test Accuracy | 65.0% |
| 10-Fold CV Mean Accuracy | 60.0% (SD = 8.2%) |
| CV Range | 40.0% – 68.4% |

Per-class accuracy:
- Control: **100%**
- PML-RARA: **100%**
- CBFB-MYH11: **62.5%**
- RUNX1-RUNX1T1: **42.9%**
- NPM1: **12.5%** *(majority misclassified as PML-RARA)*

---

## Future Work

- **CNN:** Explore richer architectures (e.g., ResNet with transfer learning); revisit data augmentation (rotation, horizontal shifts); increase training epochs
- **Naive Bayes:** Explore Linear Discriminant Analysis to capture feature covariance; validate on larger patient cohorts
- **Multimodal framework:** Combine both models into a unified probabilistic classifier:
  - CNN output: `P(y | image)`
  - Naive Bayes output: `P(y | clinical)`
  - Combined: `P(y | x_image, x_clinical) ∝ P(y | x_image) × P(y | x_clinical)`

---

## Repository Structure

```
acute-myeloid-leukemia-ml-classification/
├── cnn/
│   └── cnn_model.py          # CNN implementation (Python)
├── naive_bayes/
│   └── naive_bayes.R         # Naive Bayes implementation (R)
├── data/
│   └── README.md             # Instructions for downloading dataset via IBM Aspera
└── README.md
```

> **Note:** Raw image data is not included due to size. Download the AML-Cytomorphology MLL Helmholtz dataset from [The Cancer Imaging Archive](https://doi.org/10.7937/6PPE-4020) using the IBM Aspera server.

---

## References

1. Hehr, M., Sadafi, A., Matek, C., et al. (2023). *AML-Cytomorphology_MLL_Helmholtz* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/6PPE-4020
2. Al-Nusair, J., Lanino, L., Durmaz, A., et al. (2025). Artificial intelligence in myeloid malignancies. *Blood Reviews*, 74, 101340.
3. Yan, L., Yu, H., Xu, X., & Liu, M. (2025). Integrated machine learning-based establishment of a prognostic model in multicenter cohorts for AML. *Frontiers in Oncology*, 15, 1649594.
4. Madduru, S. (2024). Impact of machine learning in AML with prognosis approach for better accuracy. *Int J Intell Syst Appl Eng*, 12(3), 3290–3295.
