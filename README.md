# Predicting Metabolic Cost During Human Walking

## 1. Project Overview

This Machine Learning mini-project investigates the prediction of metabolic cost during human walking using biomechanical and physiological measurements.

Metabolic cost is the energy expenditure associated with walking. Measuring it directly with indirect calorimetry is time-consuming and noisy, so this project asks whether metabolic cost can be predicted from easier-to-obtain walking measurements.

The project is based on the research problem presented in **"Predicting Metabolic Cost During Human-in-the-Loop Optimization"** by Erez Krimsky and Eley Ng (Stanford CS229). A publicly available biomechanics and energetics dataset is used to implement the related prediction task.

---

## 2. Dataset

**A biomechanics and energetics dataset of neurotypical adults walking with and without kinematic constraints**

Dataset link: <https://doi.org/10.26188/c.6887854.v1>

The dataset contains walking measurements from neurotypical adults under different walking conditions:

- Electromyography (EMG)
- Ground reaction forces (GRF)
- Joint-angle measurements
- Metabolic measurements from indirect calorimetry

The project uses the **segmented MATLAB files** provided with the dataset, for subjects 1, 2, 3, 4 and 9.

---

## 3. Problem Statement

Metabolic-cost prediction is treated as a **supervised regression** problem.

- **Input:** biomechanical and physiological measurements from a walking test
- **Target:** the measured metabolic cost

The goal is to learn the relationship between the walking features and metabolic cost, and to evaluate how accurately different models predict it.

---

## 4. Features

Each walking test is described by **24 input features**:

| Group | Features | Count |
|-------|----------|-------|
| EMG | Mean EMG of 16 muscles | 16 |
| Ground reaction force | Peak vertical GRF (left, right) | 2 |
| Joint range of motion | Hip, knee, ankle ROM (left, right) | 6 |
| **Total** | | **24** |

---

## 5. Target

The target is **metabolic cost in W/kg**, computed as the mean metabolic cost over the final 20% of each walking test.

Each walking test is one row in the generated feature dataset (30 tests per subject).

---

## 6. Repository Structure

```text
ml-metabolic-cost/
├── .gitignore
├── README.md
├── requirements.txt
│
├── build_features.py      # extract features and target from the .mat files
├── plot_target.py         # data-exploration plots of the target
├── lasso_baseline.py      # LASSO regression baseline
├── neural_net.py          # one-hidden-layer neural network
├── ablation.py            # feature-group ablation
├── check_sub9.py          # Subject 9 checks: offset/calibration and EMG channels
├── snr_compare.py         # EMG signal-to-noise comparison across subjects
├── demo.py                # live demo: predict one held-out subject
│
├── features/
│   ├── all_subjects.csv
│   ├── sub1_features.csv
│   ├── sub2_features.csv
│   ├── sub3_features.csv
│   ├── sub4_features.csv
│   └── sub9_features.csv
│
└── results/               # saved plots
```

| File | Purpose |
|------|---------|
| `build_features.py` | Extracts the features and target from the segmented dataset |
| `plot_target.py` | Plots the target per subject and for tests T1-T15 vs T16-T30 |
| `lasso_baseline.py` | LASSO regression baseline |
| `neural_net.py` | One-hidden-layer neural-network regression model |
| `ablation.py` | Compares feature groups (EMG only, GRF + joint angles, all features) |
| `check_sub9.py` | Checks whether Subject 9's error is a constant offset (with calibration from the first few tests) and whether single EMG channels explain his weak EMG-only result |
| `snr_compare.py` | Reads the EMG signal-to-noise (SNR) values stored in each subject's Energetics file and compares subjects |
| `demo.py` | Trains on four subjects and predicts the held-out subject |

---

## 7. Setup

### 1. Clone the repository

```bash
git clone https://github.com/hmAdil/ml-metabolic-cost.git
cd ml-metabolic-cost
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 8. Getting the Dataset

Download the segmented files from the dataset page: <https://doi.org/10.26188/c.6887854.v1>

For each subject N, the required files are:

```text
SubN_segEnergetics.mat
SubN_segMechanics.mat
```

Place them in a folder named `Sub_N`:

```text
Sub_1/
├── Sub1_segEnergetics.mat
└── Sub1_segMechanics.mat
```

> **Note:** The raw dataset files are large (up to a few GB per subject) and must **not** be committed to GitHub. They are excluded through `.gitignore`.

The processed CSVs in `features/` are included in the repository, so `lasso_baseline.py`, `neural_net.py`, `ablation.py`, `check_sub9.py` and `demo.py` run without the `.mat` files. Only `build_features.py` and `snr_compare.py` need them.

---

## 9. Usage

Run all commands from the repository root.

### Generate features

```bash
python build_features.py
```

This writes one CSV per subject to `features/` and the combined dataset `features/all_subjects.csv`.

### Explore the target

```bash
python plot_target.py
```

Saves boxplots of metabolic cost to `results/`.

### Train and evaluate the models

```bash
python lasso_baseline.py
python neural_net.py
python ablation.py
```

### Check Subject 9

```bash
python check_sub9.py
python snr_compare.py
```

`check_sub9.py` prints, for each held-out subject, the mean residual (actual minus predicted), the share of tests under-predicted, and how calibrating from the first few tests changes the error. It also tests dropping one EMG muscle at a time for Subject 9. `snr_compare.py` prints the median EMG signal-to-noise value per muscle and per subject.

### Run the demo

```bash
python demo.py --subject 2
```

Trains LASSO on the other four subjects, prints actual vs predicted metabolic cost for each test of the chosen subject, and saves a plot to `results/demo_sub2.png`.

---

## 10. Machine Learning Approach

Two regression models are compared:

1. LASSO regression (baseline)
2. One-hidden-layer neural network

Both are evaluated with cross-validation using **Mean Squared Error (MSE)**, which is the average squared difference between actual and predicted metabolic cost (smaller is better). Cross-validation reduces dependence on a single train-test split.

Models are evaluated in two ways: per subject, and pooled with leave-one-subject-out (train on four subjects, test on the fifth). An ablation analysis investigates the contribution of the different feature groups. Subject 9 has the largest leave-one-subject-out error, so `check_sub9.py` and `snr_compare.py` test two possible explanations (a constant offset, and poor EMG signal quality).

---

## 11. Relationship to the Original Paper

The original paper studied metabolic-cost prediction using data collected during human-in-the-loop optimization with an assistive robotic device.

This project uses a public dataset instead, so the data and experimental setup are **not identical** to the paper's. The focus here is the machine-learning problem of predicting metabolic cost from walking measurements.

---

## 12. Important Notes

- Feature definitions and the target calculation must stay consistent across all scripts.
- Raw and large dataset files must not be committed to GitHub.
- The CSV files in `features/` are the processed features used for all experiments.
- Regression is used because the target is a continuous value.