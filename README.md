# Rainfall Prediction Using Machine Learning

## 🚀 Live Demo

👉 **[Open Rainfall Prediction System](https://rainfall-prediction-system.streamlit.app/)**

## Overview

This project implements a complete, leak-free machine learning workflow to predict rainfall occurrence from surface weather observations. Using historical daily meteorological records, the pipeline cleans and preprocesses raw data, addresses class imbalance, and trains and tunes multiple classification models. The project emphasizes empirical verification, comparing Logistic Regression, Decision Tree, and Random Forest architectures under rigorous 5-fold cross-validation and a strictly held-out test set. All final metrics are verified directly from model predictions without data leakage or performance fabrication.

## Problem Statement

Precipitation prediction is an essential task in agricultural planning, water resource management, and daily weather forecasting. The goal of this project is to predict binary rainfall occurrence ($1 = \text{Rain}$, $0 = \text{No Rain}$) for a given day using atmospheric surface observations, including barometric pressure, dew point, relative humidity, cloud cover, sunshine duration, wind direction, and wind speed. The problem involves handling class imbalance and multi-feature interactions while ensuring that the model generalizes to unseen test data.

## Dataset

- **Observations:** 366 daily rows (representing one localized calendar year of measurements).
- **Target Variable:** `rainfall` indicating rain occurrence ($1 = \text{Rain}$ [249 samples, 68.03%], $0 = \text{No Rain}$ [117 samples, 31.97%]).
- **Final Features:** 7 continuous meteorological features selected for final modeling.
- **Scope Note:** The dataset represents localized observations from a single geographic monitoring station; it does not capture all global climates, regional microclimates, or multi-year climate cycles.

## Features

The 7 predictor variables used in the final models are:

1. **`pressure`** (hPa): Atmospheric barometric pressure.
2. **`dewpoint`** (°C): Dew point temperature indicating atmospheric moisture saturation.
3. **`humidity`** (%): Relative atmospheric humidity percentage.
4. **`cloud`** (%): Visual cloud cover percentage.
5. **`sunshine`** (hours): Daily sunshine duration in hours.
6. **`winddirection`** (degrees): Azimuth direction of surface wind.
7. **`windspeed`** (km/h): Surface wind speed.

*Note:* Three temperature variables (`maxtemp`, `temparature`, `mintemp`) present in the raw data were dropped prior to model fitting due to extreme collinearity with dew point.

## Machine Learning Workflow

The end-to-end pipeline follows a disciplined, leak-free engineering process:

1. **Data Loading:** Read `data/Rainfall.csv`, strip column whitespace, and validate required columns.
2. **Cleaning:** Drop the non-informative `day` index and collinear temperature predictors.
3. **Target Encoding:** Map `rainfall` values (`yes` $\to 1$, `no` $\to 0$).
4. **Stratified Train/Test Split:** Split the dataset into 80% training (292 rows) and 20% testing (74 rows), stratified by the target to maintain identical class ratios.
5. **Train-Only Median Imputation:** Fit `SimpleImputer(strategy='median')` strictly on `X_train` and transform both `X_train` and `X_test` without calculating test statistics.
6. **Class Imbalance Handling:** Benchmark cost-sensitive weighting (`class_weight='balanced'`) against `SMOTE` oversampling within cross-validation folds.
7. **Model Training:** Fit candidate classification algorithms on the processed training set.
8. **5-Fold Stratified Cross-Validation:** Cross-validate hyperparameter options on training data using F1-score optimization.
9. **Hyperparameter Tuning:** Tune regularisation and tree constraints using `GridSearchCV`.
10. **Final Holdout Evaluation:** Evaluate selected models exactly once on the untouched 74-row test set.
11. **Model Serialization:** Save the selected model, imputer, and feature metadata to `models/rainfall_prediction_model.pkl`.
12. **CLI Inference:** Provide interactive and non-interactive command-line tools for predicting rainfall on new inputs.

## Models

Three classification architectures were evaluated:

- **Logistic Regression:**
  - *Baseline Experiment:* `SMOTE` inside `imblearn.pipeline.Pipeline` + `StandardScaler` + default $C=1.0$, $L2$ penalty.
  - *Tuned Experiment:* `SMOTE` + `StandardScaler` + $C=0.1$, $L1$ penalty (liblinear solver).
- **Decision Tree:**
  - *Baseline Experiment:* Unconstrained tree with `class_weight='balanced'`.
  - *Tuned Experiment:* Pruned tree with `max_depth=3`, `min_samples_leaf=4`, `min_samples_split=2`, `criterion='entropy'`.
- **Random Forest:**
  - *Baseline Experiment:* 50 estimators, unconstrained depth with `class_weight='balanced'`.
  - *Tuned Experiment:* 150 estimators, `max_depth=6`, `min_samples_leaf=2`, `min_samples_split=2`, `max_features='sqrt'`.

## Results

All reported metrics reflect actual calculations on the untouched 74-sample holdout test set (50 Rain, 24 No Rain) and 5-Fold Stratified Cross-Validation on the 292 training samples:

| Model Experiment | Balancing Method | Selected Hyperparameters | 5-Fold CV F1 Mean | CV F1 Std | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | `SMOTE` (ImbPipeline) | `C=1.0, penalty='l2'` (default) | 0.8341 | 0.0340 | **83.78%** | **0.8958** | 0.8600 | **0.8776** | 0.9250 |
| **Logistic Regression (Tuned)** | `SMOTE` (ImbPipeline) | `C=0.1, penalty='l1', liblinear` | 0.8601 | 0.0190 | 79.73% | 0.8723 | 0.8200 | 0.8454 | **0.9267** |
| **Decision Tree (Tuned)** | `class_weight=None` | `max_depth=3, min_leaf=4, criterion='entropy'` | 0.8667 | 0.0220 | 79.73% | 0.8431 | 0.8600 | 0.8515 | 0.8150 |
| **Random Forest (Tuned)** | `class_weight=None` | `n_est=150, max_depth=6, min_leaf=2, sqrt` | **0.8790** | 0.0277 | **81.08%** | 0.8000 | **0.9600** | **0.8727** | 0.8967 |

*Note: Cross-validation scores reflect training fold performance, whereas test metrics reflect generalisation on the holdout test set.*

## Selected Final Model

The **Tuned Random Forest Classifier** (`n_estimators=150, max_depth=6, min_samples_leaf=2, max_features='sqrt'`) was selected as the final project model based on evaluation criteria:
- **Highest Cross-Validation F1:** Achieved the highest 5-fold cross-validation score (**0.8790**) with low fold-to-fold variance ($\pm 0.0277$).
- **Rain Event Sensitivity:** Achieved a test recall of **96.00%**, correctly identifying 48 out of 50 actual rainfall events in the test set.
- **Ensemble Stability:** Bagging reduced variance and avoided the boundary noise vulnerabilities observed with single decision trees and SMOTE interpolation.

## Important Findings

1. **High Recall on Rainfall Events:** The tuned Random Forest achieved 96.00% recall on the held-out test set, minimizing false negatives (unpredicted rain).
2. **Linear Baseline Strength:** The baseline Logistic Regression with SMOTE achieved the highest overall test accuracy (83.78%) and F1-score (0.8776) with strong calibration (ROC-AUC 0.9250).
3. **Multi-Perspective Metrics:** Relying on accuracy alone obscures class imbalance nuances; tracking recall, precision, F1-score, and ROC-AUC together provides a complete assessment of model trade-offs.
4. **Tree Pruning Value:** Unconstrained decision trees suffered from severe overfitting (71.62% test accuracy); pruning depth to 3 improved test accuracy by over 8 percentage points to 79.73%.

## Feature Importance

Gini feature importances extracted from the tuned Random Forest model (`feature_importances_`):

| Feature | Gini Importance | Relative Contribution |
|---|:---:|:---:|
| `cloud` | 0.2807 | 28.1% |
| `sunshine` | 0.2699 | 27.0% |
| `humidity` | 0.1390 | 13.9% |
| `windspeed` | 0.1027 | 10.3% |
| `dewpoint` | 0.0822 | 8.2% |
| `pressure` | 0.0791 | 7.9% |
| `winddirection` | 0.0466 | 4.7% |

*Important:* These values reflect Gini impurity reduction across decision tree splits within this specific dataset; they do not establish direct physical causality.

## Installation

### 1. Create a Virtual Environment
```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment
- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

### Launch the Web Interface
The project includes a modern, responsive web dashboard built with Streamlit. It provides interactive rainfall predictions and probability gauges using the already trained and serialized model (`models/rainfall_prediction_model.pkl`) without retraining.

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

Launch the interface:
```bash
streamlit run app.py
```

The application opens locally in your browser (default `http://localhost:8501`).

### Execute the Full Pipeline
Runs data loading, preprocessing, model training, cross-validation, test evaluation, and figure generation:
```bash
python main.py
```

### Run Interactive Prediction (CLI)
Prompts for weather features in the terminal:
```bash
python predict.py
```

### Run Command-Line Prediction (CLI)
Supply weather measurements directly via command-line flags:
```bash
python predict.py --humidity 85.0 --cloud 80.0 --sunshine 2.0 --wind-speed 22.0 --dew-point 21.5 --pressure 1010.0 --wind-direction 75.0
```

### View Inference Options
```bash
python predict.py --help
```

## Project Structure

```
rainfall_prediction_system/
│
├── data/
│   └── Rainfall.csv               # Canonical dataset (366 daily rows)
│
├── notebooks/
│   └── rainfall_prediction.ipynb   # Documented research analysis notebook (84 cells)
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── data_loader.py             # Data loading and column validation
│   ├── preprocessing.py           # Feature dropping, encoding, stratified split, imputation
│   ├── models.py                  # Model architectures and parameter configurations
│   ├── evaluation.py              # CV scoring, metrics calculation, model bundle I/O
│   ├── visualization.py           # Confusion matrices, ROC curves, EDA plots
│   └── predict.py                 # Reusable inference functions and validation
│
├── models/
│   └── rainfall_prediction_model.pkl # Serialized final selected model bundle
│
├── outputs/
│   ├── figures/                   # High-resolution PNG evaluation figures
│   │   ├── target_class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── rainfall_relationships.png
│   │   ├── logistic_regression_confusion_matrix.png
│   │   ├── decision_tree_confusion_matrix.png
│   │   ├── random_forest_confusion_matrix.png
│   │   ├── roc_curves.png
│   │   ├── model_performance_comparison.png
│   │   └── random_forest_feature_importance.png
│   ├── model_comparison.csv       # Machine-readable metric results
│   ├── feature_importance.csv     # Ranked feature importance scores
│   └── final_results.md           # Comprehensive project summary
│
├── app.py                         # Streamlit web dashboard interface
├── main.py                        # Executable end-to-end pipeline runner
├── predict.py                     # CLI and interactive prediction interface
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

## Limitations

1. **Small Sample Size:** 366 observations represent a single calendar year of localized observations.
2. **Localized Weather Patterns:** The dataset reflects conditions at a single monitoring site and cannot be generalized to other geographic regions without retraining.
3. **Holdout Scope:** Test-set performance demonstrates generalization on this dataset and should not be interpreted as universal real-world accuracy across all seasons or years.
4. **Binary Resolution:** The target is binary ($>0$ mm rainfall) and does not quantify precipitation volume or storm duration.
5. **Associative Feature Importance:** Gini feature importance measures tree split purity and does not prove causal relationships.

## Future Improvements

- Incorporate multi-year records to capture seasonal cycles and long-term climate variability.
- Integrate additional meteorological variables such as upper-air pressure levels and radar imagery.
- Evaluate rolling time-series cross-validation to assess sequential temporal forecasting.
- Test external validation datasets from neighboring meteorological stations.
- Explore probability calibration (e.g., Platt scaling or isotonic regression) to refine rain probability estimates.
- Package the inference pipeline as a lightweight REST API for integration into external services.
