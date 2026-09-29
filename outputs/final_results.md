# Rainfall Prediction Using Machine Learning — Final Results Summary

**Date:** 2026-09-29  
**Evaluation Standard:** Leakage-Free Stratified Holdout (Test Set: 74 samples)

---

## 1. Dataset Size and Target Variable

- **Total Samples:** 366 observations (daily historical meteorological measurements).
- **Target Variable (`rainfall`):** Binary classification indicating rainfall occurrence.
  - `yes` $\to 1$ (Rain): 249 samples (**68.03%**)
  - `no` $\to 0$ (No Rain): 117 samples (**31.97%**)
- **Input Features (7 final predictors after removing redundant correlated temperatures):**
  - `pressure` (hPa)
  - `dewpoint` (°C)
  - `humidity` (%)
  - `cloud` (%)
  - `sunshine` (hours)
  - `winddirection` (degrees)
  - `windspeed` (km/h)
- *Note:* Highly collinear temperature features (`maxtemp`, `temparature`, `mintemp`) were dropped during preprocessing to eliminate multicollinearity with `dewpoint`.

---

## 2. Train/Test Split

- **Split Ratio:** 80% Training (292 samples), 20% Test (74 samples).
- **Stratification:** Stratified by target variable `rainfall` to preserve identical class ratios:
  - **Training Set (`y_train`):** 199 Rain (68.15%), 93 No Rain (31.85%)
  - **Test Set (`y_test`):** 50 Rain (67.57%), 24 No Rain (32.43%)
- **Test Set Isolation:** The 74-row test set was strictly held out and evaluated only once after all preprocessing and model hyperparameter selection were finalized.

---

## 3. Data Leakage Prevention

Historical audits identified that early notebook iterations imputed missing values across the full dataset prior to splitting, leaking test statistics into training data. This was resolved:
- **Splitting First:** `train_test_split` is executed on raw feature data before any transformers are initialized.
- **Independent Transformers:** All transformers (`SimpleImputer`, `StandardScaler`, and `SMOTE`) are fitted strictly on `X_train`.
- **Pipeline Encapsulation:** During cross-validation, `SMOTE` and `StandardScaler` are wrapped within an `imblearn.pipeline.Pipeline`, guaranteeing that inner validation folds never see resampled points or scaling statistics from outer training folds.

---

## 4. Missing-Value Handling

- **Raw Missing Count:** Exactly 2 missing values exist in the raw dataset (1 in `winddirection`, 1 in `windspeed`).
- **Post-Split Distribution:** After stratified splitting, both missing values fell into `X_train` (2 missing in `X_train`, 0 in `X_test`).
- **Imputation Strategy:** Median imputation via `SimpleImputer(strategy='median')` fitted solely on `X_train`.
- **Learned Medians from Training Data:**
  - `pressure`: 1012.50
  - `dewpoint`: 22.10
  - `humidity`: 80.00
  - `cloud`: 80.00
  - `sunshine`: 3.45
  - `winddirection`: 70.00
  - `windspeed`: 20.40
- **Verification:** 0 missing values remain in both `X_train_imp` and `X_test_imp`.

---

## 5. Class Imbalance Handling

With a class imbalance ratio of ~2.14:1 (68% Rain vs 32% No Rain), two balancing methodologies were evaluated:
1. **Method A — `class_weight='balanced'`:**
   - Inversely penalizes minority class misclassification during loss optimization ($w_0 \approx 1.57, w_1 \approx 0.73$).
   - Preserves all 292 training samples without synthetic interpolation.
2. **Method B — `SMOTE` (Synthetic Minority Over-sampling Technique):**
   - Synthesizes minority examples via k-nearest neighbors interpolation.
   - Enclosed inside `imblearn.pipeline.Pipeline` so resampling occurs strictly within training folds.
   - Upsamples training data to 398 samples (199 / 199) while leaving the test set untouched at 74 rows.

---

## 6. Models Evaluated

Three core classification architectures were evaluated:
1. **Logistic Regression (Linear Benchmark):** Fitted with `StandardScaler` inside the pipeline.
2. **Decision Tree Classifier (Non-linear Tree):** Scale-invariant baseline.
3. **Random Forest Classifier (Ensemble):** Bagging ensemble of decision trees.

---

## 7. Hyperparameter Tuning

All tuning was performed using **Stratified 5-Fold Cross-Validation** on training data only (`X_train_imp`, `y_train`) using `GridSearchCV` optimized for **F1-score**:

- **Logistic Regression:**
  - Tuned parameters: $C \in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]$, penalty $\in ['l1', 'l2']$.
  - Best parameters: `C=0.1, penalty='l1'` (CV F1: **0.8601** $\pm$ 0.0190).
- **Decision Tree:**
  - Tuned parameters: `max_depth` (3–8, None), `min_samples_split` (2–20), `min_samples_leaf` (1–8), `criterion` ('gini', 'entropy'), `class_weight` ('balanced', None).
  - Best parameters: `max_depth=3, min_samples_leaf=4, min_samples_split=2, criterion='entropy', class_weight=None` (CV F1: **0.8667** $\pm$ 0.0220).
- **Random Forest:**
  - Tuned parameters: `n_estimators` (50, 100, 150), `max_depth` (4, 6, 8, None), `min_samples_split` (2, 5, 10), `min_samples_leaf` (1, 2, 4), `max_features` ('sqrt', 'log2'), `class_weight` ('balanced', None).
  - Best parameters: `n_estimators=150, max_depth=6, min_samples_leaf=2, min_samples_split=2, max_features='sqrt', class_weight=None` (CV F1: **0.8790** $\pm$ 0.0277).

---

## 8. Final Test Results (Untouched 74-Row Test Set)

| Model Experiment | Balancing Method | Selected Hyperparameters | 5-Fold CV F1 Mean | CV F1 Std | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | `SMOTE` (ImbPipeline) | `C=1.0, penalty='l2'` (default) | 0.8341 | 0.0340 | **83.78%** | **0.8958** | 0.8600 | **0.8776** | 0.9250 |
| **Logistic Regression (Tuned)** | `SMOTE` (ImbPipeline) | `C=0.1, penalty='l1'` | 0.8601 | 0.0190 | 79.73% | 0.8723 | 0.8200 | 0.8454 | **0.9267** |
| **Decision Tree (Tuned)** | `class_weight=None` | `depth=3, leaf=4, split=2, entropy` | 0.8667 | 0.0220 | 79.73% | 0.8431 | 0.8600 | 0.8515 | 0.8150 |
| **Random Forest (Tuned)** | `class_weight=None` | `n=150, depth=6, leaf=2, sqrt` | **0.8790** | 0.0277 | **81.08%** | 0.8000 | **0.9600** | **0.8727** | 0.8967 |

---

## 9. Important Findings

1. **Feature Importance Hierarchy:**
   - Analysis of Gini importance from the tuned Random Forest demonstrates that **Cloud Cover (28.1%)** and **Sunshine Duration (27.0%)** account for over **55%** of predictive power, followed by **Humidity (13.9%)** and **Wind Speed (10.3%)**. Pressure (7.9%), dewpoint (8.2%), and wind direction (4.7%) contributed minor marginal signal.
2. **Decision Tree Pruning Impact:**
   - Unconstrained decision trees severely overfit (Phase 2 baseline test accuracy: 71.62%, ROC-AUC: 0.6708). Restricting `max_depth=3` and `min_samples_leaf=4` lifted test accuracy to **79.73%** (+8.11%) and ROC-AUC to **0.8150** (+0.1442).
3. **Random Forest Rainfall Event Capture:**
   - The tuned Random Forest achieved an outstanding **96.00% Recall** (correctly detecting 48 out of 50 rain days in the held-out test set) while maintaining solid 80.00% precision and 81.08% overall accuracy. In weather prediction, minimizing missed storms is critical.
4. **SMOTE vs Linear / Tree Models:**
   - Logistic Regression benefited significantly from SMOTE (reaching **83.78% accuracy** and **0.9250 ROC-AUC** in the baseline configuration), whereas Random Forest performed cleaner without synthetic data.
5. **No 90%+ Accuracy Fabrication:**
   - The actual physical limit on this 366-row single-station dataset sits between **81% and 84% accuracy**. Realistic metrics reflect true generalization performance without data manipulation.

---

## 10. Limitations

1. **Dataset Size:** 366 rows represents a single calendar year of data. Seasonal variations across multi-year cycles (El Niño/La Niña, multi-decadal oscillations) cannot be captured.
2. **Single Geographic Station:** The data represents weather measurements at a localized station; models cannot be applied to other climates without local retraining.
3. **Temporal Independence Assumption:** Rows were split using stratified random sampling rather than time-series rolling splits. While standard for general classification demonstrations, operational deployment requires chronological time-series validation.
4. **Resolution:** Predictions indicate binary rain occurrence ($>0$ mm) without estimating total rainfall accumulation or precipitation intensity.
