"""
Rainfall Prediction System - Main Entry Point.
Coordinates end-to-end execution:
1. Data loading and validation
2. Leakage-free preprocessing (stratified split + X_train median imputation)
3. EDA figures generation
4. Model training and evaluation (Stratified 5-Fold CV + untouched holdout test)
5. Metric comparisons and visualization export
6. Final model serialization and reload verification
"""
import os
import shutil
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.models import get_all_models
from src.evaluation import (
    evaluate_models_comparison,
    save_model_bundle,
    load_model_bundle,
    predict_raw
)
from src.visualization import (
    plot_eda_figures,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
    plot_feature_importance
)

def run_pipeline():
    print("=" * 75)
    print("      RAINFALL PREDICTION SYSTEM — PIPELINE EXECUTION")
    print("=" * 75)
    
    # 1. Load Data
    data_path = "data/Rainfall.csv"
    print(f"\n[1/6] Loading data from '{data_path}'...")
    raw_df = load_data(data_path)
    print(f"      Loaded {raw_df.shape[0]} rows and {raw_df.shape[1]} columns.")
    
    # 2. EDA Figures
    print("\n[2/6] Generating EDA visualizations...")
    plot_eda_figures(raw_df, save_dir="outputs/figures")
    print("      Saved: target_class_distribution.png, correlation_heatmap.png, rainfall_relationships.png")
    
    # 3. Preprocess Data (Leakage-free stratified split + train-only imputation)
    print("\n[3/6] Preprocessing data (stratified split + median imputation)...")
    data_bundle = prepare_data(raw_df, target_col='rainfall', test_size=0.2, random_state=42)
    X_train = data_bundle['X_train']
    X_test = data_bundle['X_test']
    y_train = data_bundle['y_train']
    y_test = data_bundle['y_test']
    imputer = data_bundle['imputer']
    feature_names = data_bundle['feature_names']
    
    print(f"      Training set: {X_train.shape[0]} samples (Class 1: {sum(y_train==1)}, Class 0: {sum(y_train==0)})")
    print(f"      Test set:     {X_test.shape[0]} samples (Class 1: {sum(y_test==1)}, Class 0: {sum(y_test==0)})")
    print(f"      Predictor features: {feature_names}")
    
    # 4. Model Training & Comparison
    print("\n[4/6] Evaluating candidate models (Stratified 5-Fold CV + holdout test)...")
    models_dict = get_all_models(random_state=42)
    comparison_df, trained_models, predictions = evaluate_models_comparison(
        models_dict, X_train, y_train, X_test, y_test, cv_folds=5, random_state=42
    )
    
    # Save comparison CSV
    os.makedirs("outputs", exist_ok=True)
    comparison_csv_path = "outputs/model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"      Saved metrics table to '{comparison_csv_path}'.")
    print("\n" + comparison_df.to_string(index=False) + "\n")
    
    # 5. Visualizations
    print("[5/6] Generating evaluation figures...")
    # Confusion matrices
    plot_confusion_matrix(
        y_test, predictions['Logistic Regression (Tuned)']['y_pred'],
        title="Logistic Regression (Tuned) Confusion Matrix",
        save_path="outputs/figures/logistic_regression_confusion_matrix.png"
    )
    plot_confusion_matrix(
        y_test, predictions['Decision Tree (Tuned)']['y_pred'],
        title="Decision Tree (Tuned) Confusion Matrix",
        save_path="outputs/figures/decision_tree_confusion_matrix.png"
    )
    plot_confusion_matrix(
        y_test, predictions['Random Forest (Tuned)']['y_pred'],
        title="Random Forest (Tuned) Confusion Matrix",
        save_path="outputs/figures/random_forest_confusion_matrix.png"
    )
    
    # Combined ROC Curves
    roc_items = [
        {'name': 'Logistic Regression Tuned',    'y_true': y_test, 'y_prob': predictions['Logistic Regression (Tuned)']['y_prob'],    'color': '#e67e22', 'linewidth': 2},
        {'name': 'Logistic Regression Baseline', 'y_true': y_test, 'y_prob': predictions['Logistic Regression (Baseline)']['y_prob'], 'color': '#f39c12', 'linestyle': ':', 'linewidth': 2},
        {'name': 'Decision Tree Tuned',          'y_true': y_test, 'y_prob': predictions['Decision Tree (Tuned)']['y_prob'],          'color': '#9b59b6', 'linewidth': 2},
        {'name': 'Random Forest Tuned',          'y_true': y_test, 'y_prob': predictions['Random Forest (Tuned)']['y_prob'],          'color': '#27ae60', 'linewidth': 2.5},
    ]
    plot_roc_curves(roc_items, save_path="outputs/figures/roc_curves.png")
    
    # Performance comparison bar chart
    plot_model_comparison(comparison_df, save_path="outputs/figures/model_performance_comparison.png")
    
    # Random Forest feature importance
    best_rf = trained_models['Random Forest (Tuned)']
    plot_feature_importance(
        best_rf, feature_names,
        save_path="outputs/figures/random_forest_feature_importance.png",
        csv_path="outputs/feature_importance.csv"
    )
    print("      All figures and CSV exports saved to 'outputs/'.")
    
    # 6. Model Serialization & Verification
    print("\n[6/6] Saving selected production model bundle...")
    rf_metrics = comparison_df[comparison_df['Model'] == 'Random Forest (Tuned)'].iloc[0].to_dict()
    model_save_path = "models/rainfall_prediction_model.pkl"
    save_model_bundle(
        model=best_rf,
        imputer=imputer,
        feature_names=feature_names,
        metrics=rf_metrics,
        filepath=model_save_path
    )
    print(f"      Model bundle serialized to '{model_save_path}'.")
    
    # Reload verification
    print("      Verifying model bundle reload and prediction compatibility...")
    loaded_bundle = load_model_bundle(model_save_path)
    sample_raw = data_bundle['X_test_raw'].iloc[:5]
    preds, probs = predict_raw(loaded_bundle, sample_raw)
    
    sample_check = pd.DataFrame({
        'Actual': y_test.iloc[:5].values,
        'Predicted': preds,
        'Probability': probs.round(4)
    })
    print("\nSample Predictions on Held-Out Test Data:")
    print(sample_check.to_string(index=False))
    
    print("\n" + "=" * 75)
    print("      PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_pipeline()
