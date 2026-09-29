"""
Evaluation Module for Rainfall Prediction System.
Calculates performance metrics, executes cross-validation, produces comparison tables,
and manages model persistence and test predictions.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Computes accuracy, precision, recall, f1, and roc_auc.
    
    Returns:
        dict: Metric dictionary.
    """
    metrics = {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0))
    }
    if y_prob is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics['roc_auc'] = None
    return metrics

def evaluate_cv(model, X_train, y_train, cv_folds=5, scoring='f1', random_state=42):
    """
    Executes Stratified K-Fold Cross-Validation on training data only.
    
    Returns:
        tuple: (mean_score, std_score, scores_array)
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    return float(scores.mean()), float(scores.std()), scores

def evaluate_models_comparison(models_dict, X_train, y_train, X_test, y_test, cv_folds=5, random_state=42):
    """
    Trains each candidate model on X_train, evaluates CV on training folds,
    evaluates holdout metrics on X_test, and compiles comparison table.
    
    Returns:
        tuple: (comparison_df, trained_models_dict, predictions_dict)
    """
    records = []
    trained_models = {}
    predictions = {}
    
    for name, config in models_dict.items():
        clf = config['model']
        
        # 1. Evaluate CV on training data only
        cv_mean, cv_std, _ = evaluate_cv(clf, X_train, y_train, cv_folds=cv_folds, random_state=random_state)
        
        # 2. Fit on full training data
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        
        # 3. Predict on held-out test set
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_prob = clf.decision_function(X_test)
        else:
            y_prob = None
            
        predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}
        
        # 4. Metrics
        m = calculate_metrics(y_test, y_pred, y_prob)
        
        records.append({
            'Model': name,
            'Balancing Method': config.get('balancing', 'N/A'),
            'Important Hyperparameters': config.get('params', 'N/A'),
            'CV F1 Mean': round(cv_mean, 4),
            'CV F1 Std': round(cv_std, 4),
            'Test Accuracy': round(m['accuracy'], 4),
            'Precision': round(m['precision'], 4),
            'Recall': round(m['recall'], 4),
            'F1': round(m['f1'], 4),
            'ROC-AUC': round(m['roc_auc'], 4) if m['roc_auc'] is not None else None
        })
        
    comparison_df = pd.DataFrame(records)
    return comparison_df, trained_models, predictions

def save_model_bundle(model, imputer, feature_names, metrics=None, hyperparameters=None, filepath="models/rainfall_prediction_model.pkl"):
    """
    Serializes model and all required preprocessing components for production portability.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {
        'model': model,
        'model_name': type(model).__name__,
        'imputer': imputer,
        'feature_names': feature_names,
        'hyperparameters': hyperparameters if hyperparameters else getattr(model, 'get_params', lambda: {})(),
        'metrics': metrics or {}
    }
    with open(filepath, 'wb') as f:
        pickle.dump(bundle, f)
    return filepath

def load_model_bundle(filepath="models/rainfall_prediction_model.pkl"):
    """
    Loads serialized model bundle. Falls back to root path if necessary.
    """
    if not os.path.exists(filepath):
        alt_path = "rainfall_prediction_model.pkl"
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at '{filepath}' or '{alt_path}'")
            
    with open(filepath, 'rb') as f:
        bundle = pickle.load(f)
    return bundle

def predict_raw(bundle, raw_features_df):
    """
    Takes raw feature dataframe, applies stored imputer, and predicts classes & probabilities.
    
    Parameters:
        bundle (dict): Loaded model bundle.
        raw_features_df (pd.DataFrame): Unprocessed feature input.
        
    Returns:
        tuple: (predictions, probabilities)
    """
    model = bundle['model']
    imputer = bundle['imputer']
    expected_cols = bundle['feature_names']
    
    # Ensure correct columns
    input_data = raw_features_df[expected_cols].copy()
    
    # Impute missing values with learned training medians
    imputed_array = imputer.transform(input_data)
    imputed_df = pd.DataFrame(imputed_array, columns=expected_cols, index=input_data.index)
    
    preds = model.predict(imputed_df)
    probs = model.predict_proba(imputed_df)[:, 1] if hasattr(model, 'predict_proba') else None
    
    return preds, probs
