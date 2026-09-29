"""
Inference Module for Rainfall Prediction System.
Handles model loading, input validation, leak-free imputation using the stored imputer,
and generating predictions and probabilities without model retraining.
"""
import os
import pickle
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = "models/rainfall_prediction_model.pkl"
EXPECTED_FEATURES = [
    'pressure', 'dewpoint', 'humidity', 'cloud',
    'sunshine', 'winddirection', 'windspeed'
]

def load_model(filepath=DEFAULT_MODEL_PATH):
    """
    Loads the serialized model bundle from disk.
    
    Parameters:
        filepath (str): Path to the pickle file.
        
    Returns:
        dict: Bundle containing model, imputer, feature_names, hyperparameters, and metrics.
    """
    if not os.path.exists(filepath):
        alt_path = "rainfall_prediction_model.pkl"
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at '{filepath}' or '{alt_path}'")
            
    with open(filepath, 'rb') as f:
        bundle = pickle.load(f)
        
    # Validate bundle integrity
    required_keys = ['model', 'imputer', 'feature_names']
    for k in required_keys:
        if k not in bundle:
            raise KeyError(f"Serialized model bundle is missing required key '{k}'")
            
    return bundle

def validate_input(input_data, expected_features=None):
    """
    Validates input features, converts values to float, handles missing inputs,
    and returns a pandas DataFrame with the exact feature order.
    
    Parameters:
        input_data (dict or pd.DataFrame): Feature name to value mapping.
        expected_features (list, optional): Expected feature list. Defaults to EXPECTED_FEATURES.
        
    Returns:
        pd.DataFrame: Formatted dataframe matching expected feature order.
    """
    if expected_features is None:
        expected_features = EXPECTED_FEATURES
        
    if isinstance(input_data, dict):
        # Convert dict keys to lowercase stripped
        normalized_input = {str(k).strip().lower().replace('-', '').replace('_', ''): v for k, v in input_data.items()}
        
        row_dict = {}
        for feat in expected_features:
            norm_feat = feat.lower().replace('_', '')
            if norm_feat in normalized_input:
                val = normalized_input[norm_feat]
                if val is None or val == '' or pd.isna(val):
                    row_dict[feat] = np.nan
                else:
                    try:
                        row_dict[feat] = float(val)
                    except (ValueError, TypeError):
                        raise ValueError(f"Feature '{feat}' received non-numeric value: '{val}'")
            else:
                # Missing feature defaults to NaN so the stored imputer can handle it
                row_dict[feat] = np.nan
                
        df = pd.DataFrame([row_dict], columns=expected_features)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace('-', '').str.replace('_', '')
        
        row_dict = {}
        for feat in expected_features:
            norm_feat = feat.lower().replace('_', '')
            if norm_feat in df.columns:
                row_dict[feat] = pd.to_numeric(df[norm_feat], errors='coerce')
            else:
                row_dict[feat] = np.nan
        df = pd.DataFrame(row_dict, columns=expected_features)
    else:
        raise TypeError(f"input_data must be a dict or pd.DataFrame, received: {type(input_data)}")
        
    return df

def predict_rainfall(model_bundle, input_data):
    """
    Generates binary rainfall prediction (1 for Rain, 0 for No Rain).
    
    Parameters:
        model_bundle (dict): Loaded model bundle.
        input_data (dict or pd.DataFrame): Raw feature values.
        
    Returns:
        tuple: (raw_prediction_int, rain_label_str)
    """
    expected_features = model_bundle.get('feature_names', EXPECTED_FEATURES)
    imputer = model_bundle['imputer']
    model = model_bundle['model']
    
    # 1. Validate & align features
    df = validate_input(input_data, expected_features=expected_features)
    
    # 2. Impute with stored training medians
    imputed_array = imputer.transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=expected_features)
    
    # 3. Predict without retraining
    preds = model.predict(imputed_df)
    raw_pred = int(preds[0])
    label = "Yes" if raw_pred == 1 else "No"
    
    return raw_pred, label

def predict_probability(model_bundle, input_data):
    """
    Generates class probabilities for rainfall occurrence.
    
    Parameters:
        model_bundle (dict): Loaded model bundle.
        input_data (dict or pd.DataFrame): Raw feature values.
        
    Returns:
        float: Probability of rainfall (between 0.0 and 1.0).
    """
    expected_features = model_bundle.get('feature_names', EXPECTED_FEATURES)
    imputer = model_bundle['imputer']
    model = model_bundle['model']
    
    df = validate_input(input_data, expected_features=expected_features)
    imputed_array = imputer.transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=expected_features)
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(imputed_df)
        rain_prob = float(probs[0, 1])
    else:
        raise AttributeError(f"Model {type(model).__name__} does not support probability estimation.")
        
    return rain_prob

def predict_single(model_bundle, input_dict):
    """
    Comprehensive single-sample inference helper returning structured results.
    
    Returns:
        dict: Complete prediction details including probabilities and formatted percentages.
    """
    raw_pred, label = predict_rainfall(model_bundle, input_dict)
    prob = predict_probability(model_bundle, input_dict)
    
    return {
        'prediction': raw_pred,
        'rain_label': label,
        'probability': round(prob, 4),
        'probability_percent': f"{prob * 100:.2f}%",
        'model_name': model_bundle.get('model_name', type(model_bundle['model']).__name__),
        'features_used': validate_input(input_dict, model_bundle.get('feature_names')).iloc[0].to_dict()
    }
