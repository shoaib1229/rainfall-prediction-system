"""
Preprocessing Module for Rainfall Prediction System.
Handles feature selection, target encoding, stratified splitting, and leak-free imputation.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Features to drop due to extreme multicollinearity with dewpoint/temperature
CORRELATED_TEMP_FEATURES = ['maxtemp', 'temparature', 'mintemp']

def drop_correlated_features(df):
    """
    Removes redundant temperature columns from the dataframe.
    """
    cols_to_drop = [col for col in CORRELATED_TEMP_FEATURES if col in df.columns]
    return df.drop(columns=cols_to_drop)

def encode_target(df, target_col='rainfall'):
    """
    Encodes binary rainfall target (yes -> 1, no -> 0).
    Works on either string or numeric inputs.
    """
    df_copy = df.copy()
    if target_col in df_copy.columns:
        if df_copy[target_col].dtype == object or isinstance(df_copy[target_col].iloc[0], str):
            clean_s = df_copy[target_col].astype(str).str.strip().str.lower()
            df_copy[target_col] = clean_s.map({'yes': 1, 'no': 0}).astype(int)
        else:
            df_copy[target_col] = df_copy[target_col].astype(int)
    return df_copy

def split_data(df, target_col='rainfall', test_size=0.2, random_state=42):
    """
    Separates features (X) and target (y) and performs stratified train/test split.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing features and target.
        target_col (str): Target column name.
        test_size (float): Proportion for test split (default 0.2).
        random_state (int): Seed for reproducibility (default 42).
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def impute_missing_values(X_train, X_test, strategy='median'):
    """
    Fits SimpleImputer strictly on X_train and transforms both X_train and X_test.
    Preserves column names and indices. Ensures zero data leakage.
    
    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        strategy (str): Imputation strategy (default 'median').
        
    Returns:
        tuple: (X_train_imp, X_test_imp, fitted_imputer)
    """
    imputer = SimpleImputer(strategy=strategy)
    
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_imp, X_test_imp, imputer

def prepare_data(df, target_col='rainfall', test_size=0.2, random_state=42):
    """
    High-level convenience function executing the complete leak-free preprocessing pipeline:
    1. Drop correlated temperature features
    2. Encode target variable
    3. Stratified split
    4. Train-only median imputation
    
    Returns:
        dict: Processed datasets, imputer, and feature column names.
    """
    df_clean = drop_correlated_features(df)
    df_encoded = encode_target(df_clean, target_col=target_col)
    
    X_train, X_test, y_train, y_test = split_data(
        df_encoded, target_col=target_col, test_size=test_size, random_state=random_state
    )
    
    X_train_imp, X_test_imp, imputer = impute_missing_values(X_train, X_test, strategy='median')
    
    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train': X_train_imp,
        'X_test': X_test_imp,
        'y_train': y_train,
        'y_test': y_test,
        'imputer': imputer,
        'feature_names': X_train.columns.tolist()
    }
