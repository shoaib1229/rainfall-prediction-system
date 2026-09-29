"""
Data Loader Module for Rainfall Prediction System.
Handles loading raw data, cleaning column headers, and validating dataset integrity.
"""
import os
import pandas as pd

EXPECTED_RAW_COLUMNS = [
    'pressure', 'maxtemp', 'temparature', 'mintemp',
    'dewpoint', 'humidity', 'cloud', 'rainfall',
    'sunshine', 'winddirection', 'windspeed'
]

def load_data(filepath="data/Rainfall.csv"):
    """
    Loads raw rainfall dataset, cleans column whitespace, drops identifier column,
    and validates expected columns.
    
    Parameters:
        filepath (str): Primary path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned raw dataframe.
    """
    # Robust path resolution across project root or notebooks subfolder
    if not os.path.exists(filepath):
        candidate_paths = [
            "data/Rainfall.csv",
            "../data/Rainfall.csv",
            os.path.join(os.path.dirname(__file__), "..", "data", "Rainfall.csv")
        ]
        found = False
        for cand in candidate_paths:
            if os.path.exists(cand):
                filepath = cand
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Dataset not found at '{filepath}' or any candidate locations.")
            
    df = pd.read_csv(filepath)
    
    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()
    
    # Drop identifier column 'day' if present
    if 'day' in df.columns:
        df = df.drop(columns=['day'])
        
    # Validate that all expected columns are present
    missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    return df
