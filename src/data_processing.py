import pandas as pd
import numpy as np
import logging
import os

def load_data(path):
    """
    Load CSV data from the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    logging.info(f"Loaded data from {path} with shape {df.shape}")
    return df

def clean_data(df):
    """
    Clean the flood dataset:
    - Fill numeric NaNs with median
    - Drop rows with remaining NaNs
    - Assert valid latitude/longitude
    """
    # Fill numeric NaNs with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    # Drop rows with any remaining NaNs
    df = df.dropna(axis=0)
    # Validate latitude/longitude
    assert df['Latitude'].between(-90, 90).all(), "Invalid latitude values detected!"
    assert df['Longitude'].between(-180, 180).all(), "Invalid longitude values detected!"
    logging.info(f"Cleaned data. Shape: {df.shape}")
    return df

def add_risk_category(df):
    """
    Add a categorical risk level column based on Danger Level.
    """
    def risk_cat(danger):
        if danger >= 300:
            return "Very High"
        elif danger >= 200:
            return "High"
        elif danger >= 100:
            return "Medium"
        elif danger >= 50:
            return "Low"
        else:
            return "Very Low"
    df['Risk_Level'] = df['Danger Level'].apply(risk_cat)
    return df

def save_data(df, path):
    """
    Save DataFrame to CSV at the given path.
    """
    df.to_csv(path, index=False)
    logging.info(f"Saved data to {path}")
