import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging

def get_features_and_target(df, target_col, drop_cols=None):
    """
    Select features and target for modeling.
    """
    if drop_cols is None:
        drop_cols = []
    features = [col for col in df.columns if col not in drop_cols + [target_col]]
    X = df[features]
    y = df[target_col]
    return X, y

def train_models(X_train, y_train):
    """
    Train multiple regression models and return them in a dict.
    """
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(),
        "MLP": MLPRegressor(random_state=42, max_iter=500)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        logging.info(f"{name} trained.")
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and return a DataFrame of metrics and predictions.
    """
    results = {}
    predictions = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        predictions[name] = preds
        logging.info(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    return pd.DataFrame(results).T, predictions

def save_model(model, path):
    """
    Save a trained model to disk.
    """
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")

def load_model(path):
    """
    Load a model from disk.
    """
    return joblib.load(path)
