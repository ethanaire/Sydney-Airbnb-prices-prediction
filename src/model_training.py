# This script covers training, evaluation, and saving models in a structured, reusable way.

import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load feature-engineered data.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def split_data(df, target, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and target.
        target (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    logging.info("Splitting data into train and test sets...")
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    """
    Train a machine learning model.

    Parameters:
        model: The machine learning model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        Trained model.
    """
    logging.info(f"Training model: {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a machine learning model.

    Parameters:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        dict: Evaluation metrics.
    """
    logging.info(f"Evaluating model: {model.__class__.__name__}...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    logging.info(f"Evaluation Metrics: {metrics}")
    return metrics


def save_model(model, output_path):
    """
    Save the trained model to a file.

    Parameters:
        model: Trained model.
        output_path (str): Path to save the model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logging.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    # File paths
    engineered_train_path = "../outputs/engineered_train.csv"
    model_output_path = "../models/airbnb_model.pkl"

    # Target variable
    target_column = "price"

    # Load data
    df = load_data(engineered_train_path)
    if df is None:
        logging.error("Data loading failed. Model training aborted.")
        exit()

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target=target_column)

    # Train models
    linear_model = LinearRegression()
    rf_model = RandomForestRegressor(random_state=42)

    trained_linear_model = train_model(linear_model, X_train, y_train)
    trained_rf_model = train_model(rf_model, X_train, y_train)

    # Evaluate models
    linear_metrics = evaluate_model(trained_linear_model, X_test, y_test)
    rf_metrics = evaluate_model(trained_rf_model, X_test, y_test)

    # Select and save the best model
    if rf_metrics["R2"] > linear_metrics["R2"]:
        best_model = trained_rf_model
        logging.info("Random Forest Regressor selected as the best model.")
    else:
        best_model = trained_linear_model
        logging.info("Linear Regression selected as the best model.")

    save_model(best_model, model_output_path)


