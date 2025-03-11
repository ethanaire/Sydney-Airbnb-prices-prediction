# This script covers loading the trained model, evaluating it on test data, and generating detailed evaluation metrics and visualizations.

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    """
    Load a trained model from a file.

    Parameters:
        model_path (str): Path to the model file.

    Returns:
        Trained model or None if loading fails.
    """
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None

    logging.info(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


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


def produce_prediction(model, file_path):
    """
    Evaluate the trained model on the test data.

    Parameters:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        dict: Evaluation metrics and predictions.
    """
    logging.info(f"Loading data from {file_path}...")
    df_test = pd.read_csv(file_path)

    logging.info("Evaluating model...")
    predictions = model.predict(df_test)

    return predictions


if __name__ == "__main__":
    # File paths
    test_data_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/feature_engineered_test.csv"
    model_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/results/best_model_Gradient Boosting.pkl"
    output_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/results/"
    predictions_file_name = "test_predictions.csv"
    
    # Load test data
    df = load_data(test_data_path)
    if df is None:
        logging.error("Test data loading failed. Evaluation aborted.")
        exit()

    # Load model
    model = load_model(model_path)
    if model is None:
        logging.error("Model loading failed. Evaluation aborted.")
        exit()

    # Apply model on test dataset
    predictions = produce_prediction(model, test_data_path)

    # Create evaluation output directory
    os.makedirs(output_path, exist_ok=True)

    # Save predictions to CSV
    df["Predicted_Price"] = predictions
    predictions_file_path = Path(output_path) / predictions_file_name
    df.to_csv(predictions_file_path, index=False)



