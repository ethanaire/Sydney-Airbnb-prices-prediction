# This script covers loading the trained model, evaluating it on test data, and generating detailed evaluation metrics and visualizations.

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.

    Parameters:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        dict: Evaluation metrics and predictions.
    """
    logging.info("Evaluating model...")
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    logging.info(f"Evaluation Metrics: {metrics}")

    return metrics, predictions


def plot_results(y_test, predictions, output_path):
    """
    Generate evaluation plots and save them.

    Parameters:
        y_test (pd.Series): Actual target values.
        predictions (np.ndarray): Predicted target values.
        output_path (str): Path to save the plot.
    """
    logging.info("Generating evaluation plots...")

    # Scatter plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", lw=2)
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "actual_vs_predicted.png"))
    logging.info(f"Scatter plot saved to {output_path}")

    # Residual plot
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "residual_distribution.png"))
    logging.info(f"Residual plot saved to {output_path}")


if __name__ == "__main__":
    # File paths
    test_data_path = "../processed/engineered_test.csv"
    model_path = "../results/models/airbnb_model.pkl"
    evaluation_output_path = "../results/figures/"

    # Target variable
    target_column = "price"

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

    # Split test data
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Evaluate model
    metrics, predictions = evaluate_model(model, X_test, y_test)

    # Print metrics
    logging.info(f"Final Evaluation Metrics: {metrics}")

    # Create evaluation output directory
    os.makedirs(evaluation_output_path, exist_ok=True)

    # Plot and save results
    plot_results(y_test, predictions, evaluation_output_path)


