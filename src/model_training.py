# This script covers training, evaluation, and saving models in a structured, reusable way.

import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
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


def train_with_grid_search(model, param_grid, X_train, y_train, cv=5):
    """
    Train a model using GridSearchCV for hyperparameter tuning.

    Parameters:
        model: The base model to train.
        param_grid (dict): Dictionary of hyperparameters to search.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cv (int): Number of cross-validation folds.

    Returns:
        Best estimator from GridSearchCV.
    """
    logging.info(f"Starting GridSearchCV for {model.__class__.__name__}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    logging.info(f"Best R2 score for {model.__class__.__name__}: {grid_search.best_score_}")
    return grid_search.best_estimator_


if __name__ == "__main__":
    # File paths
    engineered_train_path = "../processed/engineered_train.csv"
    model_output_path = "../results/models/"

    # Target variable
    target_column = "price"

    # Load data
    df = load_data(engineered_train_path)
    if df is None:
        logging.error("Data loading failed. Model training aborted.")
        exit()

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target=target_column)

    # Models and hyperparameters
    models_and_parameters = {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"alpha": [0.1, 1, 10]}),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
        ),
        "SVR": (
            SVR(),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        ),
    }

    # Train and evaluate models
    best_model = None
    best_metrics = None
    best_model_name = None

    os.makedirs(model_output_path, exist_ok=True)

    for model_name, (model, param_grid) in models_and_parameters.items():
        logging.info(f"Training {model_name}...")
        if param_grid:
            trained_model = train_with_grid_search(model, param_grid, X_train, y_train)
        else:
            trained_model = model.fit(X_train, y_train)

        metrics = evaluate_model(trained_model, X_test, y_test)

        if not best_metrics or metrics["R2"] > best_metrics["R2"]:
            best_model = trained_model
            best_metrics = metrics
            best_model_name = model_name

        # Save individual model
        model_path = os.path.join(model_output_path, f"{model_name}_model.pkl")
        joblib.dump(trained_model, model_path)
        logging.info(f"{model_name} saved to {model_path}")

    logging.info(f"Best model: {best_model_name} with metrics: {best_metrics}")

    # Save best model
    best_model_path = os.path.join(model_output_path, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    logging.info(f"Best model saved to {best_model_path}")
