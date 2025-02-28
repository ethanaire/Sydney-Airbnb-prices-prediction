# This script includes functions for feature engineering and saving processed data.

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_train_features(df):
    """
    Engineer new features for analysis and modeling.

    Parameters:
        df (pd.DataFrame): Train dataframe for feature engineering.

    Returns:
        pd.DataFrame: Train dataframe with new features.
    """
    logging.info("Starting feature engineering...")

    # Exclude object features
    df = df.select_dtypes(exclude=['object'])

    # Conduct Correlation Analysis
    correlation_matrix = df.corr()
    price_correlations = correlation_matrix["price"].sort_values(ascending=False)

    # Select features with correlation > threshold (excluding 'price' itself)
    corr_threshold = 0.2
    selected_features_corr = price_correlations[abs(price_correlations) > corr_threshold].index.tolist()
    selected_features_corr.remove("price")  # Remove target variable

    # Compute mutual information scores for feature selection
    X = df.drop(columns=["price"])
    y = df["price"]
    mutual_info = mutual_info_regression(X, y)
    mutual_info_series = pd.Series(mutual_info, index=X.columns)

    # Filter features with significant mutual information
    mi_threshold = 0.02
    selected_features_mi = mutual_info_series[mutual_info_series > mi_threshold].index.tolist()
    selected_features_mi

    # Final list of selected features (intersection of correlation and mutual information)
    final_selected_features = list(set(selected_features_corr) & set(selected_features_mi))

    logging.info("Train data feature engineering completed.")
    return df


def engineer_test_features(df):
    """
    Engineer new features for analysis and modeling.

    Parameters:
        df (pd.DataFrame): DataFrame for feature engineering.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    logging.info("Starting feature engineering...")

    # Exclude object features
    df = df.select_dtypes(exclude=['object'])

    # Conduct Correlation Analysis
    correlation_matrix = df.corr()
    price_correlations = correlation_matrix["price"].sort_values(ascending=False)

    logging.info("Test data feature engineering completed.")
    return df


def process_features(input_path, output_path):
    """
    Load cleaned data, engineer features, and save the result.

    Parameters:
        input_path (str): Path to the cleaned input dataset.
        output_path (str): Path to save the feature-engineered dataset.
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    logging.info(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)

    # Engineer features
    df = engineer_features(df)

    # Save the dataset with engineered features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Feature-engineered data saved to {output_path}")


if __name__ == "__main__":
    # File paths
    cleaned_train_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/processed_train.csv"
    cleaned_test_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/processed_test.csv"
    engineered_train_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/feature_engineered_train.csv"
    engineered_test_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/feature_engineered_test.csv"

    # Process train and test datasets
    process_features(cleaned_train_path, engineered_train_path)
    process_features(cleaned_test_path, engineered_test_path)


