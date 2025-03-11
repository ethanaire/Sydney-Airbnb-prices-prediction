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

    # Drop weak features from the training DataFrame
    df_train_selected = df[final_selected_features + ["price"]]

    logging.info("Train data feature engineering completed.")
    return df_train_selected, final_selected_features


def engineer_test_features(df, selected_features):
    """
    Engineer new features for analysis and modeling.

    Parameters:
        df (pd.DataFrame): Test dataframe for feature engineering.
        selected_features (list): List of features selected from training data.

    Returns:
        pd.DataFrame: Test dataframe with the same selected features as training.
    """
    logging.info("Starting test feature engineering...")

    # Retain only the selected features
    df_test_selected = df[selected_features]
    
    logging.info("Test data feature engineering completed.")
    return df_test_selected


if __name__ == "__main__":
    # File paths
    cleaned_train_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/processed_train.csv"
    cleaned_test_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/processed_test.csv"
    engineered_train_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/feature_engineered_train.csv"
    engineered_test_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/feature_engineered_test.csv"

    # Process train dataset
    df_train = pd.read_csv(cleaned_train_path)
    df_train_selected, final_selected_features = engineer_train_features(df_train)
    df_train_selected.to_csv(engineered_train_path, index=False)
    
    # Process test dataset
    df_test = pd.read_csv(cleaned_test_path)
    df_test_selected = engineer_test_features(df_test, final_selected_features)
    df_test_selected.to_csv(engineered_test_path, index=False)


