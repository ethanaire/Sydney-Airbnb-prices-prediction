# This script includes functions for loading data, cleaning missing values and saving preprocessed data.

import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path, parse_dates=None):
    """
    Load a CSV file into a pandas DataFrame with optional date parsing.

    Parameters:
        file_path (str): Path to the CSV file.
        parse_dates (list, optional): List of columns to parse as dates.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, parse_dates=parse_dates)
        logging.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def clean_data(df):
    """
    Perform basic data cleaning, including handling missing values.

    Parameters:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logging.info("Cleaning data...")

    # Handle missing values
    missing_threshold = 0.3  # Drop columns with >30% missing values
    df = df.dropna(thresh=len(df) * (1 - missing_threshold), axis=1)
    df.fillna({"host_is_superhost": "f", "availability_365": 0}, inplace=True)

    # Drop rows with critical missing values
    df.dropna(subset=['price', 'latitude', 'longitude'], inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    logging.info("Data cleaning completed.")
    return df


def preprocess_data(input_path, output_path, parse_dates):
    """
    Full data preprocessing pipeline: load and clean data.

    Parameters:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the cleaned dataset.
        parse_dates (list): List of columns to parse as dates.
    """
    # Load data
    df = load_data(input_path, parse_dates=parse_dates)
    if df is None:
        logging.error("Data loading failed. Preprocessing aborted.")
        return None

    # Clean data
    df = clean_data(df)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    # File paths
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    cleaned_train_path = "../processed/cleaned_train.csv"
    cleaned_test_path = "../processed/cleaned_test.csv"

    # Columns to parse as dates
    date_columns = ["host_since", "first_review", "last_review"]

    # Process train and test datasets
    preprocess_data(train_path, cleaned_train_path, parse_dates=date_columns)
    preprocess_data(test_path, cleaned_test_path, parse_dates=date_columns)

