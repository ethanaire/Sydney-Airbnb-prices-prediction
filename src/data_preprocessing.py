# This script includes functions for loading data, cleaning missing values and saving preprocessed data.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from math import radians, sin, cos, sqrt, atan2
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

# Haversine formula to calculate distance in km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])  # Convert degrees to radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def clean_data(df):
    """
    Perform basic data cleaning, including handling missing values.

    Parameters:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logging.info("Cleaning data...")

    # Drop non-essential columns
    drop_cols = ["ID", "name", "description", "neighborhood_overview", "host_about", "license"]
    df.drop(columns=drop_cols, inplace=True)

    # Convert date columns to datetime
    date_cols = ["host_since", "first_review", "last_review"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # Remove unwanted characters and transform datatype 
    percent_cols = ['host_response_rate', 'host_acceptance_rate']
    for col in percent_cols:
            df[col] = df[col].str.rstrip("%").astype(float) / 100

    # Clean price column for the train dataset
    if 'price' in df.columns: 
        df['price'] = df['price'].replace({"\$": "", ",": ""}, regex=True).astype(float)

    # Define reference location (e.g., Sydney city center)
    CITY_CENTER_LAT = -33.8688
    CITY_CENTER_LON = 151.2093

    # Add new distance_to_city_center feature
    df['distance_to_city_center'] = df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], CITY_CENTER_LAT, CITY_CENTER_LON), axis=1)
    
    # Add new review_duration feature
    df['review_duration'] = (df["last_review"] - df["first_review"]).dt.days

    # Drop original columns after creating new feature
    df.drop(columns=['latitude', 'longitude', 'first_review', 'last_review'], inplace=True)



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
    train_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/raw/train.csv"
    test_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/raw/test.csv"
    cleaned_train_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/processed_train.csv"
    cleaned_test_path = "C:/Users/haiho/GITHUB/Sydney-Airbnb-prices-prediction/data/processed/processed_test.csv"

    # Columns to parse as dates
    date_columns = ["host_since", "first_review", "last_review"]

    # Process train and test datasets
    preprocess_data(train_path, cleaned_train_path, parse_dates=date_columns)
    preprocess_data(test_path, cleaned_test_path, parse_dates=date_columns)

