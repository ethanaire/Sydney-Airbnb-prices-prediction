# This script includes functions for feature engineering and saving processed data.

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_features(df):
    """
    Engineer new features for analysis and modeling.

    Parameters:
        df (pd.DataFrame): DataFrame for feature engineering.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    logging.info("Starting feature engineering...")

    # Host-related features
    df['host_experience_days'] = (datetime.now() - df['host_since']).dt.days
    df['is_superhost'] = df['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)

    # Review features
    df['review_recency_days'] = (datetime.now() - df['last_review']).dt.days
    df['reviews_per_year'] = df['number_of_reviews'] / (df['host_experience_days'] / 365).clip(1)

    # Availability categories
    df['availability_category'] = pd.cut(
        df['availability_365'], bins=[0, 100, 200, 365], labels=["Low", "Medium", "High"]
    )

    # Geospatial features
    sydney_cbd = (-33.8688, 151.2093)
    df['distance_to_cbd'] = df.apply(
        lambda x: geodesic((x['latitude'], x['longitude']), sydney_cbd).km, axis=1
    )

    # Pricing features
    df['price_per_guest'] = df['price'] / df['accommodates'].clip(1)

    # Feature interactions
    df['host_reliability_score'] = (df['host_experience_days'] * df['reviews_per_year']).clip(upper=1000)
    df['price_availability_ratio'] = df['price'] / (df['availability_365'].clip(1))

    logging.info("Feature engineering completed.")
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
    df = pd.read_csv(input_path, parse_dates=["host_since", "first_review", "last_review"])

    # Engineer features
    df = engineer_features(df)

    # Save the dataset with engineered features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Feature-engineered data saved to {output_path}")


if __name__ == "__main__":
    # File paths
    cleaned_train_path = "../outputs/cleaned_train.csv"
    cleaned_test_path = "../outputs/cleaned_test.csv"
    engineered_train_path = "../outputs/engineered_train.csv"
    engineered_test_path = "../outputs/engineered_test.csv"

    # Process train and test datasets
    process_features(cleaned_train_path, engineered_train_path)
    process_features(cleaned_test_path, engineered_test_path)


