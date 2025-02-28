# This script includes functions for loading data, cleaning missing values and saving preprocessed data.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from math import radians, sin, cos, sqrt, atan2
import os
import logging
from sklearn.preprocessing import LabelEncoder

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
        # Normalizing Price for Different-Sized Properties
        df["price_per_bedroom"] = df["price"] / (df["bedrooms"] + 1)
        # Fill mean values for missing values of price_per_bedroom
        df["price_per_bedroom"] = df["price_per_bedroom"].fillna(df["price_per_bedroom"].mean())

    # Define reference location (e.g., Sydney city center)
    CITY_CENTER_LAT = -33.8688
    CITY_CENTER_LON = 151.2093
    df['distance_to_city_center'] = df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], CITY_CENTER_LAT, CITY_CENTER_LON), axis=1) # Add new distance_to_city_center feature
    
    # Add new review_duration feature
    df['review_duration'] = (df["last_review"] - df["first_review"]).dt.days

    # Define main verification types
    def verification_check(row, verification_type):
        return 1 if verification_type in row['host_verifications'] else 0

    verification_types = ['email', 'phone', 'reviews', 'jumio']
    for verification in verification_types:
        df[verification] = df.apply(lambda row: verification_check(row, verification), axis=1) 

    # Define main amenity types    
    def amenity_check(row, amenity_type):
        return 1 if amenity_type in row['amenities'] else 0   

    amenity_types = ['Long term stays allowed', 'Wifi', 'Essentials', 'Smoke alarm']
    for amenity in amenity_types:
        df[amenity] = df.apply(lambda row: amenity_check(row, amenity), axis=1)

    # Fill text_based columns with 'unknown'
    df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].fillna(value='unknonwn')

    # Define all numeric cols to fill with mode values
    mode_fill_cols = ['host_location', 'host_response_time', 'property_type', 'room_type', 'bathrooms']
    df[mode_fill_cols] = df[mode_fill_cols].fillna(df[mode_fill_cols].mode().iloc[0])

    # Define all numeric cols to fill with mean values
    mean_fill_cols = ['host_response_rate', 'host_acceptance_rate', 'bedrooms', 'beds', 'minimum_minimum_nights', 'maximum_maximum_nights', 'availability_365', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'reviews_per_month', 'review_duration']
    df[mean_fill_cols] = df[mean_fill_cols].fillna(df[mean_fill_cols].mean())

    # Convert selected columns into int types
    int_cols = ['bedrooms','beds','availability_365','minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm']
    df[int_cols] = df[int_cols].astype(int)    

    # Categorize and encode property_type
    entire_townhouse_types = ['Entire villa', 'Entire residential home','Entire guest suite','Entire guesthouse','Entire bungalow','Tiny house','Entire place','Entire vacation home','Dome house','Earth house','Casa particular']
    entire_rental_unit_types = ['Entire serviced apartment', 'Entire loft', 'Entire rental unit', 'Entire condominium (condo)', 'Entire cottage']
    
    def map_property_type(pt):
        if 'Private room' in str(pt): return 'Private room'
        if 'Shared room' in str(pt): return 'Shared room'
        if 'Room in' in str(pt): return 'Entire room'
        if pt in entire_townhouse_types: return 'Entire townhouse'
        if pt in entire_rental_unit_types: return 'Entire rental unit'
        return 'Other'
    
    df['property_type'] = df['property_type'].apply(map_property_type)
    property_type_mapping = {'Other':6, 'Entire townhouse':5, 'Entire rental unit':4, 'Entire room':3, 'Private room':2, 'Shared room':1}
    df['mapped_property_type'] = df['property_type'].map(property_type_mapping)

    # Simplify and map bathrooms
    def simplify_bathrooms(value):
        value = str(value) if pd.notna(value) else 'Other'
        if '1.5 baths' in value: return '1 bath'
        if '1.5 shared baths' in value: return '1 shared bath'
        if any(x in value for x in ['2', '3', '4', '5', '6', '7', '11', '19']): return 'Many baths' if 'shared' not in value else 'Many shared baths'
        if any(x in value for x in ['0', 'half', 'Half', 'nan']): return 'Other'
        return value
    
    df['bathrooms'] = df['bathrooms'].astype(str).apply(simplify_bathrooms)
    bathrooms_mapping = {'Other':1, 'Many shared baths':2, 'Many baths':3, '1 shared bath':4, '1 private bath':5, '1 bath':6}
    df['mapped_bathrooms'] = df['bathrooms'].map(bathrooms_mapping)

    # Encode room_type column
    room_mapping = {'Shared room':1, 'Hotel room':2, 'Private room':3, 'Entire home/apt':4}
    df['mapped_room_type'] = df['room_type'].map(room_mapping)

    # Encode host_response_time column 
    response_time_mapping = {'a few days or more':1, 'within a day':2, 'within a few hours':3, 'within an hour':4}
    df['response_time'] = df['host_response_time'].map(response_time_mapping)

    # Encode boolean columns 
    columns_to_replace = ['has_availability', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    df[columns_to_replace] = df[columns_to_replace].replace({'t':1, 'f':0})

    # Apply label encoding to neighbourhood_cleansed column 
    label_encoder = LabelEncoder()
    df["neighbourhood_encoded"] = label_encoder.fit_transform(df["neighbourhood_cleansed"])

    # Drop rows with critical missing values
    df.dropna(subset=['host_neighbourhood', 'neighbourhood'], inplace=True)

    # Drop original columns after creating new feature -> last step
    df.drop(columns=['latitude', 'longitude', 'first_review', 'last_review', 'host_verifications', 'amenities', 'property_type', 'bathrooms', 'room_type','host_response_time', 'neighbourhood_cleansed'], inplace=True)
    
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

