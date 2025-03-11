# Sydney Airbnb Prices Prediction

## Forcasting Problem: 

To determine the rental price of one specific Airbnb, both the host and the customer have significant and difficult responsibilities. 
- For the host, they want to set a reasonable price without sacrificing the amount of profit they can earn. 
- Customers have the right to understand the significant factors influencing the price, and search for alternative options that desire comparable prices. 

The main objective of this project is to predict Airbnb listing prices in Sydney based on their property characteristics. Specifically, several machine learning models including linear regression, random forest and other models will be adopted with the availability of Scikit-learn module library in Python after cleaning the given 2 datasets.

## Evaluation Criteria:

A set of Regression Machine Learning models to be trained and applied to predict `price` feature: 
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
  
As this forecasting problem directly relates to the supervised regression topic, quality indicators such as `Mean Squared Error (MSE)`, `R-squared (R^2)`, `Mean Absolute Error(MAE)` combined with `Cross-Validation Score` are applied to compare the performance of all possible models. Specifically, the model with the smallest mean squared error will be performed on the test dataset to predict the corresponding prices.

## Prerequisites

Ensure you have the following installed before running the project:

- Python 3.x
- Required dependencies (listed in `requirements.txt`)

## Installation

The execution is managed through the `main.py` script, which sequentially runs all required steps.

1. Clone this repository:

   ```bash
   git clone https://github.com/ethanaire/Sydney-Airbnb-prices-prediction.git
   cd Sydney-Airbnb-prices-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

To execute the full pipeline, run:

```bash
python main.py
```

The script will sequentially execute the following steps:

- **Data Preprocessing** (`src/data_preprocessing.py`)
- **Feature Engineering** (`src/feature_engineering.py`)
- **Model Training** (`src/model_training.py`)
- **Prediction** (`src/prediction.py`)

Each step's progress and errors will be logged to the console.
