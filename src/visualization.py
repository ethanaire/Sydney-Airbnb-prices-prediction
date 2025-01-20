# This script covers creating insightful plots to communicate trends, relationships, and key findings from the dataset, as well as the model's predictions and residuals.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_plot(fig, output_path, filename):
    """
    Save a plot to the specified directory.

    Parameters:
        fig: Matplotlib figure object.
        output_path (str): Directory to save the plot.
        filename (str): Name of the file.
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)
    fig.savefig(file_path, bbox_inches="tight")
    logging.info(f"Plot saved to {file_path}")


def plot_price_distribution(df, column, output_path):
    """
    Plot the distribution of the target variable (price).

    Parameters:
        df (pd.DataFrame): The dataset.
        column (str): The column to plot (e.g., 'price').
        output_path (str): Directory to save the plot.
    """
    logging.info("Plotting price distribution...")
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    save_plot(plt.gcf(), output_path, "price_distribution.png")
    plt.close()


def plot_feature_correlations(df, output_path):
    """
    Plot a heatmap of feature correlations.

    Parameters:
        df (pd.DataFrame): The dataset.
        output_path (str): Directory to save the plot.
    """
    logging.info("Plotting feature correlation heatmap...")
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", cbar=True)
    plt.title("Feature Correlation Heatmap")
    save_plot(plt.gcf(), output_path, "feature_correlation.png")
    plt.close()


def plot_actual_vs_predicted(y_test, predictions, output_path):
    """
    Plot actual vs. predicted prices.

    Parameters:
        y_test (pd.Series): Actual prices.
        predictions (np.ndarray): Predicted prices.
        output_path (str): Directory to save the plot.
    """
    logging.info("Plotting actual vs predicted prices...")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", lw=2)
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    save_plot(plt.gcf(), output_path, "actual_vs_predicted.png")
    plt.close()


def plot_residuals(y_test, predictions, output_path):
    """
    Plot residuals distribution.

    Parameters:
        y_test (pd.Series): Actual prices.
        predictions (np.ndarray): Predicted prices.
        output_path (str): Directory to save the plot.
    """
    logging.info("Plotting residual distribution...")
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    save_plot(plt.gcf(), output_path, "residual_distribution.png")
    plt.close()


if __name__ == "__main__":
    # File paths
    data_path = "../processed/engineered_test.csv"
    output_path = "../results/figures/"
    model_predictions_path = "../results/model_predictions.csv"

    # Target variable
    target_column = "price"

    # Load data
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Plot price distribution
    plot_price_distribution(df, column=target_column, output_path=output_path)

    # Plot feature correlations
    plot_feature_correlations(df, output_path=output_path)

    # Load predictions
    if os.path.exists(model_predictions_path):
        logging.info(f"Loading model predictions from {model_predictions_path}...")
        predictions_df = pd.read_csv(model_predictions_path)
        y_test = predictions_df["actual"]
        predictions = predictions_df["predicted"]

        # Plot actual vs. predicted prices
        plot_actual_vs_predicted(y_test, predictions, output_path)

        # Plot residual distribution
        plot_residuals(y_test, predictions, output_path)
    else:
        logging.warning(f"Predictions file not found: {model_predictions_path}. Skipping actual vs. predicted plots.")
