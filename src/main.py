import logging
import os
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the paths to the scripts
scripts = {
    "Data Preprocessing": "src/data_preprocessing.py",
    "Feature Engineering": "src/feature_engineering.py",
    "Model Training": "src/model_training.py",
    "Model Evaluation": "src/model_evaluation.py",
    "Visualization": "src/visualization.py",
}

def run_script(script_name, script_path):
    """
    Run a Python script and log its status.

    Parameters:
        script_name (str): Name of the script.
        script_path (str): Path to the script.

    Returns:
        bool: True if the script runs successfully, False otherwise.
    """
    logging.info(f"Starting {script_name}...")
    if not os.path.exists(script_path):
        logging.error(f"{script_name} script not found: {script_path}")
        return False

    try:
        subprocess.run(["python", script_path], check=True)
        logging.info(f"{script_name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running {script_name}: {e}")
        return False

if __name__ == "__main__":
    # Run all scripts sequentially
    for name, path in scripts.items():
        success = run_script(name, path)
        if not success:
            logging.error(f"Pipeline aborted due to failure in {name}.")
            break
    else:
        logging.info("Pipeline completed successfully.")
