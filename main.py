# Telco customer churn prediction using Scikitlearn, TensorFlow and Keras

import os
from pathlib import Path
from utils import load_data
from eda import explore_data
from models import train_models, evaluate_models
from preprocessing import preprocessing as preprocess_data

def main():
    # Set up the environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

    current_dir = Path(__file__).resolve().parent

    data_file = current_dir / "data" / "Telco-Customer-Churn.csv"

    plots_dir = current_dir / "plots"
    models_dir = current_dir / "models"
    results_dir = current_dir / "results"

    # Create required directories
    if not plots_dir.exists():
        plots_dir.mkdir()
    if not models_dir.exists():
        models_dir.mkdir()
    if not results_dir.exists():
        results_dir.mkdir()

    data_file = current_dir / "data/Telco-Customer-Churn.csv"

    # Load the dataset
    print("Loading the dataset...")
    data = load_data(data_file)

    print(f"\nDataset shape: {data.shape}")

    # Basic data exploration and visualization
    print("\nRunning exploratory data analysis...")
    explore_data(data, output_dir=plots_dir)

    # Preprocess the data for modeling
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        data
    )

    # --------------------------------------------------
    # 5. Train machine learning models
    # --------------------------------------------------
    print("\nTraining machine learning models...")
    trained_models = train_models(
        X_train,
        y_train,
        preprocessor
    )


    # --------------------------------------------------
    # 6. Evaluate models
    # --------------------------------------------------

    print("\nEvaluating models...")

    results = evaluate_models(
        trained_models,
        X_test,
        y_test,
        results_dir
    )


    # --------------------------------------------------
    # 7. Display final results
    # --------------------------------------------------

    print("\nModel Performance")
    print("-" * 60)

    print(results)

    print("\nProject completed successfully.")

if __name__ == "__main__":
    main()