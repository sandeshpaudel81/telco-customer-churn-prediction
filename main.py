# Telco customer churn prediction using Scikitlearn, TensorFlow and Keras

import os
from pathlib import Path
from utils import load_data
from eda import explore_data
from models import train_models, evaluate_models
from preprocessing import preprocessing as preprocess_data
from feature_engineering import feature_engineering
from post_analysis import run_feature_ablation, run_threshold_tuning, run_shap_analysis, run_feature_selection

def main():
    # Set up the environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

    current_dir = Path(__file__).resolve().parent

    data_file = current_dir / "data" / "Telco-Customer-Churn.csv"

    eda_plots_dir = current_dir / "plots/eda"
    model_plots_dir = current_dir / "plots/models"
    models_dir = current_dir / "models"
    results_dir = current_dir / "results"

    # Create required directories
    if not eda_plots_dir.exists():
        eda_plots_dir.mkdir(parents=True)
    if not model_plots_dir.exists():
        model_plots_dir.mkdir(parents=True)
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
    explore_data(data, output_dir=eda_plots_dir)

    # Create new features from the raw data
    print("\nEngineering features...")
    data = feature_engineering(data)

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
        results_dir,
        plots_dir=model_plots_dir
    )

    # --------------------------------------------------
    # 7. Post-training analysis
    # --------------------------------------------------
    run_feature_ablation(
        trained_models,
        X_train,
        y_train,
        plots_dir=model_plots_dir,
        results_dir=results_dir
    )
    run_feature_selection(
        trained_models,
        X_train,
        y_train,
        results_dir=results_dir
    )
    run_threshold_tuning(
        trained_models,
        X_train,
        y_train,
        X_test, 
        y_test,
        results_dir,
        model_plots_dir
    )
    run_shap_analysis(
        trained_models, 
        X_test, 
        y_test, 
        results_dir,
        model_plots_dir,
        model_key="XGBoost (SMOTE)"
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