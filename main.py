# Telco customer churn prediction using Scikitlearn, TensorFlow and Keras

import os
from pathlib import Path
from utils import load_data
from eda import explore_data

def main():
    # Set up the environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

    current_dir = Path(__file__).resolve().parent
    data_file = current_dir / "data/Telco-Customer-Churn.csv"

    # Check if plots directory exists, if not create it
    plots_dir = current_dir / "plots"
    if not plots_dir.exists():
        plots_dir.mkdir()

    # Check if models directory exists, if not create it
    models_dir = current_dir / "models"
    if not models_dir.exists():
        models_dir.mkdir()

    # Load the dataset
    print("Loading the dataset...")
    data = load_data(data_file)

    # Basic data exploration and visualization
    explore_data(data, output_dir=plots_dir)

    # Preprocess the data for modeling
    # X_train, X_test, y_train, y_test = preprocess_data(data)

    # # Build the model
    # model = build_model(input_shape=X_train.shape[1])

    # # Train the model
    # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # # Evaluate the model
    # evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()