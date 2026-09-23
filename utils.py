import pandas as pd

def load_data(file_path) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
    """
    return pd.read_csv(file_path)