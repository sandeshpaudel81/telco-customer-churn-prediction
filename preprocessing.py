import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocessing(data):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.

    Args:
        data (pd.DataFrame): The input dataset.
    Returns:
        X_train, X_test, y_train, y_test, preprocessor: Preprocessed training and testing data along with the preprocessor object.
    """

    data = data.copy()

    # Convert SeniorCitizen to readable categories for EDA
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = data['SeniorCitizen'].map({
            0: 'No',
            1: 'Yes'
        })
 
    # Fix: TotalCharges is stored as a string with blank entries for
    # customers with tenure == 0 (brand new accounts). Coerce to numeric
    # first, THEN drop/impute, otherwise dropna() won't catch the blanks.
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
 
    # Drop rows with missing values (now that TotalCharges blanks are NaN)
    data = data.dropna()
 
    # Drop identifier column — it carries no predictive signal and would
    # otherwise be one-hot encoded into ~7000 useless columns
    data = data.drop(columns=["customerID"], errors="ignore")
 
    # Separate features and target variable
    X = data.drop("Churn", axis=1)
    y = data["Churn"].apply(lambda x: 1 if x == "Yes" else 0)  # Convert target to binary
 
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
 
    # Preprocessing steps — left UNFITTED here on purpose (see note below)
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
 
    # Stratify the split: Churn is imbalanced (~73.5% / 26.5%), so
    # stratifying keeps that ratio consistent in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor