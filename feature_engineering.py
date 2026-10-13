import numpy as np
import pandas as pd

# Add-on services counted towards a customer's total number of services
SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

PROTECTION_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport"
]


def feature_engineering(data):
    """
    Create new features from the raw dataset.

    All features are derived row by row (no statistics are learned from the
    data), so this can safely run before the train/test split without leaking
    information from the test set.

    Args:
        data (pd.DataFrame): The input dataset.
    Returns:
        pd.DataFrame: A copy of the dataset with the engineered features added.
    """

    data = data.copy()

    # TotalCharges is stored as text with blanks for new customers
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Tenure group — customers in their first year churn far more often.
    # Stored as strings so the preprocessor treats it as categorical.
    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"],
    ).astype(str)

    # Number of services the customer subscribes to
    data["NumServices"] = (data[SERVICE_COLS] == "Yes").sum(axis=1)
    data["NumServices"] += (data["InternetService"] != "No").astype("int64")

    # Average monthly spend over the customer's lifetime
    # (falls back to MonthlyCharges for customers with zero tenure)
    data["AvgMonthlyCharges"] = np.where(
        data["tenure"] > 0,
        data["TotalCharges"] / data["tenure"].replace(0, np.nan),
        data["MonthlyCharges"],
    )

    # Difference between current and average monthly charge — a positive
    # value means the customer's bill has gone up over time
    data["ChargeIncrease"] = data["MonthlyCharges"] - data["AvgMonthlyCharges"]

    # Automatic payment methods are associated with lower churn
    data["AutoPayment"] = np.where(
        data["PaymentMethod"].str.contains("automatic"), "Yes", "No"
    )

    # Number of protection add-on services
    data["NumOfProtection"] = (data[PROTECTION_SERVICE_COLS] == 'Yes').sum(axis=1)

    # Interaction: new customer on a month-to-month contract
    data["NewMonthToMonth"] = np.where(
        (data["Contract"] == "Month-to-month") & (data["tenure"] <= 12), "Yes", "No"
    )

    # Interaction: fibre customers without tech support
    data["FiberNoSupport"] = np.where(
        (data["InternetService"] == "Fiber optic") & (data["TechSupport"] == "No"), "Yes", "No"
    )

    print("\nEngineered features added:")
    print(["TenureGroup", "NumServices", "AvgMonthlyCharges", "ChargeIncrease", "AutoPayment",
           "NumOfProtection", "NewMonthToMonth", "FiberNoSupport"])

    return data
