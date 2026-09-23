import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data, output_dir='plots'):
    """
    Perform basic data exploration and visualization.

    Args:
        data (pd.DataFrame): The dataset to explore.
    """
    print("\nFirst 5 rows of the dataset:")
    print(data.head())

    print("\nLast 5 rows of the dataset:")
    print(data.tail())

    print("\nData Overview:")
    print(data.info())

    print("\nStatistical Summary:")
    print(data.describe())

    print("\nList of Columns:")
    print(data.columns.tolist())

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nUnique Values per Column:")
    for column in data.columns:
        unique_values = data[column].nunique()
        print(f"{column}: {unique_values} unique values")

    print("\nClass Distribution:")
    print(data['Churn'].value_counts(normalize=True))

    # Convert 'SeniorCitizen' to categorical for better visualization
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = data['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Convert 'Churn' to categorical for better visualization
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

    # Check data types after conversion
    print("\nData Types after Conversion:")
    print(data.dtypes)

    # --- Simple visualizations ---
 
    # 1. Churn class distribution (bar chart)
    plt.figure(figsize=(5, 4))
    sns.countplot(x='Churn', data=data)
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/churn_distribution.png')
 
    # 2. Histograms for numeric columns
    numeric_cols_with_churn = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols_with_churn if col != 'Churn']  # Exclude Churn from numeric columns
    if numeric_cols:
        data[numeric_cols].hist(figsize=(10, 6), bins=20)
        plt.suptitle('Distribution of Numeric Features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/numeric_histograms.png')
 
    # 3. Boxplots of numeric columns vs Churn (helps spot outliers per class)
    for column in numeric_cols:
        if column != 'Churn':  # Avoid plotting Churn against itself
            plt.figure(figsize=(5, 4))
            sns.boxplot(x='Churn', y=column, data=data)
            plt.title(f'{column} by Churn')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/boxplot_{column}_by_churn.png')
 
    # 4. Correlation heatmap for numeric columns
    if len(numeric_cols_with_churn) > 1:
        plt.figure(figsize=(6, 5))
        sns.heatmap(data[numeric_cols_with_churn].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap (Numeric Features)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png')
 
    # 5. Churn rate by key categorical columns
    categorical_cols = ['gender', 'Partner', 'Dependents', 'Contract', 'PaymentMethod']
    for column in categorical_cols:
        if column in data.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=column, hue='Churn', data=data)
            plt.title(f'Churn by {column}')
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/churn_by_{column}.png')
    