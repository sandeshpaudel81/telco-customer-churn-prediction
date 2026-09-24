import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data, output_dir='plots'):
    """
    Perform basic data exploration and visualization.

    Args:
        data (pd.DataFrame): The dataset to explore.
    """
    # Work on a copy so the original dataframe is not changed
    eda_data = data.copy()

    # ---------------------------------------------------------
    # 1. Basic Dataset Information
    # ---------------------------------------------------------

    print("\nFirst 5 rows of the dataset:")
    print(eda_data.head())

    print("\nLast 5 rows of the dataset:")
    print(eda_data.tail())

    print("\nDataset Shape:")
    print(eda_data.shape)

    print("\nData Overview:")
    eda_data.info()

    print("\nStatistical Summary:")
    print(eda_data.describe(include='all').T)

    print("\nList of Columns:")
    print(eda_data.columns.tolist())

    # ---------------------------------------------------------
    # 2. Data Quality Checks
    # ---------------------------------------------------------

    print("\nMissing Values:")
    print(eda_data.isnull().sum())

    missing_summary = pd.DataFrame({
        'Missing Count': eda_data.isnull().sum(),
        'Missing %': (eda_data.isnull().mean() * 100).round(2)
    })

    print("\nMissing Value Summary:")
    print(missing_summary[missing_summary['Missing Count'] > 0])

    print("\nDuplicate Rows:")
    print(eda_data.duplicated().sum())

    print("\nUnique Values per Column:")
    for column in eda_data.columns:
        print(f"{column}: {eda_data[column].nunique()} unique values")

    # ---------------------------------------------------------
    # 3. Data Type Conversion for EDA
    # ---------------------------------------------------------

    # TotalCharges is numeric in meaning but may be stored as text
    if 'TotalCharges' in eda_data.columns:
        eda_data['TotalCharges'] = pd.to_numeric(
            eda_data['TotalCharges'],
            errors='coerce'
        )

    # Convert SeniorCitizen to readable categories for EDA
    if 'SeniorCitizen' in eda_data.columns:
        eda_data['SeniorCitizen'] = eda_data['SeniorCitizen'].map({
            0: 'No',
            1: 'Yes'
        })

    # Convert Churn to binary values for calculations
    if 'Churn' in eda_data.columns:
        eda_data['Churn'] = eda_data['Churn'].map({
            'No': 0,
            'Yes': 1
        })

    print("\nData Types after Conversion:")
    print(eda_data.dtypes)

    # ---------------------------------------------------------
    # 4. Target / Class Distribution
    # ---------------------------------------------------------

    if 'Churn' in eda_data.columns:

        print("\nClass Distribution:")
        print(eda_data['Churn'].value_counts())

        print("\nClass Distribution (%):")
        print((eda_data['Churn'].value_counts(normalize=True) * 100).round(2))

        plt.figure(figsize=(5, 4))
        sns.countplot(x='Churn', data=eda_data)
        plt.title('Customer Churn Distribution')
        plt.xlabel('Churn (0 = No, 1 = Yes)')
        plt.ylabel('Number of Customers')
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/churn_distribution.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # ---------------------------------------------------------
    # 5. Numerical Feature Analysis
    # ---------------------------------------------------------

    numeric_cols = eda_data.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')

    print("\nNumerical Features:")
    print(numeric_cols)

    if numeric_cols:
        eda_data[numeric_cols].hist(
            figsize=(10, 7),
            bins=20
        )
        plt.suptitle('Distribution of Numerical Features')
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/numeric_histograms.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # ---------------------------------------------------------
    # 6. Numerical Features vs Churn
    # ---------------------------------------------------------

    for column in numeric_cols:

        if 'Churn' in eda_data.columns:

            plt.figure(figsize=(5, 4))
            sns.boxplot(
                x='Churn',
                y=column,
                data=eda_data
            )
            plt.title(f'{column} by Churn')
            plt.xlabel('Churn (0 = No, 1 = Yes)')
            plt.ylabel(column)
            plt.tight_layout()
            plt.savefig(
                f'{output_dir}/boxplot_{column}_by_churn.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

    # ---------------------------------------------------------
    # 7. Correlation Analysis
    # ---------------------------------------------------------

    correlation_cols = numeric_cols.copy()

    if 'Churn' in eda_data.columns:
        correlation_cols.append('Churn')

    if len(correlation_cols) > 1:

        correlation_matrix = eda_data[correlation_cols].corr()

        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(2))

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f'
        )
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/correlation_heatmap.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # ---------------------------------------------------------
    # 8. Churn Rate by Important Categorical Features
    # ---------------------------------------------------------

    categorical_cols = [
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'Contract',
        'PaymentMethod',
        'InternetService',
        'PaperlessBilling',
        'TechSupport',
        'OnlineSecurity'
    ]

    for column in categorical_cols:

        if column in eda_data.columns and 'Churn' in eda_data.columns:

            # Churn count plot
            plt.figure(figsize=(7, 4))
            sns.countplot(
                x=column,
                hue='Churn',
                data=eda_data
            )
            plt.title(f'Churn Distribution by {column}')
            plt.xlabel(column)
            plt.ylabel('Number of Customers')
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(
                f'{output_dir}/churn_by_{column}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

            # Churn rate by category
            churn_rate = (
                eda_data.groupby(column)['Churn']
                .mean()
                .sort_values(ascending=False)
                * 100
            )

            print(f"\nChurn Rate by {column} (%):")
            print(churn_rate.round(2))

            plt.figure(figsize=(7, 4))
            churn_rate.plot(kind='bar')
            plt.title(f'Churn Rate by {column}')
            plt.xlabel(column)
            plt.ylabel('Churn Rate (%)')
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(
                f'{output_dir}/churn_rate_by_{column}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

    # ---------------------------------------------------------
    # 9. Tenure Analysis
    # ---------------------------------------------------------

    if 'tenure' in eda_data.columns and 'Churn' in eda_data.columns:

        eda_data['TenureGroup'] = pd.cut(
            eda_data['tenure'],
            bins=[-1, 12, 24, 36, 48, 60, 72],
            labels=[
                '0-12',
                '13-24',
                '25-36',
                '37-48',
                '49-60',
                '61-72'
            ]
        )

        tenure_churn = (
            eda_data.groupby(
                'TenureGroup',
                observed=False
            )['Churn']
            .mean()
            * 100
        )

        print("\nChurn Rate by Tenure Group (%):")
        print(tenure_churn.round(2))

        plt.figure(figsize=(7, 4))
        tenure_churn.plot(kind='bar')
        plt.title('Churn Rate by Tenure Group')
        plt.xlabel('Tenure (Months)')
        plt.ylabel('Churn Rate (%)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/churn_rate_by_tenure_group.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        # Tenure distribution by churn
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x='Churn',
            y='tenure',
            data=eda_data
        )
        plt.title('Tenure Distribution by Churn')
        plt.xlabel('Churn (0 = No, 1 = Yes)')
        plt.ylabel('Tenure (Months)')
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/tenure_by_churn.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # ---------------------------------------------------------
    # 10. Monthly Charges Analysis
    # ---------------------------------------------------------

    if 'MonthlyCharges' in eda_data.columns and 'Churn' in eda_data.columns:

        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x='Churn',
            y='MonthlyCharges',
            data=eda_data
        )
        plt.title('Monthly Charges by Churn')
        plt.xlabel('Churn (0 = No, 1 = Yes)')
        plt.ylabel('Monthly Charges')
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/monthly_charges_by_churn.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        plt.figure(figsize=(7, 4))
        sns.histplot(
            data=eda_data,
            x='MonthlyCharges',
            hue='Churn',
            bins=30,
            kde=True,
            element='step'
        )
        plt.title('Monthly Charges Distribution by Churn')
        plt.xlabel('Monthly Charges')
        plt.ylabel('Number of Customers')
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/monthly_charges_distribution_by_churn.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # ---------------------------------------------------------
    # 11. Total Charges Analysis
    # ---------------------------------------------------------

    if 'TotalCharges' in eda_data.columns and 'Churn' in eda_data.columns:

        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x='Churn',
            y='TotalCharges',
            data=eda_data
        )
        plt.title('Total Charges by Churn')
        plt.xlabel('Churn (0 = No, 1 = Yes)')
        plt.ylabel('Total Charges')
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/total_charges_by_churn.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # ---------------------------------------------------------
    # 12. Service Count Feature Engineering
    # ---------------------------------------------------------

    service_columns = [
        'PhoneService',
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    available_service_columns = [
        column for column in service_columns
        if column in eda_data.columns
    ]

    if available_service_columns and 'Churn' in eda_data.columns:
        service_data = eda_data[available_service_columns].replace({
            'Yes': 1,
            'No': 0,
            'No phone service': 0,
            'No internet service': 0
        })

        # Convert service columns to numeric where possible
        for column in available_service_columns:
            service_data[column] = pd.to_numeric(
                service_data[column],
                errors='coerce'
            ).fillna(0)

        eda_data['ServiceCount'] = service_data.sum(axis=1)

        service_churn = (
            eda_data.groupby('ServiceCount')['Churn']
            .mean()
            * 100
        )

        print("\nChurn Rate by Number of Services (%):")
        print(service_churn.round(2))

        plt.figure(figsize=(7, 4))
        service_churn.plot(kind='bar')
        plt.title('Churn Rate by Number of Services')
        plt.xlabel('Number of Services')
        plt.ylabel('Churn Rate (%)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/churn_rate_by_service_count.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # ---------------------------------------------------------
    # 13. Key Feature Relationships
    # ---------------------------------------------------------

    if (
        'tenure' in eda_data.columns
        and 'MonthlyCharges' in eda_data.columns
        and 'Churn' in eda_data.columns
    ):

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=eda_data,
            x='tenure',
            y='MonthlyCharges',
            hue='Churn',
            alpha=0.5
        )
        plt.title('Tenure vs Monthly Charges by Churn')
        plt.xlabel('Tenure (Months)')
        plt.ylabel('Monthly Charges')
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/tenure_vs_monthly_charges.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # ---------------------------------------------------------
    # 14. Summary of EDA Findings
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("EDA COMPLETED")
    print("=" * 60)

    print(f"Total observations: {len(eda_data)}")
    print(f"Total features: {eda_data.shape[1]}")

    if 'Churn' in eda_data.columns:
        churn_rate = eda_data['Churn'].mean() * 100
        print(f"Overall churn rate: {churn_rate:.2f}%")

    print(f"Plots saved to: {output_dir}")
    