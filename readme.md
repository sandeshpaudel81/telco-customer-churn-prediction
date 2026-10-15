# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will churn (leave the company) using the IBM Telco Customer Churn dataset. The pipeline covers exploratory data analysis, feature engineering, model training with hyperparameter tuning, class-imbalance handling (SMOTE), threshold tuning and model explainability (SHAP).

## Dataset

- **File:** `data/Telco-Customer-Churn.csv`
- **Source:** IBM Telco Customer Churn dataset (Kaggle)
- **Size:** 7,043 customers × 21 columns
- **Target:** `Churn` (Yes / No), imbalanced at roughly 26% churners
- **Features:** customer demographics (gender, senior citizen, partner, dependents), account details (tenure, contract, payment method, paperless billing, monthly and total charges) and subscribed services (phone, internet, online security, tech support, streaming, etc.)

## Project Structure

```
.
├── data/
│   └── Telco-Customer-Churn.csv   # Raw dataset
├── main.py                        # Entry point, runs the full pipeline
├── utils.py                       # Data loading helper
├── eda.py                         # Exploratory data analysis and plots
├── feature_engineering.py         # Creates new features
├── preprocessing.py               # Cleaning, encoding, scaling, train/test split
├── models.py                      # Model definitions, GridSearchCV, evaluation
├── post_analysis.py               # Feature ablation, feature selection, threshold tuning, SHAP
├── plots/                         # Generated on run
│   ├── eda/                       # EDA charts
│   └── models/                    # Model evaluation charts
├── results/                       # Generated on run, CSV metrics
└── models/                        # Generated on run
```

`plots/`, `results/` and `models/` are git-ignored and get created automatically when you run `main.py`.

## Analysis Pipeline

1. **Load data:** read the CSV (`utils.py`).
2. **EDA:** churn distribution, churn rate by categorical features, numeric distributions, box plots and a correlation heatmap (`eda.py`).
3. **Feature engineering:** adds `TenureGroup`, `NumServices`, `AvgMonthlyCharges`, `ChargeIncrease`, `AutoPayment`, `NumOfProtection`, `NewMonthToMonth` and `FiberNoSupport` (`feature_engineering.py`).
4. **Preprocessing:** converts `TotalCharges` to numeric, drops missing rows and `customerID`, then does a stratified 80/20 train/test split. Numeric columns are scaled with `StandardScaler` and categorical ones encoded with `OneHotEncoder` inside the pipeline, so nothing leaks from the test set (`preprocessing.py`).
5. **Model training:** each model is tuned with 5-fold `GridSearchCV` (scoring = F1), once on the original data and once with SMOTE oversampling (`models.py`).
6. **Evaluation:** test-set metrics, confusion matrices, ROC curves and a Normal vs SMOTE comparison.
7. **Post-training analysis** (`post_analysis.py`):
   - **Feature ablation:** performance with and without the engineered features
   - **Feature selection:** whether dropping redundant or weak features helps
   - **Threshold tuning:** F1-optimal and cost-optimal thresholds (cost = 5×FN + 1×FP), compared against SMOTE
   - **SHAP analysis:** explains the XGBoost (SMOTE) model

## Models Used

Each model is trained in two versions, **Normal** and **SMOTE**, which gives 10 models in total:

- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Artificial Neural Network (scikit-learn `MLPClassifier`)
- XGBoost

## Metrics and Outputs

**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, False Negatives, False Positives and Business Cost (threshold tuning).

**Saved results (`results/`):**

| File | Contents |
|------|----------|
| `model_comparison_results.csv` | Test metrics and best hyperparameters for all 10 models |
| `feature_ablation_results.csv` / `feature_ablation_summary.csv` | CV scores with and without engineered features |
| `feature_selection_results.csv` / `selected_features.csv` | Feature selection comparison and the kept features |
| `threshold_tuning_results.csv` | Default vs tuned thresholds vs SMOTE, with business cost |
| `shap_importance.csv` | Mean absolute SHAP value per feature |

**Saved plots:**
- `plots/eda/`: churn distribution, churn rate by feature, histograms, box plots, correlation heatmap
- `plots/models/`: `confusion_matrices.png`, `roc_curves.png`, `normal_vs_smote.png`, `feature_ablation.png`, `threshold_curves.png`, `threshold_vs_smote.png`, `shap_importance.png`, `shap_summary.png`, `shap_waterfall.png`

## How to Run

**Prerequisites:** Python 3.10+ and Git.

```bash
# 1. Clone the repository
git clone https://github.com/sandeshpaudel81/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

The full run tunes 10 models with GridSearchCV, so it can take several minutes. When it finishes, the plots are in `plots/`, the CSV metrics are in `results/`, and the model comparison table is printed in the terminal.
