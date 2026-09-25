from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from preprocessing import preprocessing


# ---------------------------------------------------------------------------
# Model + hyperparameter grid 
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "classifier__max_depth": [3, 5, 7, 10, None],
            "classifier__min_samples_leaf": [1, 5, 10],
            "classifier__criterion": ["gini", "entropy"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 5],
        },
    },
    "Naive Bayes": {
        "estimator": GaussianNB(),
        "param_grid": {
            "classifier__var_smoothing": [1e-9, 1e-8, 1e-7],
        },
    },
    "ANN (MLP)": {
        "estimator": MLPClassifier(max_iter=1000, random_state=42),
        "param_grid": {
            "classifier__hidden_layer_sizes": [(32,), (64, 32), (64, 32, 16)],
            "classifier__activation": ["relu", "tanh"],
            "classifier__alpha": [0.0001, 0.001],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            eval_metric="logloss", random_state=42
        ),
        "param_grid": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
        },
    },
}

BALANCE_STRATEGIES = {
    "Normal": None,
    "SMOTE": SMOTE(random_state=42),
}

def build_pipeline(preprocessor, estimator, sampler=None):
    """Wrap the (unfitted) preprocessor and a classifier in one Pipeline so
    that scaling/encoding is refit on every CV training fold — no leakage."""
    steps = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", estimator))
    return ImbPipeline(steps)


def run_grid_search(name, config, preprocessor, sampler, X_train, y_train, cv=5, scoring="f1"):
    """Fit a GridSearchCV over one model's pipeline and return the fitted
    search object (search.best_estimator_ is ready to predict on test data)."""
    pipe = build_pipeline(preprocessor, config["estimator"], sampler=sampler)
    search = GridSearchCV(
        pipe,
        param_grid=config["param_grid"],
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    print(f"\nRunning GridSearchCV for {name} ...")
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV {scoring}: {search.best_score_:.4f}")
    return search


def evaluate_model(name, search, X_test, y_test):
    """Score the tuned model on the held-out test set across several
    metrics, and return both the metrics dict and the confusion matrix."""
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Best CV Params": search.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    cm = confusion_matrix(y_test, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    metrics["False Negatives"] = fn  # predicted "stays", actually churned
    metrics["False Positives"] = fp

    return metrics, cm, y_proba


def plot_confusion_matrices(results_cms, output_path="confusion_matrices.png"):
    n = len(results_cms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, results_cms.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(output_path)


def plot_roc_curves(y_test, proba_by_model, output_path="roc_curves.png"):
    plt.figure(figsize=(6, 5))
    for name, y_proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)


def plot_normal_vs_smote(results_df, output_path="normal_vs_smote.png"):
    """Grouped bar chart comparing Recall and F1 for each model, Normal vs
    SMOTE — the clearest single figure for making the imbalance-handling
    point in your report/slides."""
    pivot_recall = results_df.pivot(index="Base Model", columns="Balance", values="Recall")
    pivot_f1 = results_df.pivot(index="Base Model", columns="Balance", values="F1-Score")
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pivot_recall.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Recall: Normal vs SMOTE")
    axes[0].set_ylabel("Recall")
    axes[0].tick_params(axis="x", rotation=30)
 
    pivot_f1.plot(kind="bar", ax=axes[1])
    axes[1].set_title("F1-Score: Normal vs SMOTE")
    axes[1].set_ylabel("F1-Score")
    axes[1].tick_params(axis="x", rotation=30)
 
    plt.tight_layout()
    plt.savefig(output_path)


def train_models(X_train, y_train, preprocessor, cv=5, scoring="f1"):
    """
    Run GridSearchCV for every model in MODEL_CONFIGS.

    Args:
        X_train, y_train: training split from preprocessing().
        preprocessor: the UNFITTED ColumnTransformer from preprocessing()
            — refit inside each Pipeline on every CV fold.
        cv: number of cross-validation folds.
        scoring: metric GridSearchCV optimises for (default "f1")

    Returns:
        dict[str, GridSearchCV]: fitted search objects keyed by model
        name. Each search.best_estimator_ is a ready-to-predict Pipeline.
    """
    fitted_searches = {}
    for balance_label, sampler in BALANCE_STRATEGIES.items():
        for name, config in MODEL_CONFIGS.items():
            key = f"{name} ({balance_label})"
            search = run_grid_search(name, config, preprocessor, sampler, X_train, y_train, cv=cv, scoring=scoring)
            fitted_searches[key] = {"search": search, "base_model": name, "balance": balance_label}
    return fitted_searches



def evaluate_models(trained_models, X_test, y_test, results_dir="."):
    """
    Score every fitted model from train_models() on the held-out test set
    and save comparison plots/CSV into results_dir.

    Args:
        trained_models: dict[str, GridSearchCV] returned by train_models().
        X_test, y_test: held-out test split from preprocessing().
        results_dir: directory (str or Path) to save plots and the CSV into.

    Returns:
        pd.DataFrame: comparison table sorted by ROC-AUC (best first).
    """

    all_metrics = []
    confusion_matrices = {}
    probabilities = {}
 
    for key, entry in trained_models.items():
        metrics, cm, y_proba = evaluate_model(key, entry["search"], X_test, y_test)
        metrics["Base Model"] = entry["base_model"]
        metrics["Balance"] = entry["balance"]
        all_metrics.append(metrics)
        confusion_matrices[key] = cm
        probabilities[key] = y_proba
 
    results_df = pd.DataFrame(all_metrics).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
 
    print("\n=== Model Comparison (sorted by ROC-AUC) ===")
    print(results_df[["Model", "Base Model", "Balance", "Accuracy", "Precision", "Recall",
                       "F1-Score", "ROC-AUC", "False Negatives", "False Positives"]].to_string(index=False))
 
    plot_confusion_matrices(confusion_matrices, output_path=results_dir / "confusion_matrices.png")
    plot_roc_curves(y_test, probabilities, output_path=results_dir / "roc_curves.png")
    plot_normal_vs_smote(results_df, output_path=results_dir / "normal_vs_smote.png")
    results_df.to_csv(results_dir / "model_comparison_results.csv", index=False)
 
    return results_df