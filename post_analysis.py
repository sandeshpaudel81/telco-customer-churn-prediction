"""
Post training evaluation
1. run_feature_ablation()  - With or without the engineered features, how much do they help?
2. run_threshold_tuning()  - choose decision threshold and compare against SMOTE
3. run_shap_analysis()     - explain the chosen XGBoost model with SHAP
4. run_feature_selection() - does removing redundant / weak features help?
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

ENGINEERED_FEATURES = [
    "TenureGroup",
    "NumServices",
    "AvgMonthlyCharges",
    "ChargeIncrease",
    "AutoPayment",
    "NumOfProtection",
    "NewMonthToMonth",
    "FiberNoSupport",
]

# Engineered features that duplicate information already in the original
# columns (AutoPayment ~ PaymentMethod, NewMonthToMonth ~ Contract + tenure,
# TenureGroup ~ tenure)
REDUNDANT_FEATURES = ["AutoPayment", "NewMonthToMonth", "TenureGroup"]

def _dtype_preprocessor():
    """Scaler + one-hot encoder that picks columns by dtype, so the same
    preprocessor works whether or not the engineered columns are present."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             make_column_selector(dtype_exclude=np.number)),
        ]
    )

# 1. Feature ablation
def run_feature_ablation(trained_models, X_train, y_train, plots_dir, results_dir,
                         engineered_cols=ENGINEERED_FEATURES,
                         n_splits=5, n_repeats=3, random_state=42):
    """
    For every tuned model, comparing cross-validated performance on:
      - Original features   (engineered columns dropped)
      - With engineered     (all columns)
    """
    results_dir = Path(results_dir)
    engineered_cols = [c for c in engineered_cols if c in X_train.columns]
    feature_sets = {
        "Original features": X_train.drop(columns=engineered_cols),
        "With engineered": X_train,
    }
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=random_state)
    scoring = ["f1", "roc_auc", "recall", "precision"]

    rows = []
    for key, entry in trained_models.items():
        print(f"\nAblation: {key} ...")
        for set_name, X in feature_sets.items():
            pipe = clone(entry["search"].best_estimator_)
            pipe.set_params(preprocessor=_dtype_preprocessor())
            scores = cross_validate(pipe, X, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            row = {"Model": key, "Base Model": entry["base_model"],
                   "Balance": entry["balance"], "Feature Set": set_name,
                   "Num Columns": X.shape[1]}
            for m in scoring:
                row[f"{m} mean"] = scores[f"test_{m}"].mean()
                row[f"{m} std"] = scores[f"test_{m}"].std()
            rows.append(row)

    ablation_df = pd.DataFrame(rows)

    # Side-by-side summary: change in F1 / AUC from adding the features
    wide = ablation_df.pivot(index="Model", columns="Feature Set",
                             values=["f1 mean", "roc_auc mean"])
    summary = pd.DataFrame({
        "F1 (original)": wide[("f1 mean", "Original features")],
        "F1 (engineered)": wide[("f1 mean", "With engineered")],
        "F1 change": wide[("f1 mean", "With engineered")] - wide[("f1 mean", "Original features")],
        "AUC (original)": wide[("roc_auc mean", "Original features")],
        "AUC (engineered)": wide[("roc_auc mean", "With engineered")],
        "AUC change": wide[("roc_auc mean", "With engineered")] - wide[("roc_auc mean", "Original features")],
    }).sort_values("F1 change", ascending=False)

    print("\n=== Feature ablation (repeated 5-fold CV on training set) ===")
    print(summary.round(4).to_string())
    print("\nTypical F1 standard deviation across folds: "
          f"{ablation_df['f1 std'].mean():.4f} "
          "(changes smaller than this are within noise)")

    ablation_df.to_csv(results_dir / "feature_ablation_results.csv", index=False)
    summary.to_csv(results_dir / "feature_ablation_summary.csv")
    _plot_ablation(ablation_df, plots_dir / "feature_ablation.png")
    return ablation_df, summary


def _plot_ablation(ablation_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in [(axes[0], "f1", "F1-Score"), (axes[1], "roc_auc", "ROC-AUC")]:
        means = ablation_df.pivot(index="Model", columns="Feature Set", values=f"{metric} mean")
        stds = ablation_df.pivot(index="Model", columns="Feature Set", values=f"{metric} std")
        means = means[["Original features", "With engineered"]]
        stds = stds[["Original features", "With engineered"]]
        means.plot(kind="bar", yerr=stds, ax=ax, capsize=3)
        ax.set_title(f"{label}: original vs engineered features (CV mean ± std)")
        ax.set_ylabel(label)
        ax.set_xlabel("")
        lo = (means - stds).min().min()
        ax.set_ylim(max(0, lo - 0.05), min(1, (means + stds).max().max() + 0.03))
        ax.tick_params(axis="x", rotation=45)
        for t in ax.get_xticklabels():
            t.set_ha("right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 2. Threshold tuning
# ---------------------------------------------------------------------------
def _metrics_at(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "False Negatives": fn,
        "False Positives": fp,
    }


def _best_thresholds(y_true, y_proba, fn_cost, fp_cost):
    """Return (F1-optimal threshold, cost-optimal threshold) found on
    out-of-fold training probabilities."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s, costs = [], []
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        f1s.append(f1_score(y_true, y_pred))
        costs.append(fn * fn_cost + fp * fp_cost)
    return grid[int(np.argmax(f1s))], grid[int(np.argmin(costs))]


def run_threshold_tuning(trained_models, X_train, y_train, X_test, y_test, results_dir, plots_dir,
                         balance="Normal", fn_cost=5, fp_cost=1, random_state=42):
    """
    Alternative to SMOTE: keep the model trained on the original (imbalanced)
    data but move the decision threshold away from 0.5.

    The threshold is chosen ONLY from out-of-fold predicted probabilities on
    the training set (5-fold cross_val_predict), then applied once to the
    test set - so the test set is never used to pick it.

    Two thresholds are reported:
      - F1-optimal:   maximises F1 (same objective as the grid search)
      - Cost-optimal: minimises fn_cost*FN + fp_cost*FP. The default 5:1
        ratio says missing a churner costs five times more than sending a
        retention offer to a customer who would have stayed. Change it to
        match your own business assumption.
    """
    results_dir = Path(results_dir)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []
    curves = {}

    for key, entry in trained_models.items():
        if entry["balance"] != balance:
            continue
        name = entry["base_model"]
        model = entry["search"].best_estimator_
        print(f"\nThreshold tuning: {key} ...")

        oof_proba = cross_val_predict(clone(model), X_train, y_train, cv=cv,
                                      method="predict_proba", n_jobs=-1)[:, 1]
        t_f1, t_cost = _best_thresholds(y_train, oof_proba, fn_cost, fp_cost)
        test_proba = model.predict_proba(X_test)[:, 1]

        for label, t in [("Default (0.50)", 0.5), ("F1-optimal", t_f1), ("Cost-optimal", t_cost)]:
            m = _metrics_at(y_test, test_proba, t)
            m.update({"Model": name, "Strategy": label,
                      "Business Cost": m["False Negatives"] * fn_cost + m["False Positives"] * fp_cost})
            rows.append(m)
        curves[name] = (oof_proba, t_f1, t_cost)

    # SMOTE rows (default 0.5) for direct comparison
    for key, entry in trained_models.items():
        if entry["balance"] != "SMOTE":
            continue
        test_proba = entry["search"].best_estimator_.predict_proba(X_test)[:, 1]
        m = _metrics_at(y_test, test_proba, 0.5)
        m.update({"Model": entry["base_model"], "Strategy": "SMOTE (0.50)",
                  "Business Cost": m["False Negatives"] * fn_cost + m["False Positives"] * fp_cost})
        rows.append(m)

    cols = ["Model", "Strategy", "Threshold", "Accuracy", "Precision", "Recall",
            "F1-Score", "False Negatives", "False Positives", "Business Cost"]
    order = {"Default (0.50)": 0, "SMOTE (0.50)": 1, "F1-optimal": 2, "Cost-optimal": 3}
    threshold_df = (pd.DataFrame(rows)[cols]
                    .assign(_o=lambda d: d["Strategy"].map(order))
                    .sort_values(["Model", "_o"]).drop(columns="_o")
                    .reset_index(drop=True))

    print(f"\n=== Threshold tuning vs SMOTE (test set, cost = {fn_cost}*FN + {fp_cost}*FP) ===")
    print(threshold_df.round(3).to_string(index=False))

    threshold_df.to_csv(results_dir / "threshold_tuning_results.csv", index=False)
    _plot_threshold_curves(y_train, curves, plots_dir / "threshold_curves.png")
    _plot_strategy_comparison(threshold_df, plots_dir / "threshold_vs_smote.png")
    return threshold_df


def _plot_threshold_curves(y_train, curves, output_path):
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, (proba, t_f1, t_cost)) in zip(axes, curves.items()):
        precision, recall, thresholds = precision_recall_curve(y_train, proba)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        ax.plot(thresholds, precision[:-1], label="Precision")
        ax.plot(thresholds, recall[:-1], label="Recall")
        ax.plot(thresholds, f1[:-1], label="F1", linewidth=2)
        ax.axvline(0.5, color="grey", linestyle=":", label="Default 0.50")
        ax.axvline(t_f1, color="black", linestyle="--", label=f"F1-optimal {t_f1:.2f}")
        ax.axvline(t_cost, color="red", linestyle="--", label=f"Cost-optimal {t_cost:.2f}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Decision threshold")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Score (out-of-fold, training set)")
    axes[-1].legend(fontsize=7, loc="lower left")
    plt.suptitle("Precision / Recall / F1 vs decision threshold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_strategy_comparison(threshold_df, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, ["Recall", "F1-Score", "Business Cost"]):
        pivot = threshold_df.pivot(index="Model", columns="Strategy", values=metric)
        pivot = pivot[[c for c in ["Default (0.50)", "SMOTE (0.50)", "F1-optimal", "Cost-optimal"]
                       if c in pivot.columns]]
        pivot.plot(kind="bar", ax=ax, legend=(metric == "Business Cost"))
        ax.set_title(f"{metric} on test set")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    axes[-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 3. SHAP explainability
# ---------------------------------------------------------------------------
def _clean_name(name):
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    name = name.split("__", 1)[-1]
    for col in ["Contract", "PaymentMethod", "InternetService", "TenureGroup",
                "OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection",
                "StreamingTV", "StreamingMovies", "MultipleLines", "PaperlessBilling",
                "PhoneService", "Partner", "Dependents", "SeniorCitizen", "gender",
                "AutoPayment", "NewMonthToMonth", "FiberNoSupport"]:
        if name.startswith(col + "_"):
            return f"{col}: {name[len(col) + 1:]}"
    return name


def run_shap_analysis(trained_models, X_test, y_test, results_dir, plots_dir,
                      model_key="XGBoost (SMOTE)", max_display=15):
    """
    SHAP (SHapley Additive exPlanations) for the chosen XGBoost model.

    SHAP values are computed with XGBoost's built-in exact TreeSHAP
    (pred_contribs=True); the shap package is only used for plotting.

    Saves:
      shap_summary.png    - beeswarm: every customer's SHAP value per feature
      shap_importance.png - mean |SHAP| per feature (global importance)
      shap_waterfall.png  - explanation of a single high-risk churner
    """
    import xgboost as xgb

    results_dir = Path(results_dir)
    pipe = trained_models[model_key]["search"].best_estimator_
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["classifier"]

    X_enc = preprocessor.transform(X_test)
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()
    feature_names = [_clean_name(n) for n in preprocessor.get_feature_names_out()]
    X_df = pd.DataFrame(X_enc, columns=feature_names, index=X_test.index)

    contribs = model.get_booster().predict(xgb.DMatrix(X_df.values), pred_contribs=True)
    shap_values, base_value = contribs[:, :-1], contribs[0, -1]

    importance = (pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
                  .sort_values(ascending=False))
    print(f"\n=== Top {max_display} features by mean |SHAP| ({model_key}) ===")
    print(importance.head(max_display).round(4).to_string())
    importance.to_csv(results_dir / "shap_importance.csv", header=["mean_abs_shap"])

    try:
        import shap
    except ImportError:
        shap = None
        print("shap package not installed - saving importance bar chart only "
              "(pip install shap for the beeswarm and waterfall plots).")

    # Start from a clean slate - figures left open elsewhere would otherwise
    # be drawn underneath the SHAP plots
    plt.close("all")

    # Global importance bar chart (matplotlib only)
    top = importance.head(max_display)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top.values)
    plt.xlabel("Mean |SHAP value| (impact on log-odds of churn)")
    plt.title(f"Global feature importance — {model_key}")
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_importance.png", dpi=150)
    plt.close()

    if shap is not None:
        plt.figure()
        shap.summary_plot(shap_values, X_df, max_display=max_display, show=False)
        plt.title(f"SHAP summary — {model_key}")
        plt.tight_layout()
        plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Explain the actual churner the model is most confident about
        y_arr = np.asarray(y_test)
        churner_idx = np.where(y_arr == 1)[0]
        i = churner_idx[np.argmax(shap_values[churner_idx].sum(axis=1))]
        # Show numeric features in their original units (e.g. tenure in
        # months) rather than the standardised values the model sees
        display_values = X_df.iloc[i].copy()
        for col in display_values.index:
            if col in X_test.columns:
                display_values[col] = round(float(X_test.iloc[i][col]), 2)
        explanation = shap.Explanation(values=shap_values[i], base_values=base_value,
                                       data=display_values.values,
                                       feature_names=feature_names)
        plt.figure()
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.title("Why the model flags this customer as a churner")
        plt.tight_layout()
        plt.savefig(plots_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Waterfall plot explains test-set customer at position {i} "
              f"(index {X_test.index[i]}).")

    return importance

# ---------------------------------------------------------------------------
# 4. Feature selection
# ---------------------------------------------------------------------------
def _selector():
    """Keep the encoded features whose Random Forest importance is above the
    median, i.e. roughly the top half."""
    return SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        threshold="median",
    )


def run_feature_selection(trained_models, X_train, y_train, results_dir,
                          redundant_cols=REDUNDANT_FEATURES,
                          n_splits=5, n_repeats=3, random_state=42):
    """
    For every tuned model, compare cross-validated performance on:
      - All features
      - Redundant removed  (hand-picked redundant engineered columns dropped)
      - SelectFromModel    (selector step added to the pipeline)

    The selector sits inside the pipeline, so it is refit on each CV training
    fold and never sees the validation fold or the test set.
    """
    results_dir = Path(results_dir)
    redundant_cols = [c for c in redundant_cols if c in X_train.columns]
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=random_state)
    scoring = ["f1", "roc_auc"]

    rows = []
    for key, entry in trained_models.items():
        print(f"\nFeature selection: {key} ...")
        base = clone(entry["search"].best_estimator_)
        base.set_params(preprocessor=_dtype_preprocessor())

        # preprocessor -> selector -> (sampler) -> classifier
        steps = list(base.steps)
        with_selector = ImbPipeline(steps[:1] + [("selector", _selector())] + steps[1:])

        variants = {
            "All features": (base, X_train),
            "Redundant removed": (base, X_train.drop(columns=redundant_cols)),
            "SelectFromModel": (with_selector, X_train),
        }
        for set_name, (pipe, X) in variants.items():
            scores = cross_validate(pipe, X, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            row = {"Model": key, "Feature Set": set_name}
            for m in scoring:
                row[f"{m} mean"] = scores[f"test_{m}"].mean()
                row[f"{m} std"] = scores[f"test_{m}"].std()
            rows.append(row)

    selection_df = pd.DataFrame(rows)
    print("\n=== Feature selection (repeated 5-fold CV on training set) ===")
    print(selection_df.pivot(index="Model", columns="Feature Set", values="f1 mean")
          [["All features", "Redundant removed", "SelectFromModel"]].round(4).to_string())
    selection_df.to_csv(results_dir / "feature_selection_results.csv", index=False)

    # Which features does the selector keep when fit on the full training set?
    preprocessor = _dtype_preprocessor().fit(X_train)
    X_enc = preprocessor.transform(X_train)
    selector = _selector().fit(X_enc, y_train)
    names = [_clean_name(n) for n in preprocessor.get_feature_names_out()]
    selected = pd.DataFrame({
        "Feature": names,
        "Importance": selector.estimator_.feature_importances_,
        "Selected": selector.get_support(),
    }).sort_values("Importance", ascending=False)
    print(f"\nSelectFromModel keeps {selected['Selected'].sum()} of {len(names)} "
          "encoded features")
    selected.to_csv(results_dir / "selected_features.csv", index=False)

    return selection_df, selected
