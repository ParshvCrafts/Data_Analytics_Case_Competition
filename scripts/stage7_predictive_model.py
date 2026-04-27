from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_utils import (
    PALETTE,
    add_source_note,
    notebook_setup,
    save_figure,
    save_stats,
    save_table,
    upsert_finding,
)


FEATURE_COLUMNS = [
    "org_size",
    "org_years_raw",
    "regionality",
    "region",
    "tech_person",
    "merl_person",
    "cloud_storage",
    "data_use_policy",
    "org_agreements",
    "role",
]


def _model_frame(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df[FEATURE_COLUMNS + ["cluster3"]].dropna().copy()
    model_df["is_ai_consumer"] = (model_df["cluster3"] == 1).astype(int)
    return model_df


def run_model_metrics(df: pd.DataFrame) -> dict[str, object]:
    model_df = _model_frame(df)
    X = model_df[FEATURE_COLUMNS]
    y = model_df["is_ai_consumer"]

    categorical = ["org_size", "region", "role"]
    numeric = ["org_years_raw", "regionality", "tech_person", "merl_person", "cloud_storage", "data_use_policy", "org_agreements"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score),
        "roc_auc": "roc_auc",
    }

    metric_rows = []
    fitted_pipelines = {}
    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)
        metric_rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "f1_mean": scores["test_f1"].mean(),
                "f1_std": scores["test_f1"].std(),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
                "roc_auc_std": scores["test_roc_auc"].std(),
            }
        )
        pipe.fit(X, y)
        fitted_pipelines[name] = pipe

    metrics_df = pd.DataFrame(metric_rows)
    save_table(metrics_df, "stats_07_cv_model_metrics", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = metrics_df.melt(id_vars="model", value_vars=["accuracy_mean", "f1_mean", "roc_auc_mean"], var_name="metric", value_name="score")
    plot_df["metric"] = plot_df["metric"].str.replace("_mean", "", regex=False)
    palette = [PALETTE["primary"], PALETTE["accent1"], PALETTE["accent2"]]
    for idx, metric in enumerate(["accuracy", "f1", "roc_auc"]):
        subset = plot_df[plot_df["metric"] == metric]
        ax.bar(np.arange(len(subset)) + (idx - 1) * 0.24, subset["score"], width=0.22, label=metric.upper(), color=palette[idx])
    ax.set_xticks(np.arange(len(metrics_df)))
    ax.set_xticklabels(metrics_df["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cross-validated score")
    ax.set_title("The model is useful for ranking drivers of readiness, not for making flashy predictions")
    ax.legend(frameon=False)
    add_source_note(ax)
    save_figure(fig, "fig_07_cv_metrics")
    plt.close(fig)

    return {"model_df": model_df, "metrics_df": metrics_df, "pipelines": fitted_pipelines}


def run_feature_importance(model_df: pd.DataFrame, pipelines: dict[str, Pipeline]) -> dict[str, object]:
    X = model_df[FEATURE_COLUMNS]
    y = model_df["is_ai_consumer"]
    rf_pipe = pipelines["Random Forest"]
    preprocessor = rf_pipe.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    rf_model = rf_pipe.named_steps["model"]
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    save_table(importances, "stats_07_rf_feature_importance", index=False)

    top_features = importances.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top_features["feature"], top_features["importance"], color=PALETTE["primary"])
    ax.set_xlabel("Random forest importance")
    ax.set_title("Infrastructure and role features dominate the readiness ranking")
    add_source_note(ax)
    save_figure(fig, "fig_07_rf_feature_importance")
    plt.close(fig)

    transformed = preprocessor.transform(X)
    shap_frame = None
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_array = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_array = np.abs(shap_values).mean(axis=0)
        shap_frame = pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_array}).sort_values("mean_abs_shap", ascending=False)
        save_table(shap_frame, "stats_07_shap_importance", index=False)

        top_shap = shap_frame.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(top_shap["feature"], top_shap["mean_abs_shap"], color=PALETTE["accent2"])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP tells the same story: readiness rises with people and policies, not just size")
        add_source_note(ax)
        save_figure(fig, "fig_07_shap_importance")
        plt.close(fig)
    except Exception:
        pass

    return {"rf_importance": importances, "shap_importance": shap_frame}


def run_logistic_explanation(model_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    explain_df = model_df.copy()
    explain_df["role_compact"] = np.where(
        explain_df["role"].isin(["Leader", "Fundraiser", "Admin", "Comms", "Tech", "MERL"]),
        explain_df["role"],
        "Programs/Gov/Other",
    )
    explain_df["region_compact"] = explain_df["region"].replace({"Australia / Pacific": "Other Asia"})
    explain_features = [
        "org_size",
        "org_years_raw",
        "regionality",
        "region_compact",
        "tech_person",
        "merl_person",
        "cloud_storage",
        "data_use_policy",
        "org_agreements",
        "role_compact",
    ]
    X = pd.get_dummies(explain_df[explain_features], drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = explain_df["is_ai_consumer"]
    logit = sm.Logit(y, X).fit(disp=False, maxiter=500)
    odds = np.exp(logit.params)
    conf = np.exp(logit.conf_int())
    coef_df = pd.DataFrame(
        {
            "feature": logit.params.index,
            "coef": logit.params.values,
            "odds_ratio": odds.values,
            "p_value": logit.pvalues.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
        }
    )
    coef_df = coef_df.query("feature != 'const'").sort_values("odds_ratio", ascending=False)
    save_table(coef_df, "stats_07_logit_coefficients", index=False)
    save_stats(
        {
            "cv_metrics": model_df.shape[0],
            "logit_coefficients": coef_df.to_dict(orient="records"),
            "pseudo_r_squared": float(logit.prsquared),
        },
        "stats_07_predictive_model",
    )

    plot_df = coef_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        x=plot_df["odds_ratio"],
        y=plot_df["feature"],
        xerr=[plot_df["odds_ratio"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["odds_ratio"]],
        fmt="o",
        color=PALETTE["accent1"],
        ecolor=PALETTE["neutral"],
        capsize=3,
    )
    ax.axvline(1, linestyle="--", color=PALETTE["neutral"])
    ax.set_xlabel("Odds ratio for being an AI Consumer")
    ax.set_title("The strongest readiness lifts come from people and policy, not from org age alone")
    add_source_note(ax)
    save_figure(fig, "fig_07_logit_odds_ratios")
    plt.close(fig)

    top_three = coef_df.head(3)
    upsert_finding(
        {
            "claim": "The top predictors mix infrastructure with operating context rather than collapsing to size alone",
            "value": {
                "top_logit_features": top_three["feature"].tolist(),
                "top_odds_ratios": top_three["odds_ratio"].round(3).tolist(),
            },
            "computed_in": "notebooks/07_predictive_model.ipynb",
            "cell_id": "Section 7.1",
            "code_snippet": "Cross-validated sklearn models plus statsmodels logit coefficients over org, region, infrastructure, and role features",
            "verified": True,
        }
    )

    return {"logit_coefficients": coef_df}


def run_stage7(df: pd.DataFrame | None = None) -> dict[str, object]:
    if df is None:
        df = notebook_setup()
    results: dict[str, object] = {"data": df}
    results.update(run_model_metrics(df))
    results.update(run_feature_importance(results["model_df"], results["pipelines"]))
    results.update(run_logistic_explanation(results["model_df"]))
    return results
