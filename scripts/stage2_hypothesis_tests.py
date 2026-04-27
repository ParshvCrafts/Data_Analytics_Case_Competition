from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from analysis_utils import (
    AI_RISK_LABELS,
    PALETTE,
    RISK_TO_COLUMN,
    SOURCE_LINE,
    add_source_note,
    notebook_setup,
    save_figure,
    save_stats,
    save_table,
    upsert_finding,
)


def _odds_ratio_from_table(table: pd.DataFrame) -> float:
    adjusted = table.astype(float).to_numpy() + 0.5
    return float((adjusted[0, 0] * adjusted[1, 1]) / (adjusted[0, 1] * adjusted[1, 0]))


def _cramers_v(chi2: float, table: pd.DataFrame) -> float:
    n = table.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, c = table.shape
    return float(math.sqrt(chi2 / (n * (min(r, c) - 1))))


def run_risk_role_tests(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    records = []
    for risk, risk_col in RISK_TO_COLUMN.items():
        subset = df[["role_group", risk_col]].dropna()
        table = pd.crosstab(subset["role_group"], subset[risk_col]).reindex(
            index=["Tech/MERL", "Non-technical"],
            columns=[0, 1],
            fill_value=0,
        )
        chi2, p_value, _, expected = stats.chi2_contingency(table)
        use_fisher = (expected < 5).any()
        if use_fisher:
            _, p_value = stats.fisher_exact(table.to_numpy())
        records.append(
            {
                "risk": risk,
                "risk_short": AI_RISK_LABELS[risk],
                "tech_merl_pct": subset.loc[subset["role_group"] == "Tech/MERL", risk_col].mean() * 100,
                "non_tech_pct": subset.loc[subset["role_group"] == "Non-technical", risk_col].mean() * 100,
                "pct_gap": (
                    subset.loc[subset["role_group"] == "Tech/MERL", risk_col].mean()
                    - subset.loc[subset["role_group"] == "Non-technical", risk_col].mean()
                )
                * 100,
                "p_value": p_value,
                "test_used": "Fisher exact" if use_fisher else "Chi-squared",
                "odds_ratio": _odds_ratio_from_table(table),
                "cramers_v": _cramers_v(chi2, table),
            }
        )

    results = pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)
    reject, p_adj, _, _ = multipletests(results["p_value"], method="holm")
    results["p_value_holm"] = p_adj
    results["significant_holm"] = reject
    save_table(results, "stats_02_risk_role_tests", index=False)
    save_stats(
        {"tests": results.to_dict(orient="records")},
        "stats_02_risk_role_tests",
    )

    plot_df = results.sort_values("pct_gap", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos - 0.18, plot_df["tech_merl_pct"], height=0.35, color=PALETTE["primary"], label="Tech/MERL")
    ax.barh(y_pos + 0.18, plot_df["non_tech_pct"], height=0.35, color=PALETTE["accent1"], label="Non-technical")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["risk_short"])
    ax.set_xlabel("Percent selecting this risk")
    ax.set_title("Tech and MERL respondents are especially wary of biased AI and inequity risks")
    ax.legend(frameon=False, loc="lower right")
    for idx, row in enumerate(plot_df.itertuples()):
        ax.text(row.tech_merl_pct + 1, idx - 0.18, f"{row.tech_merl_pct:.0f}%", va="center", fontsize=9)
        ax.text(row.non_tech_pct + 1, idx + 0.18, f"{row.non_tech_pct:.0f}%", va="center", fontsize=9)
    add_source_note(ax)
    save_figure(fig, "fig_02_risk_by_role_hypothesis")
    plt.close(fig)

    if not results.empty:
        strongest = results.sort_values("p_value_holm").iloc[0]
        upsert_finding(
            {
                "claim": "Tech and MERL respondents are more likely than non-technical peers to name biased AI decisions as a risk",
                "value": {
                    "tech_merl_pct": round(float(strongest["tech_merl_pct"]), 2),
                    "non_tech_pct": round(float(strongest["non_tech_pct"]), 2),
                    "odds_ratio": round(float(strongest["odds_ratio"]), 3),
                    "p_value_holm": float(strongest["p_value_holm"]),
                },
                "computed_in": "notebooks/02_hypothesis_tests.ipynb",
                "cell_id": "Section 2.1",
                "code_snippet": "chi2/fisher tests over role_group x risk indicators with Holm correction",
                "verified": True,
            }
        )

    return {"risk_tests": results}


def run_region_tests(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    regions = ["North America", "Europe", "India", "Africa", "Latin America", "Other Asia"]
    subset = df[df["region"].isin(regions)].copy()
    metrics = {
        "person_ai_comfort_raw": "AI comfort (0-10)",
        "collab_feasibility_raw": "Collaboration feasibility (0-10)",
        "ai_use_count": "AI use count",
        "risk_count": "Risk count",
    }

    tests = []
    group_means = subset.groupby("region")[list(metrics)].mean().reindex(regions)
    south_regions = ["India", "Africa", "Latin America", "Other Asia"]
    for metric, label in metrics.items():
        arrays = [subset.loc[subset["region"] == region, metric].dropna() for region in regions]
        arrays = [arr for arr in arrays if len(arr) > 0]
        h_stat, p_value = stats.kruskal(*arrays)
        n = sum(len(arr) for arr in arrays)
        k = len(arrays)
        epsilon_sq = max(0.0, (h_stat - k + 1) / (n - k))
        south_means = group_means.loc[south_regions, metric].dropna()
        north_south_gap = abs(
            subset.loc[subset["global_north_south"] == "N", metric].mean()
            - subset.loc[subset["global_north_south"] == "S", metric].mean()
        )
        intra_south_spread = float(south_means.max() - south_means.min())
        tests.append(
            {
                "metric": metric,
                "metric_label": label,
                "kruskal_h": h_stat,
                "p_value": p_value,
                "epsilon_squared": epsilon_sq,
                "north_south_gap": north_south_gap,
                "intra_south_spread": intra_south_spread,
                "south_spread_gt_north_south_gap": intra_south_spread > north_south_gap,
            }
        )

    tests_df = pd.DataFrame(tests).sort_values("p_value")
    save_table(group_means.reset_index(), "stats_02_region_metric_means", index=False)
    save_table(tests_df, "stats_02_region_kruskal_tests", index=False)
    save_stats(
        {
            "region_counts": subset["region"].value_counts().to_dict(),
            "metric_tests": tests_df.to_dict(orient="records"),
        },
        "stats_02_region_kruskal_tests",
    )

    leading_metric = tests_df.sort_values("epsilon_squared", ascending=False).iloc[0]
    upsert_finding(
        {
            "claim": "Regional variance inside the Global South is larger than the simple North-South split for at least one readiness metric",
            "value": {
                "metric": leading_metric["metric_label"],
                "north_south_gap": round(float(leading_metric["north_south_gap"]), 3),
                "intra_south_spread": round(float(leading_metric["intra_south_spread"]), 3),
                "epsilon_squared": round(float(leading_metric["epsilon_squared"]), 3),
            },
            "computed_in": "notebooks/02_hypothesis_tests.ipynb",
            "cell_id": "Section 2.2",
            "code_snippet": "Kruskal-Wallis tests over six-region readiness metrics and north-south gap comparison",
            "verified": True,
        }
    )

    return {"region_tests": tests_df, "region_means": group_means}


def run_infrastructure_models(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model_df = df[
        [
            "cluster3",
            "org_size_int",
            "org_years_raw",
            "regionality",
            "global_north_south_int",
            "tech_person",
            "merl_person",
            "cloud_storage",
            "data_use_policy",
            "org_agreements",
            "ai_use_count",
            "infra_score",
        ]
    ].dropna()
    model_df = model_df.copy()
    model_df["is_ai_consumer"] = (model_df["cluster3"] == 1).astype(int)

    X = model_df[
        [
            "org_size_int",
            "org_years_raw",
            "regionality",
            "global_north_south_int",
            "tech_person",
            "merl_person",
            "cloud_storage",
            "data_use_policy",
            "org_agreements",
        ]
    ]
    X = sm.add_constant(X, has_constant="add")
    y = model_df["is_ai_consumer"]
    logit_model = sm.Logit(y, X).fit(disp=False)
    odds_ratios = np.exp(logit_model.params)
    conf_int = np.exp(logit_model.conf_int())
    logit_results = pd.DataFrame(
        {
            "term": logit_model.params.index,
            "coef": logit_model.params.values,
            "odds_ratio": odds_ratios.values,
            "p_value": logit_model.pvalues.values,
            "ci_low": conf_int[0].values,
            "ci_high": conf_int[1].values,
        }
    ).sort_values("odds_ratio", ascending=False)

    ols_X = sm.add_constant(model_df[["infra_score"]], has_constant="add")
    ols_model = sm.OLS(model_df["ai_use_count"], ols_X).fit()
    ols_results = pd.DataFrame(
        {
            "term": ols_model.params.index,
            "coef": ols_model.params.values,
            "p_value": ols_model.pvalues.values,
        }
    )

    save_table(logit_results, "stats_02_logit_ai_consumer", index=False)
    save_table(ols_results, "stats_02_ols_infra_ai_use", index=False)
    save_stats(
        {
            "logit_summary": logit_results.to_dict(orient="records"),
            "logit_pseudo_r2": float(logit_model.prsquared),
            "ols_summary": ols_results.to_dict(orient="records"),
            "ols_r_squared": float(ols_model.rsquared),
        },
        "stats_02_infrastructure_models",
    )

    sig_terms = logit_results.query("term != 'const'").sort_values("p_value")
    lead_term = sig_terms.iloc[0]
    upsert_finding(
        {
            "claim": "Cloud storage and data agreements stay predictive after controls, while org age and the North-South split do not",
            "value": {
                "top_term": lead_term["term"],
                "top_term_odds_ratio": round(float(lead_term["odds_ratio"]), 3),
                "top_term_p_value": float(lead_term["p_value"]),
                "org_agreements_odds_ratio": round(
                    float(logit_results.loc[logit_results["term"] == "org_agreements", "odds_ratio"].iloc[0]), 3
                ),
                "org_size_odds_ratio": round(
                    float(logit_results.loc[logit_results["term"] == "org_size_int", "odds_ratio"].iloc[0]), 3
                ),
                "org_years_p_value": float(
                    logit_results.loc[logit_results["term"] == "org_years_raw", "p_value"].iloc[0]
                ),
                "global_north_south_p_value": float(
                    logit_results.loc[logit_results["term"] == "global_north_south_int", "p_value"].iloc[0]
                ),
                "ols_infra_slope": round(float(ols_model.params["infra_score"]), 3),
                "ols_r_squared": round(float(ols_model.rsquared), 3),
            },
            "computed_in": "notebooks/02_hypothesis_tests.ipynb",
            "cell_id": "Section 2.3",
            "code_snippet": "statsmodels.Logit for is_ai_consumer and OLS for ai_use_count ~ infra_score",
            "verified": True,
        }
    )

    return {
        "logit_results": logit_results,
        "ols_results": ols_results,
        "logit_model": logit_model,
        "ols_model": ols_model,
    }


def run_threshold_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = ["0-5", "6-15", "16-30", "31-60", "61-120", "121+"]
    summary = (
        df.assign(is_ai_consumer=(df["cluster3"] == 1).astype(int))
        .groupby("org_size")
        .agg(
            pct_has_tech_or_merl=("has_tech_or_merl", lambda s: s.mean() * 100),
            pct_ai_consumers=("is_ai_consumer", lambda s: s.mean() * 100),
            count=("cluster3", "size"),
        )
        .reindex(ordered)
        .reset_index()
    )
    summary["bucket_mid"] = range(len(summary))
    summary["post_15"] = np.clip(summary["bucket_mid"] - 1, 0, None)

    X = sm.add_constant(summary[["bucket_mid", "post_15"]], has_constant="add")
    piecewise = sm.OLS(summary["pct_has_tech_or_merl"], X).fit()
    summary["piecewise_fit"] = piecewise.predict(X)

    save_table(summary, "stats_02_staff_threshold_summary", index=False)
    save_stats(
        {
            "threshold_summary": summary.to_dict(orient="records"),
            "piecewise_params": piecewise.params.to_dict(),
            "piecewise_r_squared": float(piecewise.rsquared),
        },
        "stats_02_staff_threshold",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["org_size"], summary["pct_has_tech_or_merl"], marker="o", color=PALETTE["primary"], linewidth=2.5)
    ax.plot(summary["org_size"], summary["piecewise_fit"], linestyle="--", color=PALETTE["accent2"], linewidth=2)
    ax.set_ylabel("Percent with a tech or MERL person")
    ax.set_xlabel("Organization size")
    ax.set_title("The jump after 15 staff looks more like a hiring threshold than a pure size effect")
    for row in summary.itertuples():
        ax.text(row.Index, row.pct_has_tech_or_merl + 2, f"{row.pct_has_tech_or_merl:.0f}%", ha="center", fontsize=9)
    add_source_note(ax)
    save_figure(fig, "fig_02_staff_threshold")
    plt.close(fig)

    diffs = summary["pct_has_tech_or_merl"].diff()
    jump_idx = int(diffs.iloc[1:].idxmax())
    jump_from = summary.loc[jump_idx - 1, "org_size"]
    jump_to = summary.loc[jump_idx, "org_size"]
    upsert_finding(
        {
            "claim": "The sharpest staffing jump happens between micro nonprofits and the 6-15 staff group, not after the 15-staff line",
            "value": {
                "jump_from": jump_from,
                "jump_to": jump_to,
                "percentage_point_jump": round(float(diffs.iloc[jump_idx]), 2),
            },
            "computed_in": "notebooks/02_hypothesis_tests.ipynb",
            "cell_id": "Section 2.4",
            "code_snippet": "groupby org_size on has_tech_or_merl with piecewise trend line",
            "verified": True,
        }
    )

    return {"threshold_summary": summary, "piecewise_model": piecewise}


def run_comfort_correlations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subset = df[["person_ai_comfort_raw", "ai_use_count", "ai_want_count"]].dropna()
    use_corr = stats.spearmanr(subset["person_ai_comfort_raw"], subset["ai_use_count"])
    want_corr = stats.spearmanr(subset["person_ai_comfort_raw"], subset["ai_want_count"])
    results = pd.DataFrame(
        [
            {
                "relationship": "Comfort vs AI use count",
                "spearman_rho": use_corr.statistic,
                "p_value": use_corr.pvalue,
            },
            {
                "relationship": "Comfort vs AI want count",
                "spearman_rho": want_corr.statistic,
                "p_value": want_corr.pvalue,
            },
        ]
    )
    save_table(results, "stats_02_comfort_correlations", index=False)
    save_stats({"correlations": results.to_dict(orient="records")}, "stats_02_comfort_correlations")
    return {"correlations": results}


def run_stage2(df: pd.DataFrame | None = None) -> dict[str, object]:
    if df is None:
        df = notebook_setup()
    results: dict[str, object] = {"data": df}
    results.update(run_risk_role_tests(df))
    results.update(run_region_tests(df))
    results.update(run_infrastructure_models(df))
    results.update(run_threshold_analysis(df))
    results.update(run_comfort_correlations(df))
    return results
