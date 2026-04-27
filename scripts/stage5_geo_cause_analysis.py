from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from analysis_utils import (
    AI_RISK_LABELS,
    AI_USE_COLS,
    AI_WANT_COLS,
    PALETTE,
    RISK_TO_COLUMN,
    USE_COLS_MAIN,
    WANT_COLS_MAIN,
    add_source_note,
    notebook_setup,
    save_figure,
    save_stats,
    save_table,
    top_binary_labels,
    upsert_finding,
)


def run_regional_scorecard(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    regions = ["North America", "Europe", "India", "Africa", "Latin America", "Other Asia"]
    subset = df[df["region"].isin(regions)].copy()
    records = []
    risk_labels = {column: AI_RISK_LABELS[risk] for risk, column in RISK_TO_COLUMN.items()}
    for region in regions:
        region_df = subset[subset["region"] == region]
        if region_df.empty:
            continue
        records.append(
            {
                "region": region,
                "count": len(region_df),
                "pct_ai_consumers": (region_df["cluster3"] == 1).mean() * 100,
                "pct_has_tech_or_merl": region_df["has_tech_or_merl"].mean() * 100,
                "mean_comfort": region_df["person_ai_comfort_raw"].mean(),
                "mean_wug": region_df["want_use_gap"].mean(),
                "top_ai_uses": " | ".join(top_binary_labels(region_df, USE_COLS_MAIN, AI_USE_COLS)),
                "top_ai_wants": " | ".join(top_binary_labels(region_df, WANT_COLS_MAIN, AI_WANT_COLS)),
                "top_risks": " | ".join(top_binary_labels(region_df, list(RISK_TO_COLUMN.values()), risk_labels)),
            }
        )
    scorecard = pd.DataFrame(records)
    save_table(scorecard, "regional_scorecard", index=False)
    save_stats({"regional_scorecard": scorecard.to_dict(orient="records")}, "stats_05_regional_scorecard")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, row in zip(axes.flatten(), scorecard.itertuples()):
        ax.axis("off")
        ax.text(0.0, 0.96, row.region, fontsize=15, fontweight="bold", color=PALETTE["primary"], transform=ax.transAxes, va="top")
        ax.text(0.0, 0.84, f"n = {row.count}", fontsize=10, color=PALETTE["neutral"], transform=ax.transAxes)
        if row.count < 30:
            ax.text(0.22, 0.84, "Small sample", fontsize=10, color=PALETTE["accent1"], transform=ax.transAxes)
        ax.text(0.0, 0.68, f"AI Consumers: {row.pct_ai_consumers:.0f}%", fontsize=12, transform=ax.transAxes)
        ax.text(0.0, 0.58, f"Tech or MERL staff: {row.pct_has_tech_or_merl:.0f}%", fontsize=12, transform=ax.transAxes)
        ax.text(0.0, 0.48, f"Comfort: {row.mean_comfort:.1f} / 10", fontsize=12, transform=ax.transAxes)
        ax.text(0.0, 0.38, f"Mean WUG: {row.mean_wug:.2f}", fontsize=12, transform=ax.transAxes)
        ax.text(0.0, 0.25, f"Top uses: {row.top_ai_uses}", fontsize=11, transform=ax.transAxes, wrap=True)
        ax.text(0.0, 0.13, f"Top wants: {row.top_ai_wants}", fontsize=11, transform=ax.transAxes, wrap=True)
        ax.text(0.0, 0.01, f"Top risks: {row.top_risks}", fontsize=11, transform=ax.transAxes, wrap=True)
    fig.suptitle("India and Africa tell different readiness stories even before we get to the North-South split", fontsize=16)
    
    save_figure(fig, "fig_05_regional_scorecard")
    plt.close(fig)

    india_row = scorecard.loc[scorecard["region"] == "India"].iloc[0]
    africa_row = scorecard.loc[scorecard["region"] == "Africa"].iloc[0]
    upsert_finding(
        {
            "claim": "India and Africa look meaningfully different even though both are usually bundled into the Global South story",
            "value": {
                "india_mean_wug": round(float(india_row["mean_wug"]), 3),
                "africa_mean_wug": round(float(africa_row["mean_wug"]), 3),
                "india_mean_comfort": round(float(india_row["mean_comfort"]), 3),
                "africa_mean_comfort": round(float(africa_row["mean_comfort"]), 3),
            },
            "computed_in": "notebooks/05_geo_cause_analysis.ipynb",
            "cell_id": "Section 5.1",
            "code_snippet": "regional scorecard aggregations across AI consumer share, comfort, and WUG",
            "verified": True,
        }
    )

    return {"regional_scorecard": scorecard}


def run_cause_area_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    exploded = df[["cluster_label", "ai_use_count", "want_use_gap", "org_label_list"]].explode("org_label_list").copy()
    exploded["cause_area"] = exploded["org_label_list"].fillna("No Answer")
    exploded = exploded[~exploded["cause_area"].isin(["No Answer", "Other"])].copy()
    counts = exploded["cause_area"].value_counts()
    keep = counts[counts >= 12].index
    exploded = exploded[exploded["cause_area"].isin(keep)].copy()

    cluster_share = (
        exploded.groupby(["cause_area", "cluster_label"])
        .size()
        .unstack(fill_value=0)
    )
    cluster_share = (
        cluster_share.div(cluster_share.sum(axis=1), axis=0)
        .mul(100)
        .fillna(0)
    )
    metrics = (
        exploded.groupby("cause_area")
        .agg(mean_ai_use=("ai_use_count", "mean"), mean_wug=("want_use_gap", "mean"), count=("cause_area", "size"))
        .sort_values("mean_wug", ascending=False)
    )
    save_table(cluster_share.reset_index(), "cause_area_cluster_shares", index=False)
    save_table(metrics.reset_index(), "cause_area_metrics", index=False)
    save_stats(
        {
            "cause_area_cluster_shares": cluster_share.reset_index().to_dict(orient="records"),
            "cause_area_metrics": metrics.reset_index().to_dict(orient="records"),
        },
        "stats_05_cause_area",
    )

    ordered_causes = metrics.index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(ordered_causes) * 0.45)), gridspec_kw={"width_ratios": [1.3, 0.9]})
    sns.heatmap(
        cluster_share.reindex(ordered_causes)[["AI Consumers", "Late Adopters", "AI Skeptics"]],
        cmap="Blues",
        annot=True,
        fmt=".0f",
        linewidths=0.4,
        cbar_kws={"label": "% of cause-area respondents"},
        ax=axes[0],
    )
    axes[0].set_title("Cause areas split across the three clusters in visibly different ways")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    axes[1].barh(ordered_causes, metrics.loc[ordered_causes, "mean_wug"], color=PALETTE["accent1"])
    axes[1].invert_yaxis()
    axes[1].set_title("Some cause areas face much larger AI appetite than actual use")
    axes[1].set_xlabel("Mean Want-Use Gap")
    for idx, value in enumerate(metrics.loc[ordered_causes, "mean_wug"]):
        axes[1].text(value + 0.02, idx, f"{value:.2f}", va="center", fontsize=9)
    add_source_note(axes[0], y=-0.12)
    save_figure(fig, "fig_05_cause_area_heatmap")
    plt.close(fig)

    return {"cause_area_shares": cluster_share, "cause_area_metrics": metrics}


def run_india_rural_urban(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    india = df[df["region"] == "India"].copy()
    india = india[india["india_rural_urban"].notna()].copy()
    summary = (
        india.assign(is_ai_consumer=(india["cluster3"] == 1).astype(int))
        .groupby("india_rural_urban")
        .agg(
            count=("cluster3", "size"),
            pct_ai_consumers=("is_ai_consumer", lambda s: s.mean() * 100),
            mean_comfort=("person_ai_comfort_raw", "mean"),
            mean_infra=("infra_score", "mean"),
            mean_wug=("want_use_gap", "mean"),
        )
        .reset_index()
    )
    save_table(summary, "india_rural_urban_summary", index=False)

    kw_comfort = stats.kruskal(*[group["person_ai_comfort_raw"].dropna() for _, group in india.groupby("india_rural_urban")])
    kw_wug = stats.kruskal(*[group["want_use_gap"].dropna() for _, group in india.groupby("india_rural_urban")])
    save_stats(
        {
            "india_rural_urban_summary": summary.to_dict(orient="records"),
            "comfort_kruskal": {"statistic": float(kw_comfort.statistic), "p_value": float(kw_comfort.pvalue)},
            "wug_kruskal": {"statistic": float(kw_wug.statistic), "p_value": float(kw_wug.pvalue)},
        },
        "stats_05_india_rural_urban",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.barplot(
        data=summary,
        x="india_rural_urban",
        y="mean_comfort",
        hue="india_rural_urban",
        palette=[PALETTE["primary"], PALETTE["accent1"], PALETTE["accent2"]],
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Comfort with AI differs across metro, urban, and rural India")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean comfort (0-10)")
    sns.barplot(
        data=summary,
        x="india_rural_urban",
        y="mean_wug",
        hue="india_rural_urban",
        palette=[PALETTE["primary"], PALETTE["accent1"], PALETTE["accent2"]],
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("The want-use gap is not evenly distributed inside India")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean Want-Use Gap")
    add_source_note(axes[0], y=-0.2)
    save_figure(fig, "fig_05_india_rural_urban")
    plt.close(fig)

    return {"india_rural_urban": summary}


def run_stage5(df: pd.DataFrame | None = None) -> dict[str, object]:
    if df is None:
        df = notebook_setup()
    results: dict[str, object] = {"data": df}
    results.update(run_regional_scorecard(df))
    results.update(run_cause_area_analysis(df))
    results.update(run_india_rural_urban(df))
    return results
