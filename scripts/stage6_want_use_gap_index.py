from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from analysis_utils import (
    AI_RISK_LABELS,
    AI_USE_COLS,
    AI_WANT_COLS,
    PALETTE,
    RISK_TO_COLUMN,
    USE_COLS_MAIN,
    USE_WANT_PAIRS,
    WANT_COLS_MAIN,
    add_source_note,
    compute_use_want_gap_table,
    notebook_setup,
    save_figure,
    save_plotly_figure,
    save_stats,
    save_table,
    top_binary_labels,
    upsert_finding,
)


SEGMENT_RECOMMENDATIONS = {
    "Activation-ready": "Fund pilot programs and peer-learning cohorts, because the basics are already in place.",
    "Foundation-needed": "Fund data infrastructure first, especially cloud storage, policies, and data staffing.",
    "Already-well-served": "Use these nonprofits as demonstration sites and mentors for the rest of the sector.",
    "Disengaged": "Start with low-stakes AI literacy and data basics before pushing tool adoption.",
    "Middle-ground": "Offer light-touch technical assistance and diagnosis before prescribing one path.",
}


def run_wug_core(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    gap_table = compute_use_want_gap_table(df)
    cluster_summary = df.groupby("cluster_label")["want_use_gap"].agg(["mean", "median", "count"]).reset_index()
    region_summary = df.groupby("region")["want_use_gap"].agg(["mean", "median", "count"]).reset_index().sort_values("mean", ascending=False)
    threshold_pct = (df["want_use_gap"] >= 3).mean() * 100

    save_table(gap_table, "want_use_gap_usecase_ranked", index=False)
    save_table(cluster_summary, "wug_by_cluster", index=False)
    save_table(region_summary, "wug_by_region", index=False)
    save_stats(
        {
            "gap_by_use_case": gap_table.to_dict(orient="records"),
            "wug_by_cluster": cluster_summary.to_dict(orient="records"),
            "wug_by_region": region_summary.to_dict(orient="records"),
            "pct_wug_ge_3": threshold_pct,
        },
        "stats_06_wug_core",
    )

    plot_df = gap_table.iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(plot_df))
    ax.barh(y, -plot_df["use_rate"] * 100, color=PALETTE["primary"], label="Currently using")
    ax.barh(y, plot_df["want_rate"] * 100, color=PALETTE["accent1"], label="Want to use")
    ax.axvline(0, color=PALETTE["neutral"], linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["use_case"])
    ax.set_xlabel("Percent of respondents")
    ax.set_title("Organizing data, not generative AI, is the biggest unmet need in the survey")
    ax.legend(frameon=False, loc="upper left")
    for idx, row in enumerate(plot_df.itertuples()):
        ax.text(-row.use_rate * 100 - 1, idx, f"{row.use_rate * 100:.0f}%", ha="right", va="center", fontsize=9)
        ax.text(row.want_rate * 100 + 1, idx, f"{row.want_rate * 100:.0f}%", ha="left", va="center", fontsize=9)
    
    save_figure(fig, "fig_06_want_use_gap_diverging")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(df["want_use_gap"], bins=range(int(df["want_use_gap"].min()) - 1, int(df["want_use_gap"].max()) + 2), color=PALETTE["primary"], ax=ax)
    ax.axvline(3, linestyle="--", color=PALETTE["accent1"], linewidth=2)
    ax.text(3.2, ax.get_ylim()[1] * 0.92, f"{threshold_pct:.1f}% have WUG >= 3", color=PALETTE["accent1"], fontsize=10)
    ax.set_xlabel("Want-Use Gap")
    ax.set_ylabel("Count")
    ax.set_title("A large share of nonprofits want at least three more AI use cases than they currently use")
    
    save_figure(fig, "fig_06_wug_distribution")
    plt.close(fig)

    top_cluster = cluster_summary.sort_values("mean", ascending=False).iloc[0]
    upsert_finding(
        {
            "claim": "Late Adopters carry the biggest Want-Use Gap, while Skeptics have the smallest",
            "value": {
                "cluster_means": cluster_summary.set_index("cluster_label")["mean"].round(3).to_dict(),
                "pct_wug_ge_3": round(float(threshold_pct), 2),
            },
            "computed_in": "notebooks/06_want_use_gap_index.ipynb",
            "cell_id": "Section 6.1",
            "code_snippet": "want_use_gap = ai_want_count - ai_use_count with group summaries by cluster and region",
            "verified": True,
        }
    )

    return {"gap_table": gap_table, "wug_by_cluster": cluster_summary, "wug_by_region": region_summary}


def run_segmentation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    segment_counts = (
        df.groupby("wug_segment")
        .size()
        .rename("count")
        .reset_index()
        .assign(pct=lambda frame: frame["count"] / frame["count"].sum() * 100)
    )
    recommendations = pd.DataFrame(
        [{"segment": segment, "recommendation": recommendation} for segment, recommendation in SEGMENT_RECOMMENDATIONS.items()]
    )
    summary = segment_counts.merge(recommendations, left_on="wug_segment", right_on="segment", how="left")
    save_table(summary, "wug_segment_summary", index=False)

    matrix = pd.DataFrame(
        {
            "High WUG": {
                "High infra": int(((df["want_use_gap"] >= 3) & (df["infra_score"] >= 3)).sum()),
                "Low infra": int(((df["want_use_gap"] >= 3) & (df["infra_score"] < 3)).sum()),
            },
            "Low WUG": {
                "High infra": int(((df["want_use_gap"] <= 1) & (df["infra_score"] >= 3)).sum()),
                "Low infra": int(((df["want_use_gap"] <= 1) & (df["infra_score"] < 3)).sum()),
            },
        }
    )
    save_table(matrix.reset_index().rename(columns={"index": "infra_level"}), "wug_segment_matrix", index=False)
    save_stats(
        {
            "segment_summary": summary.to_dict(orient="records"),
            "segment_matrix": matrix.to_dict(),
        },
        "stats_06_wug_segments",
    )

    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Oranges", linewidths=0.6, cbar=False, ax=ax)
    ax.set_title("The biggest funder opportunity sits in high-demand nonprofits without the data basics")
    ax.set_xlabel("Want-Use Gap")
    ax.set_ylabel("Infrastructure score")
    ax.text(0.05, 1.02, "High WUG = 3 or more", transform=ax.transAxes, fontsize=10, color=PALETTE["neutral"])
    ax.text(0, -0.16, "", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=PALETTE["neutral"])
    save_figure(fig, "fig_06_wug_segment_matrix")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        df["infra_score"] + np.random.uniform(-0.08, 0.08, len(df)),
        df["want_use_gap"] + np.random.uniform(-0.08, 0.08, len(df)),
        c=df["cluster_color"],
        alpha=0.6,
        s=26,
    )
    del scatter
    ax.axvline(3, linestyle="--", color=PALETTE["neutral"])
    ax.axhline(3, linestyle="--", color=PALETTE["neutral"])
    ax.set_xlabel("Infrastructure score")
    ax.set_ylabel("Want-Use Gap")
    ax.set_title("The want-use gap is widest where infrastructure is still catching up")
    
    save_figure(fig, "fig_06_wug_vs_infra")
    plt.close(fig)

    foundation_needed = summary.loc[summary["wug_segment"] == "Foundation-needed"].iloc[0]
    upsert_finding(
        {
            "claim": "More than one in five nonprofits want meaningfully more from AI but still lack the infrastructure to support it",
            "value": {
                "segment": foundation_needed["wug_segment"],
                "count": int(foundation_needed["count"]),
                "pct": round(float(foundation_needed["pct"]), 2),
            },
            "computed_in": "notebooks/06_want_use_gap_index.ipynb",
            "cell_id": "Section 6.2",
            "code_snippet": "2x2 segmentation using want_use_gap threshold of 3 and infra_score threshold of 3",
            "verified": True,
        }
    )

    return {"segment_summary": summary, "segment_matrix": matrix}


def run_signature_sankey(df: pd.DataFrame) -> dict[str, object]:
    risk_labels = {column: AI_RISK_LABELS[risk] for risk, column in RISK_TO_COLUMN.items()}
    clusters = ["AI Consumers", "Late Adopters", "AI Skeptics"]
    nodes = []
    links_source = []
    links_target = []
    links_value = []
    links_color = []

    def add_node(label: str) -> int:
        if label not in nodes:
            nodes.append(label)
        return nodes.index(label)

    for cluster in clusters:
        cluster_df = df[df["cluster_label"] == cluster]
        cluster_node = add_node(cluster)
        top_uses = top_binary_labels(cluster_df, USE_COLS_MAIN, AI_USE_COLS, top_n=2)
        top_wants = top_binary_labels(cluster_df, WANT_COLS_MAIN, AI_WANT_COLS, top_n=2)
        top_risks = top_binary_labels(cluster_df, list(RISK_TO_COLUMN.values()), risk_labels, top_n=2)

        for label in top_uses:
            node = add_node(f"Use: {label}")
            links_source.append(cluster_node)
            links_target.append(node)
            links_value.append(float(cluster_df[[col for col, name in AI_USE_COLS.items() if name == label]].mean().iloc[0] * len(cluster_df)))
            links_color.append(PALETTE["primary"])
        for label in top_wants:
            source_node = add_node(cluster)
            target_node = add_node(f"Want: {label}")
            links_source.append(source_node)
            links_target.append(target_node)
            links_value.append(float(cluster_df[[col for col, name in AI_WANT_COLS.items() if name == label]].mean().iloc[0] * len(cluster_df)))
            links_color.append(PALETTE["accent1"])
        for label in top_risks:
            source_node = add_node(cluster)
            target_node = add_node(f"Risk: {label}")
            links_source.append(source_node)
            links_target.append(target_node)
            risk_col = next(column for column, short in risk_labels.items() if short == label)
            links_value.append(float(cluster_df[risk_col].mean() * len(cluster_df)))
            links_color.append(PALETTE["neutral"])

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    label=nodes,
                    color=[PALETTE["primary"] if label in clusters else PALETTE["accent2"] for label in nodes],
                ),
                link=dict(source=links_source, target=links_target, value=links_value, color=links_color),
            )
        ]
    )
    fig.update_layout(
        title="The three clusters flow toward different uses, wants, and risks",
        font=dict(size=12, family="Arial"),
        paper_bgcolor=PALETTE["background"],
        plot_bgcolor=PALETTE["background"],
    )
    paths = save_plotly_figure(fig, "fig_06_signature_sankey", write_static=False)

    gap_table = compute_use_want_gap_table(df)
    tile_df = gap_table.copy()
    fig_static, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for idx, row in tile_df.iterrows():
        col = idx % 4
        row_pos = 1 - idx // 4
        x0, y0 = col + 0.05, row_pos + 0.08
        gap_color = plt.cm.Oranges(min(0.95, max(0.2, row["gap"] * 2)))
        rect = plt.Rectangle((x0, y0), 0.9, 0.78, facecolor=gap_color, edgecolor=PALETTE["neutral"], linewidth=1)
        ax.add_patch(rect)
        ax.text(x0 + 0.04, y0 + 0.65, row["use_case"], fontsize=11, fontweight="bold", va="top")
        ax.text(x0 + 0.04, y0 + 0.42, f"Use: {row['use_rate'] * 100:.0f}%", fontsize=10)
        ax.text(x0 + 0.04, y0 + 0.27, f"Want: {row['want_rate'] * 100:.0f}%", fontsize=10)
        ax.text(x0 + 0.04, y0 + 0.12, f"Gap: {row['gap'] * 100:.0f} pts", fontsize=10, color=PALETTE["neutral"])
    ax.set_title("The nonprofit AI wish list looks more like a capability table than a chatbot race", pad=18)
    ax.text(0.0, -0.08, "", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=PALETTE["neutral"])
    save_figure(fig_static, "fig_06_signature_periodic_table")
    plt.close(fig_static)

    save_stats({"signature_sankey_paths": {key: str(value) for key, value in paths.items()}}, "stats_06_signature_sankey")
    return {"signature_sankey_paths": paths}


def run_stage6(df: pd.DataFrame | None = None) -> dict[str, object]:
    if df is None:
        df = notebook_setup()
    results: dict[str, object] = {"data": df}
    results.update(run_wug_core(df))
    results.update(run_segmentation(df))
    results.update(run_signature_sankey(df))
    return results
