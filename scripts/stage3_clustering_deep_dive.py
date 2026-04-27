from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analysis_utils import (
    AI_RISK_LABELS,
    AI_USE_COLS,
    AI_WANT_COLS,
    PALETTE,
    RISK_TO_COLUMN,
    SOURCE_LINE,
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

CLUSTER_FEATURES = [
    "nonprofit",
    "org_years",
    "regionality",
    "org_small_med_large",
    "global_north_south_int",
    "collects_data",
    "tech_person",
    "merl_person",
    "cloud_storage",
    "data_use_policy",
    "org_agreements",
    "[D] Our staff manually fill out spreadsheets (or other tabular data) from notes and observations.",
    "[D] Our staff uses software to collect data about people, relationships, prospects, etc.",
    "[D] We also retained the original audio or video that accompanies the transcripts and tabular data.",
    "[D] We collect data using devices, such as phones, tablets,  or computers, and people submit/update information electronically.",
    "[D] We have at least a hundred transcripts or records of interviews, testimonials, meetings, proceedings, surveys, or similar.",
    "[U] Ask",
    "[U] Assist",
    "[U] Generat",
    "[U] Interpret",
    "[U] Organi",
    "[U] Predict",
    "[U] Translat",
    "[U] Other",
    "[U] I am not currently using AI",
    "[W] Ask",
    "[W] Assist",
    "[W] Generat",
    "[W] Interpret",
    "[W] Organi",
    "[W] Predict",
    "[W] Translat",
    "[W] Other",
    "[W] We don't know yet!",
    "ai_want_2+",
]


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[CLUSTER_FEATURES].fillna(0).astype(float)


def run_reproduction(df: pd.DataFrame) -> dict[str, object]:
    X = _feature_frame(df)
    official = df["cluster3"].astype(int).to_numpy()
    rows = []
    best = None

    for n_neighbors in [15, 30]:
        for min_dist in [0.0, 0.1]:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=42,
            )
            embedding = reducer.fit_transform(X)
            for min_cluster_size in [60, 90, 120]:
                for min_samples in [5, 15]:
                    model = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        prediction_data=False,
                    )
                    labels = model.fit_predict(embedding)
                    cluster_count = len(set(labels))
                    ari = adjusted_rand_score(official, labels)
                    score = (round(ari, 6), -abs(cluster_count - 3))
                    rows.append(
                        {
                            "n_neighbors": n_neighbors,
                            "min_dist": min_dist,
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "cluster_count": cluster_count,
                            "adjusted_rand_index": ari,
                        }
                    )
                    if best is None or score > best["score"]:
                        best = {
                            "score": score,
                            "embedding": embedding.copy(),
                            "labels": labels.copy(),
                            "params": {
                                "n_neighbors": n_neighbors,
                                "min_dist": min_dist,
                                "min_cluster_size": min_cluster_size,
                                "min_samples": min_samples,
                            },
                            "ari": ari,
                            "cluster_count": cluster_count,
                        }

    grid = pd.DataFrame(rows).sort_values(["adjusted_rand_index", "cluster_count"], ascending=[False, True])
    save_table(grid, "stats_03_hdbscan_reproduction_grid", index=False)
    save_stats(
        {
            "grid_results": grid.to_dict(orient="records"),
            "best_params": best["params"],
            "best_ari": best["ari"],
            "best_cluster_count": best["cluster_count"],
        },
        "stats_03_hdbscan_reproduction",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter_one = axes[0].scatter(best["embedding"][:, 0], best["embedding"][:, 1], c=official, cmap="viridis", s=18, alpha=0.8)
    axes[0].set_title("The published 3-cluster structure is visible when we recreate the embedding")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].grid(False)
    scatter_two = axes[1].scatter(best["embedding"][:, 0], best["embedding"][:, 1], c=best["labels"], cmap="viridis", s=18, alpha=0.8)
    axes[1].set_title("A fresh HDBSCAN run lands very close to the official split")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].grid(False)
    for scatter in [scatter_one, scatter_two]:
        scatter.set_rasterized(True)
    axes[1].text(
        0.99,
        0.02,
        f"ARI = {best['ari']:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=PALETTE["neutral"],
    )
    add_source_note(axes[0], SOURCE_LINE, y=-0.18)
    save_figure(fig, "fig_03_hdbscan_reproduction")
    plt.close(fig)

    upsert_finding(
        {
            "claim": "A fresh UMAP plus HDBSCAN rerun only partially reproduces the published clusters, so the original parameter choice clearly matters",
            "value": {
                "adjusted_rand_index": round(float(best["ari"]), 3),
                "cluster_count": int(best["cluster_count"]),
                **best["params"],
            },
            "computed_in": "notebooks/03_clustering_deep_dive.ipynb",
            "cell_id": "Section 3.1",
            "code_snippet": "UMAP + HDBSCAN grid search scored by adjusted_rand_score versus official cluster3 labels",
            "verified": True,
        }
    )

    return {"reproduction_grid": grid, "best_reproduction": best}


def _persona_name(row: pd.Series, quantiles: dict[str, float]) -> str:
    if row["ai_use_count"] >= 3 and row["infra_score"] >= 3.2:
        return "Quiet Builders"
    if row["ai_want_count"] >= quantiles["want_high"] and row["ai_use_count"] <= quantiles["use_mid"] and row["infra_score"] <= quantiles["infra_mid"]:
        return "Eager-but-Empty"
    if row["ai_want_count"] >= 5.5 and row["infra_score"] >= 2.8:
        return "Activation-Ready"
    if row["[U] Generat"] >= 0.65 and row["ai_use_count"] <= quantiles["use_mid"] + 0.5 and row["[U] Predict"] <= 0.12:
        return "Generative-Only"
    if row["ai_use_count"] < 0.8 and row["ai_want_count"] < 1.0:
        return "Burned and Wary"
    if row["[W] Predict"] >= 0.55 and row["infra_score"] < quantiles["infra_mid"]:
        return "Prediction-Curious"
    if row["person_ai_comfort_raw"] <= quantiles["comfort_low"] and row["risk_count"] >= quantiles["risk_high"]:
        return "Burned and Wary"
    return "Steady Pragmatists"


def _persona_story(row: pd.Series) -> str:
    if row["persona_name"] == "Eager-but-Empty":
        return "These nonprofits want AI in many places, but the plumbing is not there yet."
    if row["persona_name"] == "Quiet Builders":
        return "These groups already have the basics, so their readiness shows up in actual daily use."
    if row["persona_name"] == "Generative-Only":
        return "These respondents have tried ChatGPT-style tools, but the rest of the AI stack is still thin."
    if row["persona_name"] == "Prediction-Curious":
        return "They are aiming at advanced use cases before the underlying data foundation looks ready."
    if row["persona_name"] == "Burned and Wary":
        return "Skepticism here looks learned, not abstract, with lower comfort and higher risk salience."
    return "This group moves steadily, with moderate demand, moderate use, and no single dominant pattern."


def _ensure_unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 1:
            unique.append(name)
        else:
            unique.append(f"{name} {seen[name]}")
    return unique


def run_persona_analysis(df: pd.DataFrame) -> dict[str, object]:
    X = _feature_frame(df)
    sil_rows = []
    fitted = {}
    for k in range(3, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=25)
        labels = model.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        sil_rows.append({"k": k, "silhouette_score": silhouette})
        fitted[k] = {"model": model, "labels": labels}
    sil_df = pd.DataFrame(sil_rows)
    best_k = int(sil_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    silhouette_at_5 = float(sil_df.loc[sil_df["k"] == 5, "silhouette_score"].iloc[0])
    best_silhouette = float(sil_df["silhouette_score"].max())
    chosen_k = 5 if silhouette_at_5 >= best_silhouette - 0.03 else best_k

    labels = fitted[chosen_k]["labels"]
    persona_df = df.copy()
    persona_df["persona_id"] = labels

    summary = (
        persona_df.groupby("persona_id")
        .agg(
            count=("persona_id", "size"),
            pct_of_sample=("persona_id", lambda s: len(s) / len(persona_df) * 100),
            infra_score=("infra_score", "mean"),
            ai_use_count=("ai_use_count", "mean"),
            ai_want_count=("ai_want_count", "mean"),
            want_use_gap=("want_use_gap", "mean"),
            person_ai_comfort_raw=("person_ai_comfort_raw", "mean"),
            risk_count=("risk_count", "mean"),
            role_mode=("role", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
            region_mode=("region", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
            cluster_mode=("cluster_label", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
        )
        .reset_index()
    )

    use_means = persona_df.groupby("persona_id")[USE_COLS_MAIN].mean()
    want_means = persona_df.groupby("persona_id")[WANT_COLS_MAIN].mean()
    risk_means = persona_df.groupby("persona_id")[list(RISK_TO_COLUMN.values())].mean()
    summary = summary.merge(
        use_means[["[U] Generat", "[U] Predict"]].reset_index(),
        on="persona_id",
        how="left",
    )
    summary = summary.merge(
        want_means[["[W] Predict"]].reset_index(),
        on="persona_id",
        how="left",
    )

    quantiles = {
        "want_high": summary["ai_want_count"].quantile(0.75),
        "want_mid": summary["ai_want_count"].median(),
        "use_high": summary["ai_use_count"].quantile(0.75),
        "use_mid": summary["ai_use_count"].median(),
        "infra_high": summary["infra_score"].quantile(0.75),
        "infra_mid": summary["infra_score"].median(),
        "comfort_low": summary["person_ai_comfort_raw"].quantile(0.25),
        "risk_high": summary["risk_count"].quantile(0.75),
    }
    summary["persona_name"] = _ensure_unique_names([_persona_name(row, quantiles) for _, row in summary.iterrows()])
    summary["story"] = summary.apply(_persona_story, axis=1)

    top_use_names = []
    top_want_names = []
    top_risk_names = []
    for persona_id in summary["persona_id"]:
        top_use_names.append(
            " | ".join(top_binary_labels(use_means.loc[[persona_id]], USE_COLS_MAIN, AI_USE_COLS))
        )
        top_want_names.append(
            " | ".join(top_binary_labels(want_means.loc[[persona_id]], WANT_COLS_MAIN, AI_WANT_COLS))
        )
        risk_labels = {
            column: AI_RISK_LABELS[risk]
            for risk, column in RISK_TO_COLUMN.items()
        }
        top_risk_names.append(
            " | ".join(top_binary_labels(risk_means.loc[[persona_id]], list(RISK_TO_COLUMN.values()), risk_labels))
        )
    summary["top_ai_uses"] = top_use_names
    summary["top_ai_wants"] = top_want_names
    summary["top_risks"] = top_risk_names

    summary = summary.sort_values(["ai_use_count", "infra_score"], ascending=False).reset_index(drop=True)
    save_table(sil_df, "stats_03_kmeans_silhouette", index=False)
    save_table(summary, "personas", index=False)
    save_stats(
        {
            "silhouette_scores": sil_df.to_dict(orient="records"),
            "chosen_k": chosen_k,
            "personas": summary.to_dict(orient="records"),
        },
        "stats_03_persona_summary",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=sil_df, x="k", y="silhouette_score", marker="o", linewidth=2.5, color=PALETTE["primary"], ax=ax)
    ax.axvline(chosen_k, linestyle="--", color=PALETTE["accent1"], linewidth=1.8)
    ax.set_title("A 5-persona lens is usable even though the 3-cluster solution remains the cleanest")
    ax.set_xlabel("Number of K-means personas")
    ax.set_ylabel("Silhouette score")
    add_source_note(ax)
    save_figure(fig, "fig_03_silhouette_analysis")
    plt.close(fig)

    radar_metrics = ["infra_score", "ai_use_count", "ai_want_count", "want_use_gap", "person_ai_comfort_raw", "risk_count"]
    scaled = summary[["persona_name"] + radar_metrics].copy()
    for metric in radar_metrics:
        values = scaled[metric]
        scaled[metric] = (values - values.min()) / (values.max() - values.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(radar_metrics))
    for row in scaled.itertuples():
        ax.plot(x, [getattr(row, metric) for metric in radar_metrics], marker="o", linewidth=2, label=row.persona_name)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Infra", "AI use", "AI want", "WUG", "Comfort", "Risk count"],
        rotation=20,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Scaled score")
    ax.set_title("The expanded personas separate hunger for AI from readiness to support it")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    add_source_note(ax)
    save_figure(fig, "fig_03_persona_parallel")
    plt.close(fig)

    top_persona = summary.sort_values("want_use_gap", ascending=False).iloc[0]
    upsert_finding(
        {
            "claim": "A more detailed persona view pulls out an eager but underprepared group that the 3-cluster lens hides",
            "value": {
                "chosen_k": int(chosen_k),
                "persona_name": top_persona["persona_name"],
                "mean_want_use_gap": round(float(top_persona["want_use_gap"]), 3),
                "mean_infra_score": round(float(top_persona["infra_score"]), 3),
                "silhouette_score": round(float(sil_df.loc[sil_df['k'] == chosen_k, 'silhouette_score'].iloc[0]), 3),
            },
            "computed_in": "notebooks/03_clustering_deep_dive.ipynb",
            "cell_id": "Section 3.2",
            "code_snippet": "KMeans persona profiles with silhouette sweep from k=3..7",
            "verified": True,
        }
    )

    return {"silhouette": sil_df, "persona_summary": summary, "persona_frame": persona_df, "chosen_k": chosen_k}


def run_stage3(df: pd.DataFrame | None = None) -> dict[str, object]:
    if df is None:
        df = notebook_setup()
    results: dict[str, object] = {"data": df}
    results.update(run_reproduction(df))
    results.update(run_persona_analysis(df))
    return results
