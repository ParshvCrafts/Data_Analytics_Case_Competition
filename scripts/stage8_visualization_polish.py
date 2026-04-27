from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import PALETTE, add_source_note, notebook_setup, save_figure, save_table, save_stats


HEADLINE_FIGURES = [
    "fig_06_want_use_gap_diverging.png",
    "fig_06_wug_segment_matrix.png",
    "fig_03_persona_parallel.png",
    "fig_02_risk_by_role_hypothesis.png",
    "fig_08_infrastructure_lift.png",
    "fig_05_regional_scorecard.png",
    "fig_04_hope_fear_ratio.png",
    "fig_06_signature_periodic_table.png",
]


def build_infrastructure_lift(df: pd.DataFrame) -> None:
    summary = df.groupby("infra_score")["ai_use_count"].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(summary["infra_score"], summary["mean"], marker="o", linewidth=2.8, color=PALETTE["primary"])
    ax.fill_between(summary["infra_score"], summary["mean"], color=PALETTE["accent2"], alpha=0.18)
    for row in summary.itertuples():
        ax.text(row.infra_score, row.mean + 0.08, f"{row.mean:.2f}", ha="center", fontsize=9)
    ax.set_xlabel("Infrastructure score (0-5)")
    ax.set_ylabel("Mean number of AI use cases")
    ax.set_title("Each extra piece of data infrastructure lifts real-world AI use")
    add_source_note(ax)
    save_figure(fig, "fig_08_infrastructure_lift")
    plt.close(fig)


def run_stage8(df: pd.DataFrame | None = None) -> dict[str, object]:
    if df is None:
        df = notebook_setup()
    build_infrastructure_lift(df)
    manifest = pd.DataFrame({"figure": HEADLINE_FIGURES, "exists": [Path("outputs/figures", fig).exists() for fig in HEADLINE_FIGURES]})
    save_table(manifest, "headline_figure_manifest", index=False)
    save_stats({"headline_figures": manifest.to_dict(orient="records")}, "stats_08_headline_figures")
    return {"manifest": manifest}


if __name__ == "__main__":
    results = run_stage8()
    print(results["manifest"].to_string(index=False))
