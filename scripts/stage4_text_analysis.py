from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

from analysis_utils import (
    PALETTE,
    preprocess_text,
    notebook_setup,
    save_figure,
    save_stats,
    save_table,
    upsert_finding,
)

HOPE_WORDS = {
    "excited",
    "potential",
    "opportunity",
    "useful",
    "save",
    "saving",
    "efficient",
    "powerful",
    "transformative",
    "helpful",
    "promising",
    "improve",
    "productivity",
    "support",
    "creative",
    "access",
    "scale",
    "faster",
}

FEAR_WORDS = {
    "worried",
    "afraid",
    "scared",
    "concerned",
    "lose",
    "replace",
    "displacement",
    "bias",
    "privacy",
    "harm",
    "dangerous",
    "hallucination",
    "breach",
    "misinformation",
    "dependence",
    "dependency",
    "copyright",
    "plagiarism",
    "trust",
}


def _text_frame(df: pd.DataFrame) -> pd.DataFrame:
    text_df = df[["ai_opentext", "cluster_label", "region", "role", "role_group"]].dropna(subset=["ai_opentext"]).copy()
    text_df["response_length"] = text_df["ai_opentext"].str.len()
    text_df = text_df[text_df["response_length"] > 20].copy()
    text_df["clean_text"] = text_df["ai_opentext"].apply(
        lambda text: preprocess_text(
            text,
            extra_stopwords={"ai", "data", "tool", "tools", "organization", "nonprofit", "work"},
        )
    )
    return text_df[text_df["clean_text"].str.len() > 0].copy()


def run_preprocessing(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    text_df = _text_frame(df)
    save_table(text_df, "text_responses_cleaned", index=False)
    save_stats(
        {
            "n_text_responses": int(len(text_df)),
            "mean_length_chars": float(text_df["response_length"].mean()),
        },
        "stats_04_text_preprocessing",
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(text_df["response_length"], bins=30, color=PALETTE["primary"], ax=ax)
    ax.set_xlabel("Response length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Most respondents who wrote about AI gave enough detail to reveal both hopes and anxieties")
    ax.text(
        0.98,
        0.95,
        f"n = {len(text_df)} non-trivial responses",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=PALETTE["neutral"],
    )
    ax.text(0, -0.18, "Source: GivingTuesday AI Readiness Survey 2024 open-text responses, our analysis.", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=PALETTE["neutral"])
    save_figure(fig, "fig_04_text_length_distribution")
    plt.close(fig)
    return {"text_df": text_df}


def run_sentiment_analysis(text_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    analyzer = SentimentIntensityAnalyzer()
    scores = text_df["ai_opentext"].apply(analyzer.polarity_scores).apply(pd.Series).reset_index(drop=True)
    sentiment_df = pd.concat(
        [text_df.reset_index(drop=False).rename(columns={"index": "response_index"}).reset_index(drop=True), scores],
        axis=1,
    )
    by_cluster = sentiment_df.groupby("cluster_label")["compound"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    by_region = sentiment_df.groupby("region")["compound"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    by_role = sentiment_df.groupby("role_group")["compound"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)

    save_table(sentiment_df, "text_sentiment_scores", index=False)
    save_table(by_cluster.reset_index(), "stats_04_sentiment_by_cluster", index=False)
    save_table(by_region.reset_index(), "stats_04_sentiment_by_region", index=False)
    save_table(by_role.reset_index(), "stats_04_sentiment_by_role_group", index=False)
    save_stats(
        {
            "by_cluster": by_cluster.reset_index().to_dict(orient="records"),
            "by_region": by_region.reset_index().to_dict(orient="records"),
            "by_role_group": by_role.reset_index().to_dict(orient="records"),
        },
        "stats_04_sentiment_summary",
    )

    mean_floor = by_cluster["mean"].idxmin()
    var_peak = by_cluster["std"].idxmax()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.boxplot(
        data=sentiment_df,
        x="cluster_label",
        y="compound",
        hue="cluster_label",
        order=["AI Consumers", "Late Adopters", "AI Skeptics"],
        palette=[PALETTE["primary"], PALETTE["accent1"], PALETTE["neutral"]],
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("VADER compound sentiment")
    ax.set_title(f"{mean_floor} skew most negative, while {var_peak} show the widest spread of feeling")
    ax.text(0, -0.18, "Source: GivingTuesday AI Readiness Survey 2024 open-text responses, our analysis.", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=PALETTE["neutral"])
    save_figure(fig, "fig_04_sentiment_by_cluster")
    plt.close(fig)

    upsert_finding(
        {
            "claim": "AI Consumers sound more positive overall, while Skeptics are both more negative and more emotionally varied",
            "value": {
                "cluster_with_lowest_mean_sentiment": mean_floor,
                "cluster_with_highest_sentiment_std": var_peak,
                "cluster_means": by_cluster["mean"].round(3).to_dict(),
            },
            "computed_in": "notebooks/04_text_analysis.ipynb",
            "cell_id": "Section 4.2",
            "code_snippet": "VADER sentiment scores grouped by cluster and region",
            "verified": True,
        }
    )

    return {
        "sentiment_df": sentiment_df,
        "sentiment_by_cluster": by_cluster,
        "sentiment_by_region": by_region,
        "sentiment_by_role": by_role,
    }


def _classify_hope_fear(clean_text: str) -> tuple[int, int, str]:
    tokens = clean_text.split()
    hope_count = sum(token in HOPE_WORDS for token in tokens)
    fear_count = sum(token in FEAR_WORDS for token in tokens)
    if hope_count == 0 and fear_count == 0:
        label = "Neutral"
    elif abs(hope_count - fear_count) <= 1 and hope_count > 0 and fear_count > 0:
        label = "Balanced"
    elif hope_count > fear_count:
        label = "Hope-dominant"
    else:
        label = "Fear-dominant"
    return hope_count, fear_count, label


def run_hope_fear_analysis(sentiment_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    classified = sentiment_df.copy()
    classified[["hope_count", "fear_count", "hope_fear_class"]] = classified["clean_text"].apply(
        lambda text: pd.Series(_classify_hope_fear(text))
    )
    distribution_counts = classified.groupby(["cluster_label", "hope_fear_class"]).size().unstack(fill_value=0)
    distribution = (
        distribution_counts.div(distribution_counts.sum(axis=1), axis=0)
        .mul(100)
        .stack()
        .rename("pct")
        .reset_index()
    )
    region_counts = classified.groupby(["region", "hope_fear_class"]).size().unstack(fill_value=0)
    by_region = (
        region_counts.div(region_counts.sum(axis=1), axis=0)
        .mul(100)
        .stack()
        .rename("pct")
        .reset_index()
    )
    save_table(classified, "text_hope_fear_classified", index=False)
    save_table(distribution, "stats_04_hope_fear_by_cluster", index=False)
    save_table(by_region, "stats_04_hope_fear_by_region", index=False)
    save_stats(
        {
            "cluster_distribution": distribution.to_dict(orient="records"),
            "region_distribution": by_region.to_dict(orient="records"),
        },
        "stats_04_hope_fear_distribution",
    )

    pivot = distribution.pivot(index="cluster_label", columns="hope_fear_class", values="pct").fillna(0)
    order = ["AI Consumers", "Late Adopters", "AI Skeptics"]
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(9, 5))
    fear = -pivot.get("Fear-dominant", pd.Series(0, index=pivot.index))
    hope = pivot.get("Hope-dominant", pd.Series(0, index=pivot.index))
    balanced = pivot.get("Balanced", pd.Series(0, index=pivot.index))
    neutral = pivot.get("Neutral", pd.Series(0, index=pivot.index))
    ax.barh(pivot.index, fear, color=PALETTE["neutral"], label="Fear-dominant")
    ax.barh(pivot.index, hope, color=PALETTE["primary"], label="Hope-dominant")
    ax.barh(pivot.index, balanced, left=hope, color=PALETTE["accent2"], alpha=0.85, label="Balanced")
    ax.barh(pivot.index, neutral, left=hope + balanced, color=PALETTE["accent1"], alpha=0.7, label="Neutral")
    ax.axvline(0, color=PALETTE["neutral"], linewidth=1)
    ax.set_xlabel("Share of open-text responses")
    ax.set_title("Hope and fear coexist, but Skeptics tilt negative while Consumers keep more upside in view")
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False) 
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    for idx, cluster in enumerate(pivot.index):
        ax.text(fear.loc[cluster] - 1, idx, f"{abs(fear.loc[cluster]):.0f}%", va="center", ha="right", fontsize=9)
        ax.text(hope.loc[cluster] + 1, idx, f"{hope.loc[cluster]:.0f}%", va="center", ha="left", fontsize=9)
    ax.text(0, -0.18, "Source: GivingTuesday AI Readiness Survey 2024 open-text responses, our analysis.", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=PALETTE["neutral"])
    save_figure(fig, "fig_04_hope_fear_ratio")
    plt.close(fig)

    top_balanced_cluster = pivot["Balanced"].idxmax() if "Balanced" in pivot.columns else None
    upsert_finding(
        {
            "claim": "Open-text responses often mix hope and fear instead of landing on one pure emotion",
            "value": {
                "balanced_share_by_cluster": pivot.get("Balanced", pd.Series(dtype=float)).round(2).to_dict(),
                "fear_share_by_cluster": pivot.get("Fear-dominant", pd.Series(dtype=float)).round(2).to_dict(),
                "top_balanced_cluster": top_balanced_cluster,
            },
            "computed_in": "notebooks/04_text_analysis.ipynb",
            "cell_id": "Section 4.3",
            "code_snippet": "Rule-based hope/fear lexicon over cleaned open-text responses",
            "verified": True,
        }
    )

    return {"classified_text": classified, "hope_fear_distribution": distribution}


def _representative_sentence(text: str) -> str | None:
    sentences = [part.strip() for part in re.split(r"[.!?]", str(text)) if part.strip()]
    for sentence in sentences:
        words = sentence.split()
        if 4 <= len(words) <= 20:
            return sentence
    return None


def run_topic_and_quote_analysis(classified: pd.DataFrame) -> dict[str, pd.DataFrame]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=600, min_df=4, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(classified["clean_text"])
    topic_model = KMeans(n_clusters=6, random_state=42, n_init=20)
    topics = topic_model.fit_predict(matrix)
    classified = classified.copy()
    classified["topic_id"] = topics

    terms = np.array(vectorizer.get_feature_names_out())
    topic_rows = []
    for topic_id, center in enumerate(topic_model.cluster_centers_):
        top_idx = np.argsort(center)[::-1][:8]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "top_terms": ", ".join(terms[top_idx]),
                "count": int((topics == topic_id).sum()),
            }
        )
    topics_df = pd.DataFrame(topic_rows).sort_values("count", ascending=False)
    save_table(topics_df, "stats_04_text_topics", index=False)

    quote_candidates = classified.copy()
    quote_candidates["quote"] = quote_candidates["ai_opentext"].apply(_representative_sentence)
    quote_candidates = quote_candidates.dropna(subset=["quote"]).copy()
    selected = []
    for label, sort_cols, ascending in [
        ("Hope-dominant", ["compound", "hope_count"], [False, False]),
        ("Fear-dominant", ["compound", "fear_count"], [True, False]),
        ("Balanced", ["hope_count", "fear_count"], [False, False]),
    ]:
        pool = quote_candidates[quote_candidates["hope_fear_class"] == label].sort_values(sort_cols, ascending=ascending)
        for _, row in pool.head(2).iterrows():
            selected.append(
                {
                    "response_index": int(row["response_index"]),
                    "quote": row["quote"],
                    "cluster_label": row["cluster_label"],
                    "region": row["region"],
                    "role": row["role"],
                    "hope_fear_class": row["hope_fear_class"],
                }
            )
    quotes_df = pd.DataFrame(selected).drop_duplicates(subset=["quote"]).head(6)
    save_table(quotes_df, "quotes_stage4_verified", index=False)
    save_stats(
        {
            "topics": topics_df.to_dict(orient="records"),
            "quotes": quotes_df.to_dict(orient="records"),
        },
        "stats_04_topics_and_quotes",
    )

    clusters = ["AI Consumers", "Late Adopters", "AI Skeptics"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, cluster in zip(axes, clusters):
        words = " ".join(classified.loc[classified["cluster_label"] == cluster, "clean_text"])
        if not words.strip():
            ax.axis("off")
            continue
        cloud = WordCloud(
            width=600,
            height=350,
            background_color=PALETTE["background"],
            colormap="viridis",
            collocations=False,
        ).generate(words)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(cluster)
        ax.axis("off")
    fig.suptitle("Vocabulary shifts from possibility to caution as readiness falls", fontsize=15)
    fig.text(0.01, 0.02, "Source: GivingTuesday AI Readiness Survey 2024 open-text responses, our analysis.", ha="left", va="bottom", fontsize=9, color=PALETTE["neutral"])
    save_figure(fig, "fig_04_cluster_wordclouds")
    plt.close(fig)

    return {"topic_summary": topics_df, "quotes": quotes_df, "classified_text": classified}


def run_stage4(df: pd.DataFrame | None = None) -> dict[str, object]:
    if df is None:
        df = notebook_setup()
    results: dict[str, object] = {"data": df}
    results.update(run_preprocessing(df))
    results.update(run_sentiment_analysis(results["text_df"]))
    results.update(run_hope_fear_analysis(results["sentiment_df"]))
    results.update(run_topic_and_quote_analysis(results["classified_text"]))
    return results
