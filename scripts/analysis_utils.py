from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from columns import (
    AI_RISK_LABELS,
    AI_USE_COLS,
    AI_WANT_COLS,
    CLUSTER_COLORS,
    CLUSTER_LABELS,
    CLUSTERING_CSV,
    FIGURES_DIR,
    FINDINGS_JSON,
    INFRA_COLS,
    OUTPUTS_DIR,
    PALETTE,
    RAW_SURVEY_CSV,
    STATS_DIR,
    TABLES_DIR,
    apply_chart_style,
    build_region,
)

SOURCE_LINE = "Source: GivingTuesday AI Readiness Survey 2024, n=930, our analysis."
USE_COLS_MAIN = [col for col in AI_USE_COLS if col != "[U] I am not currently using AI"]
WANT_COLS_MAIN = [col for col in AI_WANT_COLS if col != "[W] We don't know yet!"]
USE_WANT_PAIRS = list(zip(USE_COLS_MAIN, WANT_COLS_MAIN))
RISK_TO_COLUMN = {
    risk: "risk__" + re.sub(r"[^a-z0-9]+", "_", short.lower()).strip("_")
    for risk, short in AI_RISK_LABELS.items()
}


def ensure_output_dirs() -> None:
    for directory in [OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, STATS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def read_findings() -> list[dict[str, Any]]:
    if not FINDINGS_JSON.exists():
        return []
    with FINDINGS_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _next_finding_id(findings: list[dict[str, Any]]) -> str:
    used = []
    for finding in findings:
        finding_id = str(finding.get("id", ""))
        if finding_id.startswith("F") and finding_id[1:].isdigit():
            used.append(int(finding_id[1:]))
    return f"F{(max(used) + 1 if used else 1):03d}"


def upsert_finding(entry: dict[str, Any]) -> dict[str, Any]:
    findings = read_findings()
    entry = {key: json_ready(value) for key, value in entry.items()}
    match_idx = None
    for idx, finding in enumerate(findings):
        if (
            finding.get("claim") == entry.get("claim")
            and finding.get("computed_in") == entry.get("computed_in")
            and finding.get("cell_id") == entry.get("cell_id")
        ):
            match_idx = idx
            entry["id"] = finding.get("id", entry.get("id"))
            break
    if "id" not in entry or not entry["id"]:
        entry["id"] = _next_finding_id(findings)
    if match_idx is None:
        findings.append(entry)
    else:
        findings[match_idx] = entry
    with FINDINGS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(findings, handle, indent=2, ensure_ascii=True)
    return entry


def save_stats(payload: dict[str, Any], stem: str) -> Path:
    ensure_output_dirs()
    path = STATS_DIR / f"{stem}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=True)
    return path


def save_table(frame: pd.DataFrame, stem: str, index: bool = True) -> Path:
    ensure_output_dirs()
    path = TABLES_DIR / f"{stem}.csv"
    frame.to_csv(path, index=index)
    return path


def add_source_note(ax: plt.Axes, text: str = SOURCE_LINE, y: float = -0.16) -> None:
    ax.text(
        0,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=PALETTE["neutral"],
    )


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> tuple[Path, Path]:
    ensure_output_dirs()
    png_path = FIGURES_DIR / f"{stem}.png"
    svg_path = FIGURES_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    return png_path, svg_path


def save_plotly_figure(fig, stem: str, write_static: bool = False) -> dict[str, Path]:
    ensure_output_dirs()
    html_path = FIGURES_DIR / f"{stem}.html"
    fig.write_html(html_path)
    paths = {"html": html_path}
    if write_static:
        png_path = FIGURES_DIR / f"{stem}.png"
        svg_path = FIGURES_DIR / f"{stem}.svg"
        try:
            pio.write_image(fig, png_path, scale=2)
            paths["png"] = png_path
        except Exception:
            pass
        try:
            pio.write_image(fig, svg_path)
            paths["svg"] = svg_path
        except Exception:
            pass
    return paths


def safe_parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    parts = re.split(r"\s*,\s*", text)
    return [part.strip(" '\"") for part in parts if part.strip(" '\"")]


def parse_text_labels(series: pd.Series) -> pd.Series:
    return series.fillna("[]").apply(safe_parse_list)


def _merge_raw_and_cluster(raw: pd.DataFrame, cluster: pd.DataFrame) -> pd.DataFrame:
    overlapping = [col for col in cluster.columns if col in raw.columns and col != "Unnamed: 0"]
    cluster_only = cluster.drop(columns=overlapping, errors="ignore")
    if "Unnamed: 0" in raw.columns and "Unnamed: 0" in cluster_only.columns:
        same_order = raw["Unnamed: 0"].reset_index(drop=True).equals(
            cluster_only["Unnamed: 0"].reset_index(drop=True)
        )
        if same_order:
            return raw.merge(cluster_only, on="Unnamed: 0", how="left", validate="1:1")
    cluster_only = cluster_only.drop(columns=["Unnamed: 0"], errors="ignore")
    return pd.concat([raw.reset_index(drop=True), cluster_only.reset_index(drop=True)], axis=1)


def load_prepared_data() -> pd.DataFrame:
    raw = pd.read_csv(RAW_SURVEY_CSV, low_memory=False)
    cluster = pd.read_csv(CLUSTERING_CSV, low_memory=False)
    df = _merge_raw_and_cluster(raw, cluster)

    df["cluster_label"] = df["cluster3"].map(CLUSTER_LABELS)
    df["cluster_color"] = df["cluster_label"].map(CLUSTER_COLORS)
    df["region"] = df.apply(build_region, axis=1)
    df["role_group"] = np.where(df["role"].isin(["Tech", "MERL"]), "Tech/MERL", "Non-technical")
    df["infra_score"] = df[INFRA_COLS].fillna(0).sum(axis=1).astype(int)
    df["has_tech_or_merl"] = (
        df[["tech_person", "merl_person"]].fillna(0).max(axis=1).astype(int)
    )
    df["ai_use_count"] = df[USE_COLS_MAIN].fillna(0).sum(axis=1).astype(int)
    df["ai_want_count"] = df[WANT_COLS_MAIN].fillna(0).sum(axis=1).astype(int)
    df["want_use_gap"] = df["ai_want_count"] - df["ai_use_count"]
    df["wug_segment"] = np.select(
        [
            (df["want_use_gap"] >= 3) & (df["infra_score"] >= 3),
            (df["want_use_gap"] >= 3) & (df["infra_score"] < 3),
            (df["want_use_gap"] <= 1) & (df["infra_score"] >= 3),
            (df["want_use_gap"] <= 1) & (df["infra_score"] < 3),
        ],
        [
            "Activation-ready",
            "Foundation-needed",
            "Already-well-served",
            "Disengaged",
        ],
        default="Middle-ground",
    )

    ai_risk_parsed = parse_text_labels(df["ai_risk"])
    df["risk_list"] = ai_risk_parsed
    for risk, column in RISK_TO_COLUMN.items():
        df[column] = ai_risk_parsed.apply(lambda items: int(risk in items))
    df["risk_count"] = df[list(RISK_TO_COLUMN.values())].sum(axis=1).astype(int)

    org_labels_parsed = parse_text_labels(df["org_label"])
    df["org_label_list"] = org_labels_parsed
    df["primary_cause_area"] = org_labels_parsed.apply(lambda items: items[0] if items else "No Answer")

    risk_reward_map = {
        -1: "I don't understand AI enough to have a clear view",
        1: "Risks outweigh the benefits",
        2: "Risks somewhat outweigh the benefit",
        3: "Benefits and risks are equal",
        4: "Benefits somewhat outweigh the risks",
        5: "Benefits outweigh the risks",
    }
    if "ai_risk_reward" in df.columns:
        df["ai_risk_reward_label"] = df["ai_risk_reward"].map(risk_reward_map)

    return df


def compute_use_want_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for use_col, want_col in USE_WANT_PAIRS:
        label = AI_USE_COLS[use_col]
        use_rate = df[use_col].mean()
        want_rate = df[want_col].mean()
        rows.append(
            {
                "use_case": label,
                "use_rate": use_rate,
                "want_rate": want_rate,
                "gap": want_rate - use_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


def top_binary_labels(df: pd.DataFrame, columns: Iterable[str], label_map: dict[str, str], top_n: int = 3) -> list[str]:
    means = df[list(columns)].mean().sort_values(ascending=False)
    return [label_map[col] for col in means.head(top_n).index]


def ensure_nltk_resources() -> None:
    needed = ["stopwords", "wordnet", "omw-1.4"]
    for package in needed:
        try:
            if package == "stopwords":
                stopwords.words("english")
            elif package == "wordnet":
                WordNetLemmatizer().lemmatize("data")
        except LookupError:
            nltk_download(package, quiet=True)


def preprocess_text(text: str, extra_stopwords: Iterable[str] | None = None) -> str:
    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    if extra_stopwords:
        stop_words.update(word.lower() for word in extra_stopwords)
    lemmatizer = WordNetLemmatizer()
    tokens = re.findall(r"[a-zA-Z']+", str(text).lower())
    clean_tokens = []
    for token in tokens:
        if token in stop_words or len(token) < 3:
            continue
        clean_tokens.append(lemmatizer.lemmatize(token))
    return " ".join(clean_tokens)


def token_counter(texts: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        counter.update(re.findall(r"[a-zA-Z']+", str(text).lower()))
    return counter


def add_direct_labels_to_bars(ax: plt.Axes, fmt: str = "{:.1f}", padding: float = 2.0) -> None:
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        if height > width:
            x = patch.get_x() + width / 2
            y = patch.get_y() + height
            ax.text(x, y + padding, fmt.format(height), ha="center", va="bottom", fontsize=9)
        else:
            x = patch.get_x() + width
            y = patch.get_y() + height / 2
            ax.text(x + padding, y, fmt.format(width), ha="left", va="center", fontsize=9)


def notebook_setup() -> pd.DataFrame:
    ensure_output_dirs()
    apply_chart_style()
    np.random.seed(42)
    return load_prepared_data()
