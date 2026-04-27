"""
Single source of truth for column names and human-readable labels.

CRITICAL CLUSTER ENCODING NOTE (verified by 00_sanity_check.py):
  Both the raw survey CSV and the clustering CSV use 'cluster3' for the
  HDBSCAN 3-group model. They are CONSISTENT.

  cluster3 (both files):  HDBSCAN 3-group
    1  = AI Consumers   (n=523, largest)
    0  = Late Adopters  (n=265, middle)
   -1  = AI Skeptics    (n=142, smallest)

  cluster2 (clustering CSV only):  K-means 2-group model
    1  = larger group   (n=524)
    0  = smaller group  (n=406)

  NOTE: The data dictionary README had cluster2/cluster3 labels swapped in
  its description. Trust the actual data (verified by sanity check), not the
  README description. Always use cluster3 for the 3-cluster story.
"""

from pathlib import Path

# ── Project root (one level up from scripts/) ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
FIGURES_DIR  = OUTPUTS_DIR / "figures"
TABLES_DIR   = OUTPUTS_DIR / "tables"
STATS_DIR    = OUTPUTS_DIR / "stats"

# ── Data file paths ─────────────────────────────────────────────────────────
RAW_SURVEY_CSV     = DATA_DIR / "ai_survey_results_2024_n=930.csv"
CLUSTERING_CSV     = DATA_DIR / "ai_survey_normalized_clustering_data.csv"
FINDINGS_JSON      = PROJECT_ROOT / "findings.json"

# ── Cluster labels (for the 3-cluster HDBSCAN model) ────────────────────────
CLUSTER_LABELS = {
    1:  "AI Consumers",
    0:  "Late Adopters",
    -1: "AI Skeptics",
}

CLUSTER_COLORS = {
    "AI Consumers":  "#1F4E79",   # deep blue  – trust / established
    "Late Adopters": "#E8743B",   # warm coral – warmth / in-progress
    "AI Skeptics":   "#6B6B6B",   # warm gray  – withdrawn
}

# ── AI Use columns (binary flags, clustering CSV) ────────────────────────────
AI_USE_COLS = {
    "[U] Generat":                     "Generative AI",
    "[U] Ask":                         "Chatbot Q&A",
    "[U] Organi":                      "Organizing Data",
    "[U] Interpret":                   "Interpreting Data",
    "[U] Predict":                     "Predictive AI",
    "[U] Translat":                    "Translate / Transcribe",
    "[U] Assist":                      "Virtual Assistant",
    "[U] Other":                       "Other AI Use",
    "[U] I am not currently using AI": "Not Using AI",
}

# ── AI Want columns (binary flags, clustering CSV) ───────────────────────────
AI_WANT_COLS = {
    "[W] Generat":          "Generative AI",
    "[W] Ask":              "Chatbot Q&A",
    "[W] Organi":           "Organizing Data",
    "[W] Interpret":        "Interpreting Data",
    "[W] Predict":          "Predictive AI",
    "[W] Translat":         "Translate / Transcribe",
    "[W] Assist":           "Virtual Assistant",
    "[W] Other":            "Other AI Want",
    "[W] We don't know yet!": "Don't Know Yet",
}

# ── Data-practice columns (binary flags, clustering CSV) ─────────────────────
DATA_COLS = {
    "[D] Our staff manually fill out spreadsheets (or other tabular data) from notes and observations.":
        "Manual Spreadsheets",
    "[D] Our staff uses software to collect data about people, relationships, prospects, etc.":
        "CRM / Software",
    "[D] We also retained the original audio or video that accompanies the transcripts and tabular data.":
        "Audio / Video Records",
    "[D] We collect data using devices, such as phones, tablets, or computers, and people submit/update information electronically.":
        "Digital Device Collection",
    "[D] We have at least a hundred transcripts or records of interviews, testimonials, meetings, proceedings, surveys, or similar.":
        "100+ Transcripts / Records",
}

# ── Infrastructure flags (raw survey + clustering CSV share these) ────────────
INFRA_COLS = ["tech_person", "merl_person", "cloud_storage", "data_use_policy", "org_agreements"]

INFRA_LABELS = {
    "tech_person":     "Has Tech Person",
    "merl_person":     "Has MERL Person",
    "cloud_storage":   "Uses Cloud Storage",
    "data_use_policy": "Has Data-Use Policy",
    "org_agreements":  "Has Data Agreements",
}

# ── AI Risk categories (from raw survey 'ai_risk' multi-select) ──────────────
AI_RISK_LABELS = {
    "Decisions based on biased AI models":
        "Biased AI Decisions",
    "AI-related data breaches":
        "Data Breaches",
    "Replacing workers with AI":
        "Worker Replacement",
    "Increasing inequity, if lower-capacity organizations are unable to adopt it, or if AI bias harms groups":
        "Increasing Inequity",
    "Plagiarism, violating copyrights, and/or losing intellectual property":
        "IP / Plagiarism",
    "(Over) Dependency on commercial AI products":
        "Commercial AI Dependency",
    "Environmental impact of using AI":
        "Environmental Impact",
}

# ── Risk-reward categories ────────────────────────────────────────────────────
RISK_REWARD_ORDER = [
    "Risks outweigh the benefits",
    "Risks somewhat outweigh the benefit",
    "Benefits and risks are equal",
    "Benefits somewhat outweigh the risks",
    "Benefits outweigh the risks",
    "I don't understand AI enough to have a clear view",
]

# ── Role groupings ────────────────────────────────────────────────────────────
TECH_ROLES    = ["Tech", "MERL"]
NON_TECH_ROLES = ["Leader", "Fundraiser", "Governance", "Admin", "Comms", "Programs", "Individual", "Other"]

# ── Org size order for plots ──────────────────────────────────────────────────
ORG_SIZE_ORDER = ["0-5", "6-15", "16-30", "31-60", "61-120", "121+"]

# ── Region mapping: continent → finer region ─────────────────────────────────
def build_region(row):
    """Map continent + India flag to a finer 6-region variable."""
    continent = row.get("continent", "")
    in_india   = row.get("in_india", 0)
    if continent == "Asia" and in_india == 1:
        return "India"
    elif continent == "Asia":
        return "Other Asia"
    elif continent == "Africa":
        return "Africa"
    elif continent == "North America":
        return "North America"
    elif continent == "Europe":
        return "Europe"
    elif continent == "South America":
        return "Latin America"
    elif continent == "Australia":
        return "Australia / Pacific"
    return "Unknown"

REGION_ORDER = [
    "North America", "Europe", "India", "Africa",
    "Latin America", "Other Asia", "Australia / Pacific",
]

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":    "#1F4E79",
    "accent1":    "#E8743B",
    "accent2":    "#3CB4AC",
    "neutral":    "#6B6B6B",
    "background": "#F7F4EE",
}

# ── Chart defaults (apply in every notebook) ─────────────────────────────────
import matplotlib as mpl

def apply_chart_style():
    """Call once per notebook to enforce consistent chart styling."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "figure.facecolor":  PALETTE["background"],
        "axes.facecolor":    PALETTE["background"],
        "font.size":         11,
        "axes.titlesize":    15,
        "axes.labelsize":    11,
    })
