"""
Stage 0 sanity check. Run this before any analysis.

Verifies:
  - Both CSVs load without error
  - Row counts match data dictionary (930 total)
  - Cluster counts match dictionary:
      raw survey cluster3: {1:523, 0:265, -1:142}
      clustering CSV cluster3: {1:523, 0:265, -1:142}
      clustering CSV cluster2: {1:524, 0:406}
  - No KeyError on columns we plan to use
  - Cluster naming confusion check: clustering CSV cluster2 vs cluster3

Usage:
    python scripts/00_sanity_check.py
"""

import sys
import json
from pathlib import Path

# Allow importing from scripts/ when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from columns import (
    RAW_SURVEY_CSV, CLUSTERING_CSV, FINDINGS_JSON,
    AI_USE_COLS, AI_WANT_COLS, INFRA_COLS, PALETTE,
    FIGURES_DIR, TABLES_DIR, STATS_DIR,
)

import pandas as pd
import numpy as np

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    marker = "OK" if condition else "!!"
    print(f"  [{marker}] {label}")
    if detail:
        print(f"       {detail}")
    results.append({"label": label, "status": status, "detail": detail})
    return condition


print("=" * 65)
print("STAGE 0 SANITY CHECK")
print("=" * 65)

# ── 1. Output directories exist ───────────────────────────────────────────
print("\n[1] Output directories")
for d in [FIGURES_DIR, TABLES_DIR, STATS_DIR]:
    check(str(d.relative_to(Path(__file__).parent.parent)), d.exists())

# ── 2. Load raw survey ────────────────────────────────────────────────────
print("\n[2] Raw survey CSV")
try:
    df = pd.read_csv(RAW_SURVEY_CSV, low_memory=False)
    check("Loads without error", True, f"Shape: {df.shape}")
    check("Row count == 930", df.shape[0] == 930, f"Got {df.shape[0]}")
    check("Column count >= 40", df.shape[1] >= 40, f"Got {df.shape[1]}")
except Exception as e:
    check("Loads without error", False, str(e))
    print("CRITICAL: Cannot load raw survey. Stopping.")
    sys.exit(1)

# ── 3. Load clustering CSV ────────────────────────────────────────────────
print("\n[3] Clustering CSV")
try:
    dfc = pd.read_csv(CLUSTERING_CSV, low_memory=False)
    check("Loads without error", True, f"Shape: {dfc.shape}")
    check("Row count == 930", dfc.shape[0] == 930, f"Got {dfc.shape[0]}")
except Exception as e:
    check("Loads without error", False, str(e))
    print("CRITICAL: Cannot load clustering CSV. Stopping.")
    sys.exit(1)

# ── 4. Cluster verification ───────────────────────────────────────────────
print("\n[4] Cluster label verification")
if "cluster3" in df.columns:
    vc = df["cluster3"].value_counts().to_dict()
    print(f"       raw survey cluster3 counts: {vc}")
    check("Consumers (cluster3==1) ~523", 500 <= vc.get(1, 0) <= 540,
          f"Got {vc.get(1, 0)}")
    check("Late Adopters (cluster3==0) ~265", 240 <= vc.get(0, 0) <= 290,
          f"Got {vc.get(0, 0)}")
    check("Skeptics (cluster3==-1) ~142", 120 <= vc.get(-1, 0) <= 160,
          f"Got {vc.get(-1, 0)}")
else:
    check("cluster3 exists in raw survey", False)

# Clustering CSV: cluster3 should be the 3-group HDBSCAN model
if "cluster3" in dfc.columns:
    vc3 = dfc["cluster3"].value_counts().to_dict()
    print(f"       clustering CSV cluster3 counts (HDBSCAN 3-group): {vc3}")
    check(
        "clustering CSV cluster3 matches raw survey cluster3 counts",
        vc3 == vc,
        f"raw={vc}, clustering={vc3}",
    )

# Clustering CSV: cluster2 should be the K-means 2-group model
if "cluster2" in dfc.columns:
    vc2 = dfc["cluster2"].value_counts().to_dict()
    print(f"       clustering CSV cluster2 counts (K-means 2-group): {vc2}")
    check(
        "clustering CSV cluster2 is 2-group (not 3)",
        len(vc2) == 2,
        f"Unique values: {list(vc2.keys())}",
    )
else:
    print("  [WW] cluster2 not found in clustering CSV")

# ── 5. Required columns in raw survey ────────────────────────────────────
print("\n[5] Required columns in raw survey")
REQUIRED_RAW = [
    "cluster3", "continent", "role", "org_size", "global_north_south",
    "ref", "person_ai_comfort_raw", "collab_feasibility_raw",
    "org_years_raw", "person_org_years_raw", "ai_opentext", "org_opentext",
    "org_label", "in_india", "india_state", "india_rural_urban",
    "tech_person", "merl_person", "cloud_storage", "data_use_policy",
    "org_agreements", "ai_risk_reward", "global_north_south_int",
    "org_size_int",
]
missing_raw = [c for c in REQUIRED_RAW if c not in df.columns]
check("All required raw columns present", len(missing_raw) == 0,
      f"Missing: {missing_raw}" if missing_raw else "")

# ── 6. Required columns in clustering CSV ────────────────────────────────
print("\n[6] Required columns in clustering CSV")
REQUIRED_CLUST = list(AI_USE_COLS.keys()) + list(AI_WANT_COLS.keys())
# Filter out 'Not Using AI' from clustering CSV check (it may have slightly different naming)
missing_clust = [c for c in REQUIRED_CLUST if c not in dfc.columns]
if missing_clust:
    print(f"       Missing clustering cols: {missing_clust}")
    check("All AI use/want binary cols present", False,
          f"Missing {len(missing_clust)} cols")
else:
    check("All AI use/want binary cols present", True)

# ── 7. Key value counts ───────────────────────────────────────────────────
print("\n[7] Key categorical value counts")
for col in ["continent", "role", "org_size", "global_north_south", "ref"]:
    if col in df.columns:
        vc = df[col].value_counts()
        print(f"       {col}: {vc.to_dict()}")

# Verify official report claims
gn_pct = (df["global_north_south"] == "N").mean() * 100
check(
    "Global North ~63%, Global South ~37%",
    55 <= gn_pct <= 70,
    f"Global North: {gn_pct:.1f}%",
)

small_staff_pct = (df["org_size"].isin(["0-5", "6-15"])).mean() * 100
check(
    "~63% organizations have 15 or fewer staff",
    55 <= small_staff_pct <= 72,
    f"Got {small_staff_pct:.1f}%",
)

# ── 8. Missing data overview ──────────────────────────────────────────────
print("\n[8] Top 15 columns by missing data (raw survey)")
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(missing_pct.head(15).to_string())

# ── 9. findings.json ─────────────────────────────────────────────────────
print("\n[9] findings.json")
if FINDINGS_JSON.exists():
    with open(FINDINGS_JSON) as f:
        findings = json.load(f)
    check("findings.json exists and is a list", isinstance(findings, list),
          f"Contains {len(findings)} entries")
else:
    check("findings.json exists", False, "Run Stage 0 setup to create it")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
fails = [r for r in results if r["status"] == FAIL]
print(f"SUMMARY: {len(results) - len(fails)}/{len(results)} checks passed.")
if fails:
    print("FAILED checks:")
    for f in fails:
        print(f"  - {f['label']}: {f['detail']}")
else:
    print("All checks passed. Ready for Stage 1.")
print("=" * 65)
