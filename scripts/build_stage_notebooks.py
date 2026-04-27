from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP_CELL = """
import sys
import warnings
from pathlib import Path
from IPython.display import display

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analysis_utils import notebook_setup

df = notebook_setup()
print(f"Loaded prepared analysis frame with shape: {df.shape}")
"""


NOTEBOOK_SPECS = {
    "02_hypothesis_tests.ipynb": [
        md(
            """
            # Stage 2: Statistical Hypothesis Testing

            This notebook moves from description to inference. Each section saves its tables, charts, and stats files to disk before we use the results anywhere else.
            """
        ),
        code(SETUP_CELL),
        md("## Section 2.1 - Tech and MERL staff are more risk-aware"),
        code(
            """
            from stage2_hypothesis_tests import run_risk_role_tests

            risk_results = run_risk_role_tests(df)
            display(risk_results["risk_tests"])
            """
        ),
        md("## Section 2.2 - Global South is not one story"),
        code(
            """
            from stage2_hypothesis_tests import run_region_tests

            region_results = run_region_tests(df)
            display(region_results["region_tests"])
            display(region_results["region_means"])
            """
        ),
        md("## Section 2.3 - Data infrastructure predicts readiness better than raw size"),
        code(
            """
            from stage2_hypothesis_tests import run_infrastructure_models

            infra_results = run_infrastructure_models(df)
            display(infra_results["logit_results"])
            display(infra_results["ols_results"])
            """
        ),
        md("## Section 2.4 - Test the 15-staff threshold"),
        code(
            """
            from stage2_hypothesis_tests import run_threshold_analysis

            threshold_results = run_threshold_analysis(df)
            display(threshold_results["threshold_summary"])
            """
        ),
        md("## Section 2.5 - Comfort is related to use and demand, but not causally"),
        code(
            """
            from stage2_hypothesis_tests import run_comfort_correlations

            correlation_results = run_comfort_correlations(df)
            display(correlation_results["correlations"])
            """
        ),
    ],
    "03_clustering_deep_dive.ipynb": [
        md(
            """
            # Stage 3: Clustering Deep Dive

            This notebook checks whether we can recreate the published clustering logic, then asks whether a more detailed persona view helps us tell a sharper story.
            """
        ),
        code(SETUP_CELL),
        md("## Section 3.1 - Reproduce the official UMAP plus HDBSCAN structure"),
        code(
            """
            from stage3_clustering_deep_dive import run_reproduction

            reproduction_results = run_reproduction(df)
            display(reproduction_results["reproduction_grid"].head(10))
            """
        ),
        md("## Section 3.2 - Build and profile an expanded persona set"),
        code(
            """
            from stage3_clustering_deep_dive import run_persona_analysis

            persona_results = run_persona_analysis(df)
            print(f"Chosen persona count: {persona_results['chosen_k']}")
            display(persona_results["silhouette"])
            display(persona_results["persona_summary"])
            """
        ),
    ],
    "04_text_analysis.ipynb": [
        md(
            """
            # Stage 4: Open Text Analysis

            The open-text responses let us test whether hopes and fears coexist, how sentiment changes by cluster, and which quotes are strong enough to carry the deck.
            """
        ),
        code(SETUP_CELL),
        md("## Section 4.1 - Clean and prepare the text"),
        code(
            """
            from stage4_text_analysis import run_preprocessing

            text_results = run_preprocessing(df)
            display(text_results["text_df"].head())
            """
        ),
        md("## Section 4.2 - Sentiment analysis"),
        code(
            """
            from stage4_text_analysis import run_sentiment_analysis

            sentiment_results = run_sentiment_analysis(text_results["text_df"])
            display(sentiment_results["sentiment_by_cluster"])
            display(sentiment_results["sentiment_by_region"].head(10))
            """
        ),
        md("## Section 4.3 - Rule-based hope versus fear classification"),
        code(
            """
            from stage4_text_analysis import run_hope_fear_analysis

            hope_fear_results = run_hope_fear_analysis(sentiment_results["sentiment_df"])
            display(hope_fear_results["hope_fear_distribution"])
            """
        ),
        md("## Section 4.4 and 4.5 - Topic themes, verified quotes, and optional word clouds"),
        code(
            """
            from stage4_text_analysis import run_topic_and_quote_analysis

            topic_results = run_topic_and_quote_analysis(hope_fear_results["classified_text"])
            display(topic_results["topic_summary"])
            display(topic_results["quotes"])
            """
        ),
    ],
    "05_geo_cause_analysis.ipynb": [
        md(
            """
            # Stage 5: Geographic and Cause-Area Deep Dive

            This notebook pushes beyond the simple Global North versus Global South split and looks at who is actually over- or under-represented inside each readiness cluster.
            """
        ),
        code(SETUP_CELL),
        md("## Section 5.1 - Regional readiness scorecard"),
        code(
            """
            from stage5_geo_cause_analysis import run_regional_scorecard

            regional_results = run_regional_scorecard(df)
            display(regional_results["regional_scorecard"])
            """
        ),
        md("## Section 5.2 - Cause area by cluster"),
        code(
            """
            from stage5_geo_cause_analysis import run_cause_area_analysis

            cause_results = run_cause_area_analysis(df)
            display(cause_results["cause_area_shares"])
            display(cause_results["cause_area_metrics"].head(15))
            """
        ),
        md("## Section 5.3 - India rural versus urban"),
        code(
            """
            from stage5_geo_cause_analysis import run_india_rural_urban

            india_results = run_india_rural_urban(df)
            display(india_results["india_rural_urban"])
            """
        ),
    ],
    "06_want_use_gap_index.ipynb": [
        md(
            """
            # Stage 6: Want-Use Gap Index

            This notebook turns the gap between AI demand and actual use into a simple metric we can explain on one slide and defend in Q&A.
            """
        ),
        code(SETUP_CELL),
        md("## Section 6.1 - Compute the Want-Use Gap and rank unmet needs"),
        code(
            """
            from stage6_want_use_gap_index import run_wug_core

            wug_results = run_wug_core(df)
            display(wug_results["gap_table"])
            display(wug_results["wug_by_cluster"])
            """
        ),
        md("## Section 6.2 - Turn the WUG into a 2x2 intervention framework"),
        code(
            """
            from stage6_want_use_gap_index import run_segmentation

            segment_results = run_segmentation(df)
            display(segment_results["segment_summary"])
            display(segment_results["segment_matrix"])
            """
        ),
        md("## Section 6.3 - Build a memorable signature visualization"),
        code(
            """
            from stage6_want_use_gap_index import run_signature_sankey

            sankey_results = run_signature_sankey(df)
            sankey_results["signature_sankey_paths"]
            """
        ),
    ],
    "07_predictive_model.ipynb": [
        md(
            """
            # Stage 7: Predictive Model

            This notebook uses interpretable models to rank what predicts AI readiness. The point is explanation, not hype.
            """
        ),
        code(SETUP_CELL),
        md("## Section 7.1 - Cross-validated model metrics"),
        code(
            """
            from stage7_predictive_model import run_model_metrics

            model_results = run_model_metrics(df)
            display(model_results["metrics_df"])
            """
        ),
        md("## Section 7.2 - Random forest feature importance and SHAP"),
        code(
            """
            from stage7_predictive_model import run_feature_importance

            importance_results = run_feature_importance(model_results["model_df"], model_results["pipelines"])
            display(importance_results["rf_importance"].head(15))
            if importance_results["shap_importance"] is not None:
                display(importance_results["shap_importance"].head(15))
            """
        ),
        md("## Section 7.3 - Logistic regression explanation"),
        code(
            """
            from stage7_predictive_model import run_logistic_explanation

            logit_results = run_logistic_explanation(model_results["model_df"])
            display(logit_results["logit_coefficients"].head(15))
            """
        ),
    ],
}


def build_notebooks() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in NOTEBOOK_SPECS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb["metadata"]["language_info"] = {
            "name": "python",
            "version": "3.11",
        }
        path = NOTEBOOK_DIR / filename
        with path.open("w", encoding="utf-8") as handle:
            nbf.write(nb, handle)
        print(f"Wrote {path}")


if __name__ == "__main__":
    build_notebooks()
