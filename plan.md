## **0. CRITICAL META-INSTRUCTIONS (Read First, Read Twice)**

**Hey Claude Code, before you do anything, read this section like your life depends on it.**

This is a Data Analytics case competition for the Bay Area, hosted in partnership with the Gates Foundation, OECD, and GivingTuesday Data Commons (GTDC). The dataset is GivingTuesday's 2024 AI Readiness Survey of 930 nonprofits worldwide. The official report from GivingTuesday already exists at https://ai.givingtuesday.org/ai-readiness-report-2024/. Our job is **NOT** to recreate their report. Our job is to find **novel, non-obvious insights** that go beyond what they already published, and present them in a way that wows judges.

### Hard rules you must follow:

1. **NEVER hallucinate numbers.** Every single statistic, percentage, count, correlation, or finding in the final deliverable must come from a Python computation you actually ran on the dataset. If you write "67% of nonprofits..." in a slide, you must be able to point to the exact line of code that produced 67%. No exceptions.
2. **Run the analysis FIRST, then write the narrative.** Do not draft slides with placeholder findings and fill them in later. Every slide claim must be grounded in a `df.groupby(...).agg(...)` or equivalent that already executed successfully.
3. **Save every chart and every stats output to disk** with descriptive filenames (e.g., `fig_03_cluster_risk_heatmap.png`, `stats_03_chi2_global_north_south_ai_risk.json`). This makes the work auditable.
4. **Build the analysis in atomic notebook cells (or scripts).** One question per cell. Print summary stats. Do not chain 50 lines of pandas without inspecting intermediate outputs.
5. **Cycle for every finding:** plan → analyze → read code → implement → review → test (sanity-check) → bug-fix → log finding to `findings.json`. Only after a finding is logged and verified does it become eligible for the slide deck.
6. **The user is not an expert.** When you find something interesting, explain in plain English in a comment what the number means and why it matters before moving on.
7. **Voice and style:** Conversational, classmate-to-classmate, sincere. No em dashes. No corporate buzzwords. No "delve," "leverage," "tapestry," "navigate the complexities of." Write the way a thoughtful undergrad presenting at a Bay Area case comp would actually talk.
8. **Visualizations:** A single, restrained color palette throughout. The GivingTuesday brand uses deep blues, warm coral/orange, and creamy off-white. We'll use a similar palette so the deck feels cohesive and professional, not like a Plotly demo.
9. **If something disagrees with the official GivingTuesday report, that is GOOD.** Flag it explicitly. Judges love when you challenge prior work with evidence.

---

## **1. THE BIG IDEA / THESIS (This is what we're selling)**

Most AI-adoption stories are about hype. The GivingTuesday official report already told the standard one: 56% are AI Consumers, 28% are Late Adopters, 15% are Skeptics. Capacity does not predict readiness. The 15-staff threshold matters. Generative AI dominates current usage. Data organization tools are the biggest unmet need.

We need to go **further**. Our thesis, which we will spend the deck proving:

> **"AI readiness in the nonprofit sector is not a technology problem. It is a data infrastructure problem dressed up as a technology problem, and the gap between what nonprofits want from AI and what they're set up to actually use is widening into a chasm. The sector is sitting at Gartner's 'trough of disillusionment' for some, and 'peak of inflated expectations' for others, simultaneously. The path forward is not more AI tools. It is helping nonprofits build the data plumbing first."**

Sub-thesis statements we will prove with data:
- **(A) The AI Want-Use Gap is the real story.** The biggest unmet need is not generative AI. It's data organization. We will quantify this gap per use-case and show which organizations have the worst gaps.
- **(B) AI risk perception flips with technical knowledge.** Tech and MERL staff are MORE worried about AI risks than non-technical staff, not less. This challenges the "if only people understood AI better, they'd trust it more" narrative.
- **(C) The Global South is not a monolith.** India is one story. Africa is another. Lumping them together (as the official report largely does) hides important differences. We will show the cause-area-by-region heatmap that nobody else has shown.
- **(D) Data infrastructure is the gating factor.** Organizations with data-use policies, cloud storage, AND a tech-or-MERL person are dramatically more AI-ready than those missing any one piece. We can quantify the marginal lift of each.
- **(E) The 15-staff threshold is real but not magic.** It's a proxy for hiring patterns, not size itself. We will show that two organizations of the same size can have wildly different readiness based on whether they hired tech/MERL.
- **(F) Hopes and fears are not opposites.** Open-text analysis will show that the same person often expresses both, and the ratio shifts by cluster, region, and role.

This thesis lets us **disagree productively** with the official GTDC narrative while still respecting their work.

---

## **2. STAGE-BY-STAGE IMPLEMENTATION PLAN**

### **STAGE 0: Environment Setup and Sanity Check (~30 min)**

**Goal:** Make sure we can load the data and have all the tools.

1. Create a project structure:
   ```
   /project_root
     /data                     # raw CSVs/parquets/pkls (read-only)
     /notebooks                # Jupyter notebooks, one per analysis stage
     /scripts                  # reusable .py modules
     /outputs
       /figures                # all .png and .svg charts
       /tables                 # all .csv summary tables
       /stats                  # all .json statistical test outputs
     /deliverables
       /slides                 # the final deck (HTML or pptx)
       /github_readme          # README.md for the repo
     plan.md                   # this file
     findings.json             # running ledger of every verified insight
     requirements.txt
   ```

2. `requirements.txt` must include: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `statsmodels`, `umap-learn`, `hdbscan`, `wordcloud`, `nltk`, `vaderSentiment`, `pyarrow` (for parquet).

3. **Sanity check script** (`scripts/00_sanity_check.py`):
   - Load both CSVs.
   - Print `df.shape` for both. The full survey should be `(930, ~80)`. The clustering set should be `(930, ~40)`.
   - Print `df.columns.tolist()` for both.
   - Print `df.head(3).T` (transpose, more readable for wide data).
   - Print `df.info()` and `df.isnull().sum().sort_values(ascending=False).head(20)`.
   - Print value counts for the key categorical columns: `cluster3`, `continent`, `role`, `org_size`, `global_north_south`, `ref`.
   - **Verify the cluster split matches the data dictionary**: `{1: ~523 AI Consumers, 0: ~265 Late Adopters, -1: ~142 AI Skeptics}`. If anything is off, stop and flag it.

4. **Defensive data hygiene:** Some columns have the `[U]` (using), `[W]` (want), `[D]` (data) prefixes. Build a small dict in `scripts/columns.py` that maps these to human-readable labels and groups them. Example:
   ```python
   AI_USE_COLS = {
       '[U] Generat': 'Generative AI',
       '[U] Ask': 'Chatbot Q&A',
       '[U] Organi': 'Organizing data',
       '[U] Interpret': 'Interpreting data',
       '[U] Predict': 'Predictive AI',
       '[U] Translat': 'Translate/Transcribe',
       '[U] Assist': 'Virtual Assistant',
       '[U] Other': 'Other',
       '[U] I am not currently using AI': 'Not using AI',
   }
   AI_WANT_COLS = { ... } # mirror structure
   DATA_COLS = { ... }
   ```
   This dict is the single source of truth for labels in every chart. No more typing column names by hand in plotting code.

**Cycle gate before moving to Stage 1:**
- [ ] Both files load without error.
- [ ] Row counts match dictionary (930 total).
- [ ] Cluster counts match dictionary (~523/265/142).
- [ ] No KeyError on any column we plan to use.
- [ ] `findings.json` exists as an empty list.

---

### **STAGE 1: Exploratory Data Analysis, the Honest Way (~3 hours)**

**Goal:** Understand the dataset deeply before forming opinions. We will NOT skip this. The whole point of this stage is to surface surprises that didn't make it into the official report.

Create `notebooks/01_eda.ipynb`. Structure it in clear sections:

#### 1.1 — Sample composition

- Distribution of respondents by `continent`, `org_size`, `role`, `global_north_south`, `ref` (survey source).
- Cross-tabs: continent × org_size, continent × role, role × org_size.
- **Key check:** Verify the official report's claim that "63% of organizations have 15 or fewer staff." Compute it ourselves.
- **Key check:** Verify "63% Global North, 37% Global South."
- Histogram of `org_years_raw` and `person_org_years_raw`.
- **Cluster distribution by source (`ref`).** If the GivingTuesday main list dominates one cluster, that's a sampling-bias finding worth flagging.

#### 1.2 — Cluster validation

- Print the mean of every relevant column grouped by `cluster3`. This should reproduce the data dictionary's claim that cluster 1 is the "AI Consumers" (high data use, high AI use, high AI want), cluster 0 is "Late Adopters" (medium data, low AI use, medium-high AI want), and cluster -1 is "AI Skeptics" (low data, some AI use, low AI want).
- **If our reproduction doesn't match the official narrative, FLAG IT.** The dictionary explicitly says: "Largest group are AI Consumers; middle size group are Late AI Adopters; smallest group are AI Skeptics." We need to verify this maps cleanly to cluster IDs 1, 0, -1.
- For each cluster, compute mean comfort (`person_ai_comfort_raw`), mean collaboration feasibility (`collab_feasibility_raw`), and percent in each global region.

#### 1.3 — The AI Use vs AI Want gap (THIS IS GOLD)

This is one of our headline findings. For each AI use-case (8 total: Generative, Ask, Organize, Interpret, Predict, Translate, Assist, Other):

```
gap[use_case] = mean(want[use_case]) - mean(use[use_case])
```

- Sort use-cases by gap descending. The largest gap is the biggest unmet need.
- Compute this gap separately for each cluster, each region, each org_size bucket, and each role.
- **Hypothesis to test:** AI Skeptics will have the smallest gap because they don't want more. AI Consumers will have a moderate gap because they're already using a lot. Late Adopters will have the LARGEST gap because they want the most but use the least. **If the data confirms this, it's a powerful slide.** If not, it's an even more interesting slide.

#### 1.4 — Risk perception breakdown

The `ai_risk` column is a multi-select. Parse it. The risks are:
- Decisions based on biased AI models
- AI-related data breaches
- Replacing workers with AI
- Increasing inequity (if lower-capacity orgs can't adopt)
- Plagiarism / IP loss
- Over-dependency on commercial AI
- Environmental impact

For each risk:
- Overall % concerned.
- % concerned by cluster.
- % concerned by role (especially `Tech` vs non-tech).
- % concerned by Global North vs Global South.
- % concerned by org_size.

**Key hypothesis (sub-thesis B):** Tech and MERL staff are MORE worried about specific risks than non-technical leaders. The official report hints at this but doesn't quantify it. We will. Use a chi-squared test or Fisher's exact test for each risk × role pair and report p-values.

**Citation hook for slide:** GivingTuesday report explicitly states "those with a more technical background tend to be more wary of AI risks than other organizational personnel in general." (Source: https://ai.givingtuesday.org/ai-readiness-report-2024/, "Key takeaways" section.) We will quantify and visualize what they only stated in prose.

#### 1.5 — Risk-reward sentiment

- Distribution of `ai_risk_reward` (6 categories from "Risks outweigh benefits" to "Benefits outweigh risks" plus "I don't understand").
- Cross-tab with cluster, role, region, comfort score.
- **Key check:** The official report says "half our sample either doesn't know how to evaluate the risks of AI, or thinks the risks and benefits are roughly equal." Verify this 50% claim.

#### 1.6 — Data infrastructure inventory

For each binary infrastructure variable (`tech_person`, `merl_person`, `cloud_storage`, `data_use_policy`, `org_agreements`):
- Overall % yes.
- % by cluster, region, org_size.
- **Build a "Data Infrastructure Score"** = sum of these 5 binary flags (range 0 to 5). Plot histogram, then plot mean AI use intensity by infrastructure score. This becomes a CORE slide for sub-thesis D.

#### 1.7 — Open text exploration

- Length distribution of `org_opentext` and `ai_opentext` responses.
- Number of respondents who answered the AI hopes/fears question (`ai_opentext`).
- Quick word frequency. Top 50 words after stopword removal.
- Save the full open text as a separate dataframe so we can work with it in Stage 4.

**Cycle gate before moving to Stage 2:**
- [ ] Every claim in the official GTDC executive summary is either confirmed or flagged as different in our data.
- [ ] We have at least 3 surprises (findings the official report did not emphasize).
- [ ] Every chart has a saved `.png` file in `/outputs/figures/`.
- [ ] Every numeric finding is logged to `findings.json` with the format:
  ```json
  {
    "id": "F001",
    "claim": "AI Skeptics show the smallest want-use gap (X.XX), while Late Adopters show the largest (Y.YY)",
    "value": 0.42,
    "computed_in": "notebooks/01_eda.ipynb",
    "cell_id": "1.3",
    "code_snippet": "df.groupby('cluster3')[want_cols].mean() - df.groupby('cluster3')[use_cols].mean()",
    "verified": true
  }
  ```

---

### **STAGE 2: Statistical Hypothesis Testing (~2 hours)**

**Goal:** Move from descriptive to inferential. Anything we say is "significant" must have a p-value behind it.

Create `notebooks/02_hypothesis_tests.ipynb`.

#### 2.1 — Test sub-thesis B: Tech staff are MORE risk-averse

For each risk category:
- 2×2 contingency table: `role == 'Tech'` (or 'Tech' or 'MERL') vs `concerned about risk X` (1/0).
- Chi-squared test or Fisher's exact (for small cells).
- Effect size: odds ratio or Cramér's V.
- Store all p-values in a results table. Apply Holm-Bonferroni correction for multiple comparisons (we'll do 7 tests).

#### 2.2 — Test sub-thesis C: Global South ≠ India ≠ Africa

The data has `continent`, `af_region`, `india_state`, and `india_rural_urban`. Build a finer-grained region variable:
- North America
- Europe
- India (with sub-breakdowns where sample allows)
- Africa (with sub-region where sample allows)
- Latin America
- Other Asia

Then run ANOVA or Kruskal-Wallis on:
- `person_ai_comfort_raw` by region
- `collab_feasibility_raw` by region
- Mean number of AI use-cases used by region
- Mean number of risk concerns by region

Report effect sizes. **The big finding we expect:** intra-Global-South variance > Global-North-vs-South variance. If we find this, it directly challenges how the official report handles regional differences.

#### 2.3 — Test sub-thesis D: Data infrastructure is the gating factor

Logistic regression. Outcome: `is_AI_consumer = (cluster3 == 1)`.
Predictors:
- `org_size_int` (size)
- `org_years_raw` (age)
- `regionality`
- `global_north_south_int`
- `tech_person`
- `merl_person`
- `cloud_storage`
- `data_use_policy`
- `org_agreements`
- Optionally: interaction terms

Use `statsmodels.Logit` so we get coefficients with p-values and standard errors. Report odds ratios.

**Expected finding:** Once you control for tech/MERL person and data infrastructure, raw size and age have weak or non-significant effects. This QUANTIFIES the official report's qualitative claim that "organizational capacity is not a good predictor of AI readiness."

Also fit a simpler model: AI use intensity (continuous: count of AI uses) as a function of infrastructure score. Use OLS. Report R² and coefficient.

#### 2.4 — The 15-staff threshold

The official report claims a 15-staff threshold. Test it:
- Plot `% with tech_person OR merl_person` against org_size buckets.
- Look for a kink or jump.
- Fit a piecewise/segmented regression if the visual confirms.
- Report where the actual threshold falls in our data.

#### 2.5 — Comfort vs use: does comfort drive use, or does use drive comfort?

Two correlations:
- `person_ai_comfort` × number of AI uses
- `person_ai_comfort` × number of AI wants

Comfort is likely correlated with both. Important caveat: we cannot establish causation from cross-sectional survey data, and we should say so explicitly. But the strength of the correlation is interesting.

**Cycle gate:**
- [ ] Every test has assumptions checked (e.g., expected cell counts ≥5 for chi-squared, or use Fisher).
- [ ] Multiple comparisons corrections applied where appropriate.
- [ ] Results saved to `/outputs/stats/*.json`.
- [ ] Each statistically significant finding logged to `findings.json` with p-value and effect size.

---

### **STAGE 3: Re-clustering and Persona Deepening (~2 hours)**

**Goal:** We have the official 3-cluster solution from the dictionary. Let's see if a different lens reveals different personas.

Create `notebooks/03_clustering_deep_dive.ipynb`.

#### 3.1 — Reproduce the official clustering on our own

Use the clustering parquet/CSV. Run UMAP + HDBSCAN with reasonable params. Confirm we reproduce roughly the same 3-cluster structure. This is a methodological gut-check.

#### 3.2 — A 5-persona alternative

The official 3 clusters are coarse. Try a 5- or 6-cluster solution using K-means on the clustering data. Look at the cluster means. We might find personas like:

- **The Eager-but-Empty** — high want, low data infrastructure, low current use. (Likely a slice of "Late Adopters.")
- **The Quiet Builders** — medium use, high data infrastructure, low public-facing AI talk. (Likely a slice of "Consumers.")
- **The Burned** — tried AI, were disappointed, stopped. (Likely the "Skeptics" but more nuanced.)
- **The Generative-Only** — uses ChatGPT, hasn't moved past it.
- **The Prediction-Curious** — wants predictive AI even without data infrastructure to support it. (A risky group.)

Profile each persona using the same dimensions: data use, AI use, AI want, region, role, comfort, risk concerns. Give each persona a memorable name and a one-sentence story.

This is a STORYTELLING device. Three clusters are forgettable. Five clusters with stories are sticky.

#### 3.3 — Validate persona quality

- Silhouette score for K-means at k=3, 4, 5, 6, 7.
- Choose the k that balances interpretability and quality.
- If 3 truly is best, that's also a finding worth reporting. We will not force more clusters than the data supports.

**Cycle gate:**
- [ ] Personas have evocative, human names (not "Cluster 0").
- [ ] Each persona has a one-page profile in `/outputs/tables/personas.csv`.
- [ ] Silhouette analysis is saved.

---

### **STAGE 4: Open Text Analysis — Hopes and Fears (~3 hours)**

**Goal:** The richest data in this survey is `ai_opentext`. The official report quoted a few responses. We will go further.

Create `notebooks/04_text_analysis.ipynb`.

#### 4.1 — Preprocessing

- Filter to non-null, non-trivial responses (length > 20 chars).
- Lowercase, lemmatize, remove stopwords.
- Save cleaned text alongside cluster, region, role for stratification.

#### 4.2 — Sentiment analysis

- VADER sentiment (it's free, fast, decent for survey responses).
- Score each response: compound score, plus pos/neg/neu breakdown.
- Compute mean sentiment by cluster, region, role.
- **Hypothesis:** AI Consumers have more positive sentiment than Skeptics. This is unsurprising, but the magnitude matters.
- **Better hypothesis:** Within each cluster, what's the variance? High-variance clusters have polarized opinions. Low-variance clusters are consensual. The official report doesn't show variance.

#### 4.3 — Hope vs Fear extraction

This is the creative bit. For each response, classify whether the response leans more toward HOPE, FEAR, or BOTH. Two approaches:

**Approach A (rule-based, transparent, defensible):**
Build keyword lists:
- Hope words: "excited," "potential," "opportunity," "useful," "saves time," "efficient," "powerful," "transformative," "helpful," "promising"
- Fear words: "worried," "afraid," "scared," "concerned," "lose," "replace," "displacement," "bias," "privacy," "harm," "dangerous"

Count hope words and fear words in each response. Classify:
- Hope-dominant: hope >> fear
- Fear-dominant: fear >> hope
- Balanced: roughly equal, both > 0
- Neutral: neither

Compute the distribution by cluster, role, region.

**Approach B (topic modeling, more sophisticated):**
- Use BERTopic or even simpler: TF-IDF + KMeans on responses. Cluster responses into ~10 topics.
- Manually label each topic from a sample of representative quotes.
- Likely topics: workforce displacement, data privacy, productivity gains, equity concerns, hallucination/accuracy, learning curve, vendor lock-in.

Use approach A first (it's robust and explainable). If time permits, layer approach B on top.

#### 4.4 — Quotable quotes

For the deck, pull 4-6 powerful, anonymized quotes that illustrate each major theme. Keep quotes under 20 words each. Tag each quote with cluster, region, role.

The official report already used: "You won't be replaced by AI. You'll be replaced by someone who knows how to use AI." We need different quotes that we found ourselves.

#### 4.5 — Word clouds (use sparingly)

A word cloud per cluster can be a striking visual ONLY if it shows real differences. If the three clusters all have "AI" "data" "use" "work" as top words, skip it. If Skeptics have noticeably different vocabulary (e.g., "human," "trust," "afraid"), the contrast is the visual.

**Cycle gate:**
- [ ] Sentiment scores computed and stored.
- [ ] Hope/fear classification has transparent rules in code.
- [ ] At least 6 quotes pulled and verified to exist in the dataset.

---

### **STAGE 5: Geographic and Cause-Area Deep Dive (~2 hours)**

**Goal:** Sub-thesis C demands we go beyond Global North/South. Build the regional cube the official report didn't.

Create `notebooks/05_geo_cause_analysis.ipynb`.

#### 5.1 — Regional AI readiness scorecard

For each region (NA, Europe, India, Africa, LATAM, Rest of Asia):
- Sample size (and warn if small).
- % AI Consumers
- % with tech/MERL person
- Mean comfort
- Top 3 AI uses
- Top 3 AI wants
- Top 3 risk concerns

Visualize as a small-multiples chart: each region is one tile. **Be transparent about small samples.** Africa and LATAM are tiny in this dataset; we will note this and not over-interpret.

#### 5.2 — Cause area × cluster heatmap

The `org_label` field has cause areas. For each cause area:
- % in each cluster
- Mean AI use count
- Mean want-use gap

This generates a heatmap. **Expected finding (from official report):** Education and Health overrepresent in AI Consumers. Arts/Culture and Religious overrepresent in Skeptics. We can confirm and visualize this elegantly.

#### 5.3 — Rural vs urban (India sample only)

The India responses have rural/urban tagging. With n≈251 from India (subset further by rural/urban), we can test:
- Is rural-India different from urban-India in AI readiness?
- This is a niche finding but POWERFUL because it's local enough to be actionable for Indian funders.

**Be honest about sample sizes** — this becomes a secondary slide, not a headline.

**Cycle gate:**
- [ ] All small-sample warnings logged.
- [ ] Heatmap renders cleanly.

---

### **STAGE 6: The "Want-Use Gap Index" — Our Original Contribution (~2 hours)**

**Goal:** Coin a metric. Make it ours. Make it memorable.

Create `notebooks/06_want_use_gap_index.ipynb`.

We define:

> **AI Capability Gap Index (AICGI)** = average over the 7 AI use-cases of `(want_i - use_i)`, weighted by data-readiness deficit.

Or, simpler and more defensible:

> **Want-Use Gap (WUG)** for organization i = (number of AI use-cases wanted) − (number of AI use-cases currently used).

A WUG of +5 means an organization wants 5 things they aren't doing. A WUG of 0 means perfect alignment. A WUG of -2 means they use more than they want (rare, but possible if "want" was answered differently from "use").

Compute WUG for every respondent. Then:

- WUG distribution overall (histogram).
- Mean WUG by cluster. Late Adopters should have the highest WUG.
- Mean WUG by region.
- WUG vs data infrastructure score (scatter or boxplot).
- **Highlight finding:** "X% of nonprofits have a WUG ≥ 3 — they want at least 3 things from AI that they currently can't or don't do."

Then map WUG to actionable segments:
- **Activation-ready:** WUG ≥ 3 AND infrastructure score ≥ 3 → these orgs just need pilot programs.
- **Foundation-needed:** WUG ≥ 3 AND infrastructure score < 3 → these orgs need data infrastructure first.
- **Already-well-served:** WUG ≤ 1 AND infrastructure score ≥ 3 → these orgs are doing fine.
- **Disengaged:** WUG ≤ 1 AND infrastructure score < 3 → low motivation, low capacity (overlaps heavily with Skeptics).

This 2x2 segmentation is a SLIDE WORTH MEMORIZING. Funders can immediately see where their dollars should go.

**Cycle gate:**
- [ ] WUG computed for all 930 rows.
- [ ] 2x2 matrix populated with counts and percentages.
- [ ] Each segment has a recommended intervention written in plain language.

---

### **STAGE 7: Predictive Model — What Predicts AI Readiness? (~2 hours)**

**Goal:** Train a small interpretable model. Use it for explanation, not deployment.

Create `notebooks/07_predictive_model.ipynb`.

- Target: `is_AI_consumer` (binary).
- Features: org_size, org_years, regionality, region (as categoricals), all infrastructure flags, role.
- Models: Logistic regression (interpretable) AND a Random Forest or Gradient Boosted (for feature importance comparison).
- Report:
  - Train/test split (stratified).
  - Accuracy, F1, AUC.
  - Feature importance from RF.
  - Coefficients with confidence intervals from logistic.
  - SHAP values if time permits (one bar chart of mean |SHAP| per feature is enough).

**Be careful:** This is a small dataset (930). Don't oversell predictive performance. The point of the model is to RANK feature importance, not to forecast. In the deck, the headline is "the top 3 predictors of AI readiness are X, Y, Z," not "we built a model with 87% accuracy."

**Cycle gate:**
- [ ] Cross-validated metrics, not single-split.
- [ ] Feature importance chart saved.
- [ ] Caveats documented.

---

### **STAGE 8: Visualization Polish (~3 hours)**

**Goal:** The deck stands or falls on visuals. Time to upgrade every chart.

#### 8.1 — Color palette

Use a fixed palette across every chart:
- **Primary**: GivingTuesday-inspired deep blue `#1F4E79` (or similar — verify against their site).
- **Accent 1 (warm)**: Coral / warm orange `#E8743B` for highlighting.
- **Accent 2 (cool)**: Soft teal `#3CB4AC`.
- **Neutral**: Warm gray `#6B6B6B` for secondary text.
- **Background**: Cream / off-white `#F7F4EE`.

Three-cluster encoding:
- AI Consumers: deep blue `#1F4E79`
- Late Adopters: coral `#E8743B`
- AI Skeptics: warm gray `#6B6B6B`

This encoding is intentional: blue = trust/established, orange = warmth/in-progress, gray = withdrawn. It's emotionally honest about each cluster's relationship to AI. **Use this same encoding in EVERY cluster chart.** Consistency builds credibility.

For colorblind safety, double-check with a simulator (the matplotlib `cycler` and `colormap` choices must be CB-friendly).

#### 8.2 — Chart-by-chart upgrade

Every figure must have:
- A descriptive title that states the takeaway in a sentence (e.g., "Late Adopters want 4x more from AI than they currently do" — not "AI use vs want by cluster").
- Direct labeling on bars/lines (not legends, when possible).
- Source line in small text: "Source: GivingTuesday AI Readiness Survey 2024, n=930."
- Removed chartjunk: no gridlines unless needed, no 3D, no shadows.
- Sans-serif font (Helvetica/Arial). Title 16-18pt, axis labels 11-12pt.

Use `matplotlib` + `seaborn` for static charts. Use `plotly` ONLY if we need interactivity in an HTML dashboard (Stage 9).

#### 8.3 — The headline charts (priority order)

1. **The Want-Use Gap chart** (Stage 6). Diverging bars: each AI use-case has a "use" bar going left and "want" bar going right. Sort by gap.
2. **The 2x2 segmentation chart** (Stage 6). Quadrant scatter or 2x2 grid with counts.
3. **The cluster persona quad** (Stage 3). Four small radar charts or one parallel coordinates plot.
4. **The risk perception by role chart** (Stage 2.1). Horizontal stacked bars.
5. **The infrastructure score → AI use slope chart** (Stage 1.6). Clean line plot.
6. **The regional readiness scorecard** (Stage 5.1). Small multiples.
7. **The hope-vs-fear ratio chart** (Stage 4.3). Diverging bars by cluster.

Save all as both `.png` (300 DPI for slides) and `.svg` (vector for the GitHub README).

#### 8.4 — One signature visualization

Build ONE truly memorable visual. Options:
- An interactive Plotly Sankey diagram showing flow from cluster → top AI uses → top AI wants → top risks. Judges will remember this.
- A "Periodic table" of AI use-cases, where each cell shows current % use and desired % use, color-coded by gap size.
- An animated GIF showing how the 3 clusters distribute across regions.

Pick ONE based on what the data best supports. The Sankey is probably safest and highest-impact.

**Cycle gate:**
- [ ] Every figure regenerated with the new palette.
- [ ] Every title states a takeaway, not a description.
- [ ] At least one signature visualization built.

---

### **STAGE 9: Building the Deliverable (~3 hours)**

**Goal:** A slide deck (HTML or PPTX) plus a polished GitHub README.

#### 9.1 — Deck structure (Pyramid Principle: lead with the recommendation)

Follow the Kellogg / Management Consulted advice cited in research: top-down, recommendation-first, every slide title is a full sentence stating a takeaway.

Slide-by-slide outline (target: 12-15 slides + appendix):

1. **Title** — "Why Nonprofits Want AI but Can't Use It: A Story from 930 Voices Worldwide." Sub-line: "Bay Area Data Analytics Case Competition · GivingTuesday × Gates Foundation Track."

2. **The headline** — "Most nonprofits are not held back by AI. They're held back by data plumbing." One striking number from our analysis (e.g., "X% want AI organization tools, Y% have any cloud storage").

3. **The data** — "We analyzed GivingTuesday's 2024 survey of 930 nonprofits across 6 continents." Sample summary infographic: continents, cluster sizes, role mix.

4. **The official story** — "Three personas: Consumers, Late Adopters, Skeptics. Capacity does not predict readiness." (Brief acknowledgment of the official GTDC findings, with link.) This positions our work as a "yes, and."

5. **Where we go further** — Our 5 sub-theses listed crisply.

6. **Sub-thesis A: The Want-Use Gap is the real story.** The diverging bar chart. Headline: data organization is the #1 unmet need, larger than generative AI's gap. Insight: nonprofits don't want more chatbots. They want their data sorted out.

7. **Sub-thesis B: Tech staff are MORE worried about AI risks, not less.** The risk-by-role chart. Counter-intuitive. Quotes from tech respondents.

8. **Sub-thesis C: The Global South is not one place.** Regional scorecard small-multiples. Highlight that India and Africa have very different AI use patterns and risk concerns.

9. **Sub-thesis D: Data infrastructure is the gating factor.** The infrastructure score → AI use slope. Show the dose-response: each piece of infrastructure adds measurable AI capability.

10. **Sub-thesis E: The 15-staff threshold is real but it's a hiring proxy.** Visual showing the kink at ~15 staff in tech/MERL hiring rates.

11. **Sub-thesis F: Hope and Fear coexist.** The dual-axis or diverging bar of hope-words vs fear-words. Two carefully chosen quotes.

12. **The 2x2 segmentation: where should funders invest?** Our original framework. Activation-Ready, Foundation-Needed, Already-Served, Disengaged. Numbers and intervention recommendations.

13. **The recommendations** — Three concrete asks, in priority order:
   - **For funders**: Stop funding AI tools. Start funding data infrastructure subsidies for Foundation-Needed nonprofits.
   - **For tool builders**: Build for the 2x2 quadrant. Different products for different segments.
   - **For nonprofits**: Audit your data infrastructure before adopting any AI tool. Use our infrastructure score as a self-check.

14. **Limitations and what we'd do with more time** — Honest about: cross-sectional data, sampling biases (mostly US + India), self-report. What we'd add: longitudinal follow-up, intervention pilots.

15. **Thank you / Q&A.**

**Appendix slides** (5-10): all the supporting analyses, hypothesis test outputs, predictive model details, methodology notes. The appendix is where we shine in Q&A.

#### 9.2 — Slide format

Use **HTML slides** (reveal.js) — they look modern, they version-control well, they embed Plotly cleanly, and they GitHub-Page beautifully for the case competition deliverable link.

Alternative: pptx via the `pptx` skill. **Read the skill at `/mnt/skills/public/pptx/SKILL.md` first.** PPTX is what most case comp judges expect, so this is probably the safer choice. Build the pptx programmatically so we can regenerate it if any stat changes.

**Decision rule:** Build PPTX as the primary deliverable (judges expect it), and HTML as a stretch goal for the GitHub link.

#### 9.3 — GitHub README

Use the `frontend-design` skill or just craft clean Markdown. The README must include:

- The thesis in 2 sentences.
- A "Key Findings" section with 5-7 bullets, each citing the relevant slide/figure.
- A reproducibility section: how to clone, install, and run all notebooks in order to reproduce every chart and stat.
- Links to all figures (rendered inline).
- A methodology note.
- Citations to the GivingTuesday official report and other sources.
- Author info / contact.

**Cycle gate:**
- [ ] Every slide title is a full-sentence takeaway.
- [ ] Every chart in the deck appears in `/outputs/figures/`.
- [ ] Every claim in every slide has a `findings.json` entry backing it.
- [ ] README renders cleanly on GitHub.

---

## **3. CITATIONS AND SOURCES (For the Deck and README)**

These are the specific links Claude Code should reference in the README and the slide footnotes. Verified during research:

**Primary source:**
- GivingTuesday AI Readiness Report 2024 (Global): https://ai.givingtuesday.org/ai-readiness-report-2024/
- GivingTuesday AI Readiness Report 2024 (India Supplement): https://ai.givingtuesday.org/ai-readiness-survey-report-2024-india/
- GivingTuesday Generosity AI hub: https://ai.givingtuesday.org/
- Early insights blog post (May 2024): https://www.givingtuesday.org/blog/early-insights-from-ai-readiness-survey/
- Generosity AI Problem Library blog: https://www.givingtuesday.org/blog/insights-ai-problem-library/

**Comparable benchmark surveys (for sector triangulation):**
- TechSoup × Tapp Network "State of AI in Nonprofits 2025": https://page.techsoup.org/ai-benchmark-report-2025
- NonProfit PRO "12 Revealing Nonprofit Stats from 2025": https://www.nonprofitpro.com/article/12-revealing-nonprofit-stats-from-2025/
- Whole Whale "Top Nonprofit AI Policies 2025": https://wholewhale.com/tips/top-nonprofit-ai-policies-2025-analysis-and-trends/
- The AI Equity Project (Namaste Data + Giving Compass): https://www.namastedata.org/

**Theoretical / framing references:**
- Stanford HAI 2025 AI Index Report (cited by Kiteworks): https://www.kiteworks.com/cybersecurity-risk-management/ai-data-privacy-risks-stanford-index-report-2025/
- Microsoft AI Diffusion Report 2025 (cited via CRN): https://www.crnasia.com/news/2026/artificial-intelligence/microsoft-s-ai-report-highlights-uneven-adoption-across-coun
- Rogers' Diffusion of Innovations + Moore's Crossing the Chasm framing: https://en.wikipedia.org/wiki/Crossing_the_Chasm
- Gartner Hype Cycle (referenced in the official GTDC report).
- Tandfonline "AI in the Nonprofit Human Services: Hype, Harm, and Hope": https://www.tandfonline.com/doi/full/10.1080/23303131.2024.2427459
- ArXiv "AI Adoption Across Mission-Driven Organizations" (2025): https://arxiv.org/pdf/2510.03868
- WEF "AI Divide between Global North and South": https://www.weforum.org/stories/2023/01/davos23-ai-divide-global-north-global-south/
- data.org Data Maturity Assessment: https://data.org/dma/

**Data storytelling craft (so we know how to present):**
- Kellogg "Six Strategies for Winning Case Competitions": https://www.kellogg.northwestern.edu/news/blog/2019/04/23/strategies-for-winning-case-competitions/
- Management Consulted "Case Competition Tips" (Pyramid Principle): https://managementconsulted.com/case-competition/
- NetSuite "Data Storytelling Tips": https://www.netsuite.com/portal/resource/articles/data-warehouse/data-storytelling-tips.shtml

In the deck, every external claim gets a footnoted source. Every internal claim from our analysis gets a "Source: GivingTuesday 2024 AI Readiness Survey, n=930, our analysis." This is non-negotiable.

---

## **4. PRE-FLIGHT CHECKLIST AND COMMON PITFALLS**

Before considering the deliverable done, walk through this list:

**Data integrity:**
- [ ] Did we double-check that `cluster3 == 1` is actually the AI Consumers and not Late Adopters? The dictionary uses both `{1: 523, 0: 265, -1: 142}` for the HDBSCAN result and `{1: 524, 0: 406}` for the K-means result. Verify which one is in which file. **Don't mix them up.** This is the #1 source of catastrophic error.
- [ ] Are normalized columns (0-1 scale) and raw columns (0-10 scale) being used appropriately? Don't show "comfort = 0.7" on a slide; convert to "7 out of 10."
- [ ] Multi-select columns (`ai_use`, `ai_want`, `ai_risk`, `data_kinds`) — are we parsing them correctly? Each is likely a comma-separated string in the raw CSV. Look at the [U] / [W] / [D] prefixed binary columns in the clustering data; those should be the parsed versions. Use those.

**Analytical honesty:**
- [ ] Did we apply multiple-comparisons corrections where appropriate?
- [ ] Did we report effect sizes alongside p-values?
- [ ] Did we acknowledge sampling limitations (especially small Africa/LATAM samples)?
- [ ] Did we avoid causal language where only correlation is supported?
- [ ] Are confidence intervals shown on key estimates?

**Presentation craft:**
- [ ] Every slide passes the "stranger test": a judge with no prior context understands the slide in 5 seconds.
- [ ] No slide has more than 1 main idea.
- [ ] No chart has more than 5 colors (for cluster encoding, 3 colors).
- [ ] Every quote is anonymized AND verified to exist verbatim in the dataset.
- [ ] No em dashes anywhere in the prose. Use commas, parentheses, or just rewrite.

**Voice (this matters more than people think):**
- [ ] Read every slide aloud. If it sounds corporate ("synergize," "leverage," "robust framework"), rewrite it.
- [ ] If it sounds like a sincere undergrad explaining to a friend, keep it.
- [ ] Acknowledge uncertainty. "We think" and "the data suggests" are stronger than fake certainty.

**Reproducibility:**
- [ ] Every notebook runs end-to-end on a clean install of `requirements.txt`.
- [ ] No hardcoded paths. Use a `config.py` or `pathlib.Path(__file__).parent`.
- [ ] Random seeds set everywhere (`np.random.seed(42)` and `random.seed(42)`).

---

## **5. RISK REGISTER (What Could Go Wrong)**

Let me think through what's most likely to break:

1. **Cluster ID confusion.** The clustering data has both `cluster2` and `cluster3` columns and they encode different models with different conventions (`-1` is the noise label in HDBSCAN, but here it labels the AI Skeptics group). Mix these up and our entire narrative breaks. **Mitigation:** Stage 1.2 explicitly verifies cluster labels against dictionary expectations before any downstream analysis.

2. **Multi-select parsing errors.** Survey multi-selects often have inconsistent encoding (some columns might be `1/0`, others `True/False`, others comma-separated strings). **Mitigation:** Use the pre-parsed `[U]`, `[W]`, `[D]` columns from the clustering CSV wherever possible. They're already booleans.

3. **Normalized vs raw column mix-up.** Showing `0.42` when we mean `4.2/10` is embarrassing. **Mitigation:** Use the `_raw` suffix consistently in plotting code. Helper function `to_human_scale()` that asserts the variable is raw.

4. **Small-sample over-interpretation.** Africa has small n. India has decent n but rural-India is small. **Mitigation:** Hard rule: if subgroup n < 30, do not put a precise % on a slide. Use language like "directionally" or "limited evidence suggests."

5. **The official GTDC report has already published the obvious findings.** If our slides recap their findings, we lose. **Mitigation:** Stage 1's "where do we disagree or go further" check. Every headline slide must contain a finding NOT in the official executive summary.

6. **Chart overload.** Putting 30 figures into 15 slides means 30 cluttered slides. **Mitigation:** One headline chart per slide. Supporting evidence in appendix.

7. **Scope creep.** This plan is already ambitious. Trying to also build a real-time dashboard, do BERTopic, train transformers, etc., burns time. **Mitigation:** The signature visualization (Sankey or quadrant) is the only "stretch" item. Everything else is core.

8. **Hallucinated quotes.** Easy mistake when summarizing open text. **Mitigation:** Every quote in the deck must come from `df.loc[idx, 'ai_opentext']` directly. Save the index alongside.

9. **Color accessibility.** Some judges may be colorblind. **Mitigation:** Test palette in Coblis simulator. Always pair color with another visual cue (position, shape, label).

10. **Time pressure to skip the cycle.** Plan → analyze → review → test → log. It's tempting to ship findings fast. **Mitigation:** The `findings.json` ledger is mandatory. No finding makes it to the deck without an entry.

---

## **6. ESTIMATED TIME BUDGET**

Total: ~22 hours of focused work, give or take.

- Stage 0 (Setup): 0.5h
- Stage 1 (EDA): 3h
- Stage 2 (Hypothesis testing): 2h
- Stage 3 (Clustering deepening): 2h
- Stage 4 (Text analysis): 3h
- Stage 5 (Geo/cause): 2h
- Stage 6 (Want-Use Gap Index): 2h
- Stage 7 (Predictive model): 2h
- Stage 8 (Visualization polish): 3h
- Stage 9 (Deliverable: deck + README): 3h

If time is short, the priority cuts are: drop Stage 7 (predictive model) first, then Stage 4.5 (word clouds) and Stage 4.4 topic modeling layer. Do not cut Stages 1, 2, 6, 8, or 9 — they are load-bearing.

---

## **7. FINAL WORDS TO CLAUDE CODE**

Hey Claude Code, a few last things before you start.

The user is a UC Berkeley freshman in Data Science, sharp, but not yet a 20-year veteran. Treat them like a smart partner who needs you to **show your work, explain your reasoning, and never invent a statistic.** When something is uncertain, say so. When the data surprises you, lean into it instead of forcing it into a tidy story.

The official GivingTuesday report is good. We are not trying to dunk on it. We are trying to add a layer of original analysis that a judging panel will find genuinely fresh. The thesis ("data infrastructure is the gating factor, not the technology") is one we believe the data will support, but **if the data doesn't support it, change the thesis**. Don't bend numbers to fit a narrative.

Storytelling matters as much as analysis here. A finding is only useful if a judge remembers it 24 hours later. So when you write any slide title, prose, or quote, ask yourself: "If I told my roommate this finding at dinner, would they nod and say 'huh, that's actually interesting'?" If not, dig deeper.

And finally: have fun with this. The data is rich. The topic is genuinely important. Generosity, AI, equity, the Global South, the future of work in the social sector — these are real questions that real people care about. A case competition is a chance to take 930 voices in a spreadsheet and turn them into something a roomful of judges will think about on the drive home.

Plan, analyze, read code, implement, code review, test, bug fix, log finding, repeat. Trust the cycle.

Go win this thing.

— end of plan.md —