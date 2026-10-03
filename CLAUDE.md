# NTU Datathon 2026 — Track 1: Smart Campus Analytics

We are building a competition submission, not a tutorial exercise. **The goal is to win** — stand out from every other team, not just produce a working model. Read this whole file before doing any modeling work in this repo.

## The problem (official, locked in at opening ceremony 2026-10-03)

Predict a campus facility's **energy consumption** (`energy_usage`, continuous) using historical usage, environmental conditions, building information, and time-based patterns. Regression problem.

- `Track 1 Dataset/Track 1 Training Dataset.csv` — 8000 rows: `id, building_id, building_type, hour, day_of_week, month, temperature, humidity, occupancy, previous_usage, energy_usage`
- `Track 1 Dataset/Track 1 Testing Dataset.csv` — 3000 rows, same minus `energy_usage`
- `building_id` examples: LEC_A, SCI_A, RES_A, ENG_A, SCI_B, SPT_A — multiple specific buildings per `building_type` category
- `previous_usage` is the prime lag-feature suspect (the analogous feature in the practice workshop had 0.90 correlation with target) — **verify with EDA, don't assume**
- Full official docs: `PS/NTU-Datathon-2026-Information-Booklet.pdf` and `PS/IMG_163[3-8].jpg` (opening ceremony slides, same content)

## How we're judged (this determines what effort goes where)

**Round 1 (score-based, eliminates down to finalists):**
- 60% Model Performance — **RMSE is primary**, remainder of this bucket from MAE/R²
- 40% Technical Proposal — the 1-page PDF report
- Scored against a **private test set**, not the public leaderboard. The public leaderboard is for self-tracking only and does NOT reflect final rank — never tune against it, that's how you overfit and lose silently.

**Final round (in-person booth/showcase, if shortlisted):** technical methodology, results & analysis, innovation & creativity, real-world applicability, implementation quality, Q&A defence — presented straight from the submitted notebook.

**Their own golden rule, stated explicitly on the problem slide: "Best model ≠ most complex model."** This is the guardrail on the winner-mentality instinct below — every added layer of complexity must earn its place with a measurable metric or insight gain, and you must be able to defend it in Q&A. A fancy model you can't justify loses to a simple one you can.

## Operating mode: winner mentality, with guardrails

Shrey wants aggressive, competition-grade depth — not the safe/textbook solution a typical prompter would get. Concretely:

- Treat the DLW workshop pipeline (see below) as the **floor every other team will hit**, not the ceiling. Most teams won't even bother referencing it — visibly building on top of it (not ignoring it, not just matching it) is itself part of standing out.
- Push for gradient boosting (XGBoost/LightGBM/CatBoost), ensembling/stacking, serious feature engineering, proper hyperparameter search, and cross-validation beyond what the workshop taught.
- But: respect the actual judged metric (RMSE-led), avoid data leakage and invalid validation even in pursuit of a better number, and be ready to justify every choice — a disqualified, overfit, or indefensible submission doesn't win.
- Benchmark every improvement against the baseline so the gain is visible and explainable, both for the technical report and the final-round Q&A.

## Build process: phased, not one-shot

Do not attempt to build "the whole winning solution" in a single pass — that forces silent decisions (validation strategy, which features to trust, metric weighting, model family) that should be checkpointed with Shrey first, especially since this is time-ordered, lag-feature-heavy data where a wrong split silently invalidates every number downstream. Sequence:

1. **EDA** — show what the data actually says (distributions, correlations, per-building/type patterns, missingness) before touching any model.
2. **Lock validation strategy + metric** together with Shrey. Given `hour`/`day_of_week`/`month` columns, a time-aware split is likely required again (see workshop leakage lessons) — confirm whether rows are time-ordered within `id` before assuming.
3. **Baseline** — simplest reasonable model, get an honest number to beat.
4. **Iterate** — feature engineering → model comparison → advanced models/ensembling, each step building on a checkpoint already seen.
5. **Dashboard/presentation** — only after the model is solid and validated; don't let it eat into modeling time early. Keep it lightweight (Streamlit/Plotly) for the Round 2 showcase.

## Decision log — mandatory, running, not reconstructed after the fact

While building, keep an explicit, presentation-ready log of decisions and their justification: validation strategy chosen and why, metric weighting, features engineered/dropped and why (flag anything leakage-risky), models tried and why one won, hyperparameter choices, any explicit tradeoff accepted. Short entries, written as decisions happen. This feeds directly into:
- the mandatory 1-page technical report (must focus on "reasoning behind your decisions, key insights, limitations/improvements")
- the final-round Q&A defence
- Shrey's dashboard/presentation

## What to build on from the workshops (practice problems, NOT this dataset — techniques transfer, data doesn't)

From the theme-park regression workshop: time-aware train/val split (never random on time-ordered data), one shared `make_features()` function applied identically to train and test, cyclical encoding for hour, lag features, explicit data-leakage checks (if a feature can't be computed from the test file's columns, it's leakage), RMSE/MAE/R² read together (RMSE ≫ MAE flags a few catastrophic misses), Random Forest beat Linear Regression on that dataset but don't assume the same holds here.

From the asteroid imbalanced-classification workshop (not directly applicable here since this is regression, but keep the mindset): never trust a single metric in isolation, validate the metric choice against what's actually rare/costly, question every feature's correlation with target before keeping it (one feature there had ~0 correlation and was a trap).

## Deliverables & hard constraints (both tracks, from the official booklet)

Single submission per team, via official portal, must include ALL of:
1. `.ipynb` notebook — full pipeline: data loading/preprocessing, EDA, feature engineering, model training, evaluation, prediction generation, any other steps needed to reproduce
2. Trained model as `.pkl` (`joblib.dump((model, feature_names), "model.pkl")`) — reload in a clean Colab, reapply the exact same `make_features()`, predict on test, save as `<TEAM_NAME>_Datathon_2026_Track_1_Prediction.csv`
3. Technical report — **1 page, PDF only**
4. `requirements.txt`

**Must run end-to-end with zero errors in a clean environment (e.g. fresh Google Colab) before submitting — any execution error during evaluation may cause disqualification.** This is non-negotiable and should be checked before every submission, not just the final one.

**Deadline: 11:30 AM, 4 October 2026.** Final round (if shortlisted): registration 2:30–4:30pm same day at NTU ARC Basement 2, judging 4:30–6:00pm, prizes/closing 6:00–6:30pm at LT1A North Spine.
