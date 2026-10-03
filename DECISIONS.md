# Decision Log — Track 1: Smart Campus Analytics

Running record of every consequential modelling decision and the evidence behind it.
Feeds directly into the 1-page technical report, the final-round Q&A, and the dashboard.

---

## Phase 1 — EDA & validation strategy

| # | Decision | Justification | Evidence |
|---|---|---|---|
| **D1** | Keep all 8,000 training rows; no deduplication or row filtering | Dataset is clean in the conventional sense — nothing to remove | 0 duplicate rows, 0 duplicate ids, 0 missing targets |
| **D2** | Use **random K-fold CV**, not a time-aware split | `id` carries no temporal signal, so the workshop's time-split rule does not apply here. A time split would cost us 20% of training data and fold-variance reduction to defend against leakage that doesn't exist | All `id`↔feature correlations within ±0.016; target flat across id-deciles; id-ordered holdout RMSE **3.606** vs random 5-fold **3.560 ± 0.071** — difference inside fold noise |
| **D3** | Grade every model on **missingness-injected validation**, not naive validation | Test set is missing data 4–5× more often than train. Naive validation grades the model on an easier problem than it will face, and can misrank candidates that degrade differently under NaN | Same model, same rows: naive RMSE **3.560** → realistic RMSE **3.960**. Optimism gap **+0.400 RMSE (+11.2%)**. Rows with ≥2 missing: train 0.14% vs test 7.70% (**55×**) |
| **D4** | Treat `previous_usage` as the anchor; the model's job is the **correction** to it, not the level | One feature explains almost all the variance; the remaining signal is a small structured residual, which is what modelling effort should target | r = 0.945 with target. Residual (delta) mean +1.60, sd 6.20; shows hour structure (+3.3 at 07–09h, −2.2 at 23h) and mean reversion (r = −0.24 vs `previous_usage`) |
| **D5** | Decide target framing (raw vs `log1p` vs delta) **empirically**, not from skew | RMSE is scored on the raw scale, so a transform changes the implied loss function — lower skew does not automatically mean lower RMSE | Raw skew +0.72; log1p skew −0.15; delta skew −1.25; ratio skew +6.40 |
| **D6** | Prefer models with **native NaN handling** (LightGBM / XGBoost / CatBoost / HistGBR) over impute-then-fit pipelines | 18.2% of test rows carry at least one NaN. How a model behaves with absent inputs is a first-class design question here, not a preprocessing detail | `previous_usage` — the strongest feature — is absent in 6.8% of test rows |

### Baselines to beat (established before modelling)

| Reference | RMSE |
|---|---|
| Predict global mean | 18.45 |
| Predict building's mean | 13.22 |
| **Predict `previous_usage` (no model)** | **6.40** |
| Predict `previous_usage` + 1.60 | 6.20 |
| Quick HistGBR, realistic validation | 3.96 |

---

## Phase 2 — Feature engineering & model comparison

_(in progress)_
