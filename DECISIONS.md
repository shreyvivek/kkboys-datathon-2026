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

## Phase 2 — Feature engineering & model selection

All scores below are **realistic RMSE** (5-fold CV with test-level missingness injected into
every validation fold). Naive numbers are never used to decide anything.

| # | Decision | Justification | Evidence |
|---|---|---|---|
| **D7** | Encode per-building structure explicitly: intercept, 24-hour profile, weekend offset, and own slopes on occupancy/temperature/previous_usage (365 features) | Buildings differ in *shape*, not just level, and a linear model cannot discover interactions by itself — they must be written out | Building means span 30.8 (ADM_A) → 79.2 (SCI_A); sd varies 6.5 (Residential) → 19.1 (LectureHall) |
| **D8** | Impute missing values with a **sub-model** (one LightGBM per gap-prone column) rather than a median | A global median is wrong in context — `previous_usage` for a Science building at 2pm is nothing like 49.7. Worse, the 12 building-slope terms are `is_building × value`, so one gap wiped out the whole building relationship for that row | Ridge **3.852 → 3.339**; CatBoost **3.632 → 3.490**. Beat native-NaN (3.632), building-median (3.614) and MICE (3.562) |
| **D9** | Keep `*_missing` flags and `n_missing` after imputation | The model must be able to distinguish an observed value from an estimated one, and multi-gap rows are the distinctive hard case in test | Test has 7.7% multi-gap rows vs 0.14% in train (55×) |
| **D10** | **Augment training data** with injected missingness at test rates | Training contains ~11 multi-gap examples vs 231 in test — the model had almost no exposure to the regime it is graded on | Helped every model: CatBoost −0.265, XGBoost −0.150, HistGBR −0.134, LightGBM −0.115, Ridge −0.069. Also halved fold variance (CatBoost sd 0.150 → 0.067) |
| **D11** | **Ship Ridge alone — reject the 4-model ensemble** | Non-nested, the stack looked better (3.2259). Nested, it scores 3.2605 — *worse* than Ridge at 3.2326. The apparent edge was weights fitted on the scored predictions. Also fails the brief's "best model ≠ most complex model", and Ridge coefficients are interpretable for the proposal/Q&A | Nested stack 3.2605 ± 0.0196 across 3 outer folds (weights 71–82% Ridge); Ridge alone 3.2326 |
| **D12** | Serialise `model.pkl` with `cloudpickle` (by value), not plain joblib | The pipeline contains a custom transformer; a standard pickle stores only a class *reference*, so loading in a fresh notebook would raise `AttributeError`. By-value embeds the definition | Verified: loads and predicts in an empty directory, fresh process, zero class definitions present |
| **D13** | **Reject a dedicated "no-`previous_usage`" model** for rows where `previous_usage` is missing (a second Ridge trained with every prev column removed, routed in by an if-missing switch) | Idea: avoid applying the strong `previous_usage` weight to an *estimated* value. But the sub-model imputer + rebuilt slope terms + augmentation (D8–D10) already capture this; the extra model adds a moving part for no gain | 5× repeated 5-fold realistic CV (seeds 42, 1–4), paired vs Ridge alone: routed **+0.0087 RMSE, worse in 4/5**; also worse on prev-missing rows (4.194 vs 4.095). Averaging both on those rows: −0.0006, better in 3/5 — inside noise |

### Final model

**Ridge (RidgeCV, α = 1.74) on 365 building-aware features, with sub-model imputation,
trained on missingness-augmented data.**

| Metric | Realistic validation |
|---|---|
| **RMSE** | **3.2326** |
| MAE | 2.5279 |
| R² | 0.9693 |

### Model comparison (realistic validation)

| Model | RMSE |
|---|---|
| **Ridge + interactions** | **3.2326** |
| NNLS stack, nested (honest) | 3.2605 |
| CatBoost | 3.4867 |
| LightGBM | 3.5477 |
| XGBoost | 3.5824 |
| Simple average of all four | 3.3641 |

### Progress against references

| Reference | RMSE |
|---|---|
| Predict global mean | 18.45 |
| Predict building mean | 13.22 |
| Copy `previous_usage` | 6.40 |
| Earlier team blend (naive 3.297) measured honestly | 3.745 |
| **Final model** | **3.233** |

---

## Phase 3 / 5 — optimisation attempts

Twelve further variations tested on the same protocol. **One kept.** The reference point for all
of these is the single-seed model at **3.2257**.

| # | Decision | Justification | Evidence |
|---|---|---|---|
| **D13** | **Keep the raw target** — reject `log1p` and delta framings (closes D5) | RMSE is scored on the raw scale, so a transform changes the implied loss. The delta framing additionally makes the prediction *structurally dependent* on `previous_usage`, which is absent in 6.8% of test rows — recovering a prediction requires adding back a column that isn't there | log1p **3.571**; delta **5.594** vs raw **3.226** |
| **D14** | Reject additional interaction terms, splines, and `n_missing` interactions | The existing 288 building×hour profile terms and per-building slopes already express this structure; adding global versions duplicates it | extra interactions 3.247; splines 3.245; `n_missing`×prev 3.226 (identical to baseline) |
| **D15** | Reject a specialist model for rows missing `previous_usage` | Routing those rows to a model trained without the feature scores the same as imputing it — meaning the sub-model imputer is already performing as well as avoiding the feature entirely. Imputation is no longer the bottleneck | specialist routing 3.228 vs 3.226 |
| **D16** | Keep the imputer at 300 trees / lr 0.05, single pass | Larger and iterated imputers do not help; the bigger configuration overfits the observed rows | 800 trees 3.258; 800 2-pass 3.238; 1500 trees/63 leaves 3.241 |
| **D17** | **Keep augmentation at exactly the observed test rate (×1.0)** | Confirms the original choice with evidence. Training on conditions harsher than reality degrades the learnable signal faster than it builds robustness | ×0.5 **3.230**, ×1.0 **3.226**, ×1.5 **3.231**, ×2.0 **3.238** |
| **D18** | **Adopt 5-seed averaging** (the one accepted change) | The augmentation draw is random, so the seed is an arbitrary choice; averaging removes it. Variance reduction within a single model class, so it does not contradict rejecting the heterogeneous ensemble. Gain is small and inside fold noise — adopted because it is near-risk-free, not claimed as material | 1 seed 3.2257 → 3 seeds 3.2188 → **5 seeds 3.2176** |
| — | *Not adopted:* 10-fold CV | Scores better (3.2173) but this changes the **measurement**, not the model — the final model trains on 100% of data regardless. Treated as a better estimate, not an improvement | — |

### Interpretation

Eleven rejections out of twelve is itself a finding: with `previous_usage` correlating 0.945 with
the target, the remaining error is largely irreducible, concentrated in genuinely volatile
buildings. Effort was redirected to the technical report and error analysis, which carry 40% of
the Round 1 score.

---

## Phase 6–8 — further optimisation

| # | Decision | Justification | Evidence |
|---|---|---|---|
| **D19** | **Adopt `temperature × hour` interactions** (`temp × hour_sin/cos`, `temp²`, `temp × occupancy`) | Cooling load depends on heat *and* time of day jointly — a hot afternoon is not a hot 3am. This was the one feature family never encoded | 3.2458 → **3.2399** |
| **D20** | **Adopt the transductive imputer** — fit the per-column imputers on train **and test features combined** | The imputer predicts *features from features*; the target is never involved, so using unlabelled test rows is legitimate. Gives it 11,000 rows instead of 8,000 **and** direct exposure to the distribution it must impute into. Since imputation was our highest-leverage component, improving it compounds | 3.2199 → **3.2086**; with 5 seeds → **3.1994** |
| **D21** | Reject residual boosting, target encoding, per-building models, building×month, occupancy transforms | Ridge's residuals carry no learnable structure; the rest duplicate what the building×hour terms already express | residual boost 3.257/3.284; target encoding 3.246; per-building 3.249; building×month 3.257; occupancy transforms 3.246 |
| **D22** | **Measure covariate shift, then deliberately leave it uncorrected** | A train-vs-test classifier reaches **AUC 0.688**. This is *not* a missingness artefact — removing the flags leaves it at 0.689, and complete-cases-only still gives **0.654**. But all four importance-weighting variants made things worse. Covariate shift only induces bias under misspecification; with a stable `P(energy│features)`, reweighting removes no bias and adds variance. Raw weights were also pathological (0.005 to 164, median 0.21) | unweighted 3.226 vs weighted 3.280–3.510 |
| **D23** | Reject pseudo-labelling | Marginally helpful alone (3.2431), but *harmful* once the imputer went transductive (3.2110 vs 3.2086) — both exploit the same unlabelled information, so they compete rather than compound | — |

---

## Platform compatibility — the submissions folder

The organisers' template folder (`requirements-image.txt`, `template_notebook.ipynb`,
`sample_submission.csv`) revealed three issues that would have caused failure or a zero.

| # | Decision | Justification |
|---|---|---|
| **D24** | **Serialise with plain `joblib`, not `cloudpickle`** | `cloudpickle` is **not in the platform image**. A by-value pickle needs it at load time, so `model.pkl` would have raised `ModuleNotFoundError`. The prediction notebook now defines `make_features` and `SubModelImputer` itself, which is exactly how the official template expects feature engineering to be carried across |
| **D25** | **Train against the platform library versions** (sklearn 1.5.2, pandas 2.2.3, lightgbm 4.5.0) | We had built against sklearn 1.9.1 / pandas 3.0.6. New pickles frequently fail to load under older libraries; the reverse is usually safe. Verified the scores are **identical** across both environments (RMSE 3.1994 either way), so this costs nothing and removes a failure mode |
| **D26** | **Add a separate prediction notebook** (`kkboys_Track1_Prediction_Notebook.ipynb`) | The platform runs the uploaded notebook in a sandbox with **no internet and no `train.csv`** — it must only load the model and predict. Our training notebook is the artefact judges read; this is the artefact the platform executes |
| **D27** | **Leaderboard CSV uses `id,prediction`; sandbox output uses `prediction` only** | Both formats are correct for different purposes. `sample_submission.csv` has `id,prediction` for the leaderboard upload; the platform's test file has no ID column and shuffled rows, so the sandbox writes one prediction per row in input order. Verified our ids and ordering match the official sample exactly |

**Sandbox verification:** executed the prediction notebook under platform library versions, with no
`train.csv`, a test file stripped of its ID column and shuffled, and `DATATHON_INPUT_PATH` /
`DATATHON_OUTPUT_PATH` set. Ran clean, wrote 3,000 predictions.

---

## Final result

**Ridge (α ≈ 1.2–1.7) on 370 building-aware features, with transductive sub-model imputation,
trained on missingness-augmented data, averaged over 5 seeds.**

| Metric | Realistic validation |
|---|---|
| **RMSE** | **3.1994** |
| MAE | 2.5072 |
| R² | 0.9699 |

| Reference | RMSE |
|---|---|
| Predict global mean | 18.45 |
| Predict building mean | 13.22 |
| Copy `previous_usage` (no model) | 6.40 |
| Earlier team blend, measured honestly | 3.745 |
| **Final model** | **3.199** |

**33 variations tested, 4 kept** (sub-model imputation, missingness augmentation, `temp × hour`,
transductive imputation + seed averaging).

---

## Deliverables

| File | Status |
|---|---|
| `Track1_Smart_Campus_Analytics.ipynb` | Full pipeline, executes end-to-end with zero errors |
| `model.pkl` | Seed-averaged final model, `cloudpickle` by-value, verified in a clean environment |
| `report/technical_report.pdf` | 1 page, verified |
| `requirements.txt` | All notebook dependencies pinned |
| `predict_from_model.py` | Standalone inference script |

### Open items

- Submission CSV currently has a single `prediction` column and no `id`, matching the booklet's
  example code — but the workshop's `sample_submission.csv` used `id,prediction`. **Confirm with
  organisers**; a format mismatch would score zero silently.
- Error analysis by building (high-variance buildings like LectureHall dominate residual error)
  is not yet written up — it is explicitly named in Track 1's judging focus.

---

## Phase 9 — test-matched validation & v2 model (2026-10-03, in progress)

| # | Decision | Justification | Evidence |
|---|---|---|---|
| **D28** | **Replace independent per-column NaN injection with test-matched gap patterns** (validation *and* training augmentation) | Test gaps are clustered by count, not independent: 81.8% none / 10.5% one / **6.9% two** / 0.8% three, with the gapped columns uniform and unrelated to building/hour/values. Independent injection gives only ~2.6% two-gap rows, so it under-weights exactly the hardest rows ~3× | Pair lifts over independence in test: 2.7–4.0×. Current production model scores **3.338** under test-matched validation (vs 3.20 under the old protocol) |
| **D29** | **NaN-native imputer**: co-missing predictors passed to the per-column LightGBMs as NaN (not median-filled), plus 15% predictor-gap augmentation when fitting them | The old imputer filled a co-missing driver with its global median before imputing, so a row missing occupancy *and* prev imputed each from a fake value of the other | 3.338 → **3.291** (mean of 2 CV reps); occ+prev rows 7.45/6.70 → 6.38/5.55 |
| **D30** | **Blend Ridge with an MLP and a native-NaN LightGBM** (fixed NNLS weights 0.66/0.22/0.12) | Complete-row linear modelling is saturated (14 feature families, stacking, transition-hour terms all ≥ Ridge), but an MLP is a genuinely different error source | 3.291 → **3.265** held out across reps (weights fit on one rep, scored on the other) |
| **D31** | **Rows missing `previous_usage` get their own weights, leaning ~48% on a Ridge trained without prev** | When the strongest feature is absent, a model that never relied on it beats imputing it. Two groups only: a 7-group/pair-level scheme scored the same (3.2572) with ~25 more weights and visibly overfit on ~100-row groups | **3.258** held out; weights stable across reps |
| **D32** | Reject scenario-regime features and scenario-weighted training | The test oversamples generated scenarios ~5× (heatwave 34–36 °C at 12–16h: 0.9%→4.3%; storm humidity ≥95: 0.9%→4.3%; night prev-drop: 0.6%→3.5%; occupancy surges: 2%→10%). Their extra error is noise, not missed structure | Indicator/interaction features and test-ratio sample weights all worse even on a test-weighted metric (3.094–3.147 vs 3.089) |

**Net (test-matched CV, before 5-seed averaging): 3.338 → 3.258 (−0.081).**
Pending: production-class CV check (`train_final_model.py --cv`), final 5-seed `model.pkl`, sandbox run of the
regenerated prediction notebook, report update.
