# Track 1 — Reference Notes

## 1. Interactions

An **interaction** is a new column made by multiplying two features. It lets a linear model learn a
*separate* effect of one feature for each value of another (here: for each building).

```python
is_b = (df["building_id"] == b)              # 1 if row is building b, else 0
X[f"b_{b}_temp"] = is_b * df["temperature"]  # temperature, but only in building b's rows
```

A SCI_A row at 30°C → `b_SCI_A_temp = 30`, all other `b_*_temp = 0`.
Ridge learns one weight per column → each building gets its own temperature slope.

### Per-building interactions in `make_features()`

| Interaction | Columns | What Ridge learns | Why it makes sense here |
|---|---|---|---|
| building (one-hot) | 12 | Each building's baseline level | Science avg ~75, Admin ~31 |
| building × hour | 12 × 24 = 288 | Each building's own 24-hour load profile | Sports peaks evening, offices peak midday |
| building × temperature | 12 | Each building's cooling sensitivity | Labs/server rooms react differently from dorms |
| building × occupancy | 12 | Energy per extra person, per building | Residential has high occupancy (~171) but mid energy |
| building × previous_usage | 12 | How strongly each building's usage carries over | Some buildings are steadier than others |
| building × weekend | 12 | Each building's weekend shift | Offices drop on weekends, dorms don't |

Total features: 365. Result: Ridge + interactions RMSE 3.36 vs gradient boosting 3.56 (5-fold CV),
which suggests the data is mostly "each building follows its own linear rules".

**Trees (RF, gradient boosting) don't need these** — they find interactions themselves by splitting
on building first, then on temperature etc. Interactions are mainly for linear models.

### Is this always done for Ridge?

No. Plain Ridge assumes each feature has **one** effect for every row. You add interactions only when
you have reason to believe that assumption is wrong for a specific pair of features. Adding every
possible pair blows up the column count (overfitting, slow, hard to explain), so it's a deliberate choice.

### How to decide which interactions to build

1. **Domain logic first.** Ask "would feature A's effect plausibly differ by group B?"
   Building type changing the temperature/occupancy/hour effect is an obvious yes.
2. **Check it in EDA.** Plot the target vs feature A, one line per group B.
   Different slopes or shapes → interaction is worth trying. (Our "mean energy by hour per building type"
   plot shows clearly different shapes → building × hour.)
3. **Let a tree model hint.** If gradient boosting beats plain Ridge by a lot, it's finding interactions
   Ridge can't. Feature importance / SHAP interaction values show which pairs matter.
4. **Validate with CV.** Add one interaction group at a time and keep it only if CV RMSE improves.
5. **Prefer a low-cardinality grouping column.** 12 buildings × 1 feature = 12 columns is fine;
   interacting two continuous features or two many-level categoricals explodes quickly.
6. **Regularisation is your safety net.** Ridge's penalty shrinks useless interaction weights toward 0,
   which is why 365 columns on 8,000 rows doesn't overfit badly. Still, don't add them blindly.

### Gotcha: missing values inside interactions

`0 × NaN = NaN`, so if `previous_usage` is missing, **all 12** `b_*_prev` columns become NaN.
The median imputer then fills each with its column median, which is **0** (~92% of each column is zeros
from the other buildings) → the building-specific effect silently switches off for that row.
**Fix:** fill missing values *before* building interactions.

## 2. Missing values & the "was missing" flag

- `previous_usage` missing: train 1.4%, **test 6.8%** → these rows matter more on the leaderboard.
- Those rows had ~2× error (RMSE 6.5 vs 3.2) with plain median filling.
- `X[c + "_missing"] = df[c].isna().astype(int)` — a 0/1 column, used like any other feature.
  - **Gradient boosting** can split on the flag and use a different rule for flagged rows → genuinely relies less on `previous_usage`.
  - **Ridge** can only add a constant shift for flagged rows; it can't change the weight on `previous_usage`
    unless you add a `flag × previous_usage` interaction.

### Imputation experiment (~8% of `previous_usage` masked, Ridge+HGB blend, 5-fold CV)

| Strategy | Overall RMSE | Missing-prev rows | Rows with prev |
|---|---|---|---|
| Global median (current) | 3.396 | 4.779 | 3.243 |
| Group median (building × hour) | 3.394 | 4.789 | 3.239 |
| Model imputation (Ridge predicts prev) | 3.328 | **4.031** | 3.258 |
| KNN imputer (all columns) | **3.298** | 4.295 | **3.193** |
| Iterative imputer (raw columns, before interactions) | **3.298** | 4.266 | 3.197 |
| Separate model for missing rows | 3.304 | **3.922** | 3.243 |

- Iterative imputer on all 365 columns was too slow (killed after 30 min); re-run on the 4 raw numeric
  columns + building/hour/weekend, filled *before* `make_features()`.
- **Separate model** is best on the missing rows (−18% vs median). **KNN / iterative** are best on rows that
  have prev, because they also fill temperature/humidity/occupancy properly.
- Likely best: iterative/KNN fill for temperature, humidity, occupancy + separate model when prev is missing.

- **KNN imputer:** fills a gap with the average of the 10 most similar rows. Mostly matches on building + hour
  (those columns dominate the distance). Stores the training set inside `model.pkl`.
- **Model imputation:** a Ridge model trained (on rows where prev is known) to predict `previous_usage`
  from everything except prev and the target. It explains R² ≈ 0.83 of `previous_usage` (RMSE 7.8),
  vs building alone 0.43 and building + hour 0.74.
- Correlation with `previous_usage`: temperature 0.46, occupancy 0.37, hour 0.25, humidity −0.23, month −0.06.

## 3. Residual diagnostic — is the model at the noise floor?

Took the out-of-fold errors of Shrey's final Ridge (rows with all inputs present) and tried to
predict them with LightGBM. If it can, there's structure left; if not, the rest is noise.

| Residual model | R² on the errors |
|---|---|
| LightGBM shallow | −0.016 |
| LightGBM deeper | −0.074 |
| Null check (shuffled errors) | −0.020 |

→ **No learnable structure left.** Core RMSE ≈ 3.06 on complete rows is noise.
- Clean-input RMSE 3.112 vs realistic 3.233 → missing values cost only ~0.12 RMSE in total
  (the ceiling for *every* missing-value idea).
- Error grows with load (RMSE 2.4 at low prev → ~3.7 at high; SCI buildings worst) → noise scales
  with the building's level. Explains per-building differences; can't be modelled away.
- Building × month offsets range −1.1 to +1.6 — consistent with noise (~60 rows per cell).

## 4. Temperature curvature (hinge / square) — TO DISCUSS

**Finding:** the model under-predicts at both temperature extremes (U-shape in residuals):

| Temperature | Rows | Mean error (actual − predicted) |
|---|---|---|
| ≤ 25 °C | 233 | +0.76 |
| 25–31 °C | ~6,900 | ≈ 0 |
| > 31 °C | ~405 | +0.65 to +0.70 |

Physical story: heating/cooling kicks in past a comfort band, so energy rises faster at both ends
than a straight-line temperature effect allows.

**Candidate features:**
- **Hinge:** `temp_cold = max(0, 25 − T)`, `temp_hot = max(0, T − 31)` — zero inside the band, grows linearly outside.
- **Square:** `temp_sq = (T − 28)²` — one smooth U-shaped term, no thresholds to choose.
- **Per-building hinges:** the hinges × each building (24 cols).

**Implementation gotcha:** if temperature is missing, the extra terms would be NaN → 0 (same bug as the
slope columns). They must be rebuilt from the *filled* temperature inside the imputer.

**Result** (Shrey's pipeline + the extra term, 5× repeated realistic CV, paired vs base):

| Variant | RMSE | Extreme-temp RMSE | Extreme-temp bias | Paired diff | Better in |
|---|---|---|---|---|---|
| Base (Shrey) | 3.2323 | 3.452 | +0.67 | — | — |
| Hinge (25/31) | 3.2277 | 3.412 | +0.33 | −0.0046 | 5/5 |
| **Square** | **3.2272** | **3.409** | **+0.31** | **−0.0051** | **5/5** |
| Per-building hinges | 3.2325 | 3.468 | +0.34 | +0.0002 | 3/5 |

**Takeaways:**
- Small (−0.005 RMSE) but **consistent: better in every repeat**, and it halves the bias at extreme temperatures.
- **Prefer the square term:** equally good, one feature, and no thresholds. The 25/31 hinge thresholds were
  read off these same residuals, so the hinge result is slightly optimistic.
- Per-building version adds 24 columns for nothing → reject.
- May matter more on the private test ("edge cases" per booklet) if it has hot/cold spells.
- Not yet in the team notebook. Adding it means changing `make_features` (notebook **and**
  `predict_from_model.py`), the imputer's rebuild step, then retraining `model.pkl`.

## 5. Validation — why random K-fold, and how we decide a change is real

**5-fold CV:** split the 8,000 rows into 5 chunks; train on 4, predict the 5th, rotate until every row
has an out-of-fold prediction from a model that never saw it. CV is only for *choosing and measuring* —
the shipped `model.pkl` is retrained on all rows.

### Why a random split, not a time split (D2)

The workshop rule "never split time-ordered data randomly" only applies if the rows *are* time-ordered.
We tested it and found no time signal:

- `id` ↔ every feature: correlations all within **±0.016**
- Target is flat across `id` deciles
- `id`-ordered holdout RMSE **3.606** vs random 5-fold **3.560 ± 0.071** → difference is inside fold noise

So there is no leakage to defend against. A time split would only cost us 20% of the training
data and give a noisier estimate. **Q&A line:** "We didn't assume — we checked for a time signal, found
none, and that's why we used shuffled K-fold."

### Fold noise — the ruler for every comparison

The **± 0.071** is how much RMSE moves from fold to fold just from *which rows* land in the test fold.
Any difference smaller than the noise can't be trusted from a single CV run.

To measure small changes anyway, we use **paired, repeated CV**: run the base and the candidate on the
**same folds**, repeat with 5 seeds, and look at (a) the average difference and (b) how many of the 5
repeats the candidate wins. Comparing on the same folds cancels most of the "which rows" noise, so the
bar for a *difference* is much lower than ±0.071. **Consistency across repeats is the test**:

| Change | Paired diff | Wins | Verdict |
|---|---|---|---|
| Temperature square term (§4) | −0.0051 | 5/5 | Small but real — consistent |
| Per-building hinges (§4) | +0.0002 | 3/5 | Noise → reject (24 columns for nothing) |
| Dedicated no-prev model (D13) | +0.0087 | 1/5 | Worse → reject |
| Averaging no-prev model + Ridge on missing rows (D13) | −0.0006 | 3/5 | Noise → reject (extra moving part) |

Rule: **keep a change only if it wins consistently, not just on average** — and extra complexity has to
pay for itself ("best model ≠ most complex model").

### Related trap: an optimistic estimate (D11)

The 4-model stack looked better (3.2259) only because its blend weights were fitted on the same
predictions they were scored on. Scored honestly with **nested CV** (weights fitted on inner folds only) it
was **3.2605 — worse** than Ridge alone (3.2326). Same lesson: the validation must not see what it's grading.

*(Not to be confused with §3's "noise floor" — that's the irreducible error in the data itself, ≈ 3.06
RMSE. Fold noise is uncertainty in our **measurement** of RMSE.)*

## Log target, weighted Ridge, neural network (MLP) — all rejected

Shrey's pipeline, 5× repeated realistic CV, paired vs base (3.2323):

| Variant | RMSE | MAE | Paired diff | Better in |
|---|---|---|---|---|
| **Base (Shrey's Ridge)** | **3.2323** | **2.524** | — | — |
| Log target (exp back-transform) | 3.5800 | 2.722 | +0.348 | 0/5 |
| Log target + smearing correction | 3.5853 | 2.722 | +0.353 | 0/5 |
| Weighted Ridge, w = 1/ŷ | 3.2316 | 2.523 | −0.0007 | 4/5 (noise) |
| Weighted Ridge, w = 1/ŷ² | 3.2370 | 2.526 | +0.0047 | 1/5 |
| MLP (128, 64), early stopping | 4.0449 | 3.171 | +0.813 | 0/5 |

- **Log target is much worse:** the relationships are additive on the raw scale (energy ≈ previous_usage + building/hour
  offsets + slopes). Logging turns them multiplicative, so a linear model on log(energy) no longer fits that structure.
  The noise grows with level, but the *signal* is additive — the transform fixes the wrong thing.
- **Weighting gives nothing:** RMSE scores big-building errors most, so down-weighting them doesn't help the metric.
- **Neural network is far worse (4.04 vs 3.23):** 8,000 rows of tabular data with additive structure is where
  small neural nets lose to explicit linear interactions. Useful as a "we tried deep learning" row for Q&A.
