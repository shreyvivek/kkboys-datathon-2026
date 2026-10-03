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
