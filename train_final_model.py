"""
Train the final Track 1 model and save model.pkl
================================================
Run this with the PLATFORM library versions (requirements-image.txt) so the saved
pickle loads cleanly in the evaluation sandbox:

    python -m venv .venv_platform
    .venv_platform/Scripts/pip install -r requirements-image.txt     (bin/ on mac/linux)
    .venv_platform/Scripts/python train_final_model.py               # train + save model.pkl
    .venv_platform/Scripts/python train_final_model.py --cv          # test-matched CV of the full model

Model (v2):
  * Ridge on 370 building-aware features, imputed by a NaN-native sub-model imputer
    (one LightGBM per gap-prone column, fitted transductively on train+test FEATURES;
    co-missing predictors are passed as NaN instead of median-filled)          -- D28/D29
  * blended with an MLP and a native-NaN LightGBM (fixed weights from CV)       -- D30
  * rows missing previous_usage additionally lean on a Ridge trained WITHOUT it -- D31
  * every component trained on test-matched missingness augmentation, 5 seeds averaged

Saves (model, feature_names) as a plain joblib pickle. The prediction notebook defines
the classes below itself (no cloudpickle in the platform image -- D24).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

RANDOM_STATE = 42
N_SEEDS = 5

BUILDINGS = ["ADM_A", "BUS_A", "BUS_B", "ENG_A", "ENG_B", "LEC_A",
             "LIB_A", "RES_A", "RES_B", "SCI_A", "SCI_B", "SPT_A"]
DOW_MAP = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
NUMERIC = ["temperature", "humidity", "occupancy", "previous_usage"]
HELPER = ["hour", "month", "dow", "weekend", "building_code"]
DENSE = (["hour_sin", "hour_cos", "dow", "weekend", "month_sin", "month_cos"] + NUMERIC
         + [c + "_missing" for c in NUMERIC] + ["n_missing"] + [f"b_{b}" for b in BUILDINGS])

# Blend weights, fitted by non-negative least squares on out-of-fold predictions under the
# test-matched validation protocol (stable across two independent CV repetitions -- D30/D31).
BLEND_WEIGHTS = {
    "prev_observed": {"ridge": 0.655, "mlp": 0.234, "lgbm": 0.111},
    "prev_missing":  {"ridge": 0.313, "mlp": 0.141, "lgbm": 0.071, "ridge_no_prev": 0.475},
}


# ---------------------------------------------------------------------------
# Feature engineering  (must be copied verbatim into the prediction notebook)
# ---------------------------------------------------------------------------
def make_features(df):
    """Pure: learns nothing from data, so train and test transform identically."""
    df = df.copy()
    X = pd.DataFrame(index=df.index)

    X["hour"] = df["hour"]
    X["month"] = df["month"]
    X["dow"] = df["day_of_week"].map(DOW_MAP)
    X["weekend"] = (X["dow"] >= 5).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    X["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for c in NUMERIC:
        X[c] = df[c]
        X[c + "_missing"] = df[c].isna().astype(int)
    X["n_missing"] = df[NUMERIC].isna().sum(axis=1)

    X["building_code"] = df["building_id"].map({b: i for i, b in enumerate(BUILDINGS)})

    extra = {}
    for b in BUILDINGS:
        is_b = (df["building_id"] == b).astype(float)
        extra[f"b_{b}"] = is_b
        extra[f"b_{b}_weekend"] = is_b * X["weekend"]
        for src, nm in [("occupancy", "occ"), ("temperature", "temp"), ("previous_usage", "prev")]:
            extra[f"b_{b}_{nm}"] = is_b * df[src]
        for h in range(24):
            extra[f"b_{b}_h{h}"] = is_b * (df["hour"] == h).astype(float)

    # temperature x time-of-day: cooling load depends on heat and hour jointly
    t = df["temperature"]
    extra["temp_x_hour_sin"] = t * X["hour_sin"]
    extra["temp_x_hour_cos"] = t * X["hour_cos"]
    extra["temp_sq"] = t ** 2
    extra["temp_x_occ"] = t * df["occupancy"]

    return pd.concat([X, pd.DataFrame(extra, index=df.index)], axis=1)


def without_prev(feature_names):
    """Feature list with previous_usage and every term derived from it removed."""
    return [c for c in feature_names if c != "previous_usage" and not c.endswith("_prev")]


def compact_view(Xf):
    """Raw drivers (gaps left as NaN) + calendar + building, for the native-NaN LightGBM."""
    C = Xf[["building_code", "hour", "dow", "month", "weekend"]].copy()
    for c in NUMERIC:
        C[c] = Xf[c].to_numpy()
    return C


class SubModelImputer(BaseEstimator, TransformerMixin):
    """Predict each missing value with a dedicated LightGBM trained on the rows where that column
    was observed, then regenerate the derived interaction terms.

    * `extra_frame`: unlabelled rows (the test FEATURES) the per-column models may also learn from.
      Only features are used -- never the target.
    * Co-missing predictors are passed to LightGBM as NaN (native missing-value routing) rather than
      median-filled. A median is wrong in exactly the hard case: a row missing both occupancy and
      previous_usage would impute each from a fake median of the other.
    * `aug_pred`: each per-column model also trains on a copy of its data with that fraction of the
      *predictors* knocked out, so the NaN routing is learned from enough examples.
    """

    def __init__(self, extra_frame=None, aug_pred=0.15, seed=RANDOM_STATE):
        self.extra_frame = extra_frame
        self.aug_pred = aug_pred
        self.seed = seed

    def fit(self, X, y=None):
        src = X if self.extra_frame is None else pd.concat([X, self.extra_frame], ignore_index=True)
        rng = np.random.default_rng(self.seed)
        self.models_ = {}
        for c in NUMERIC:
            others = [o for o in NUMERIC if o != c]
            feats = HELPER + others
            observed = src[c].notna()
            Xt, yt = src.loc[observed, feats].copy(), src.loc[observed, c]
            if self.aug_pred > 0:
                Xd = Xt.copy()
                for o in others:
                    Xd.loc[rng.random(len(Xd)) < self.aug_pred, o] = np.nan
                Xt, yt = pd.concat([Xt, Xd]), pd.concat([yt, yt])
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
            self.models_[c] = (m.fit(Xt, yt), feats)
        return self

    def transform(self, X):
        X = X.copy()
        raw = X[NUMERIC].copy()
        for c in NUMERIC:
            gaps = raw[c].isna()
            if not gaps.any():
                continue
            m, feats = self.models_[c]
            others = [o for o in NUMERIC if o != c]
            tmp = X.loc[gaps, feats].copy()
            tmp[others] = raw.loc[gaps, others]          # co-missing predictors stay NaN
            X.loc[gaps, c] = m.predict(tmp)

        # derived terms were built from the raw (gappy) columns -- rebuild them
        for b in BUILDINGS:
            is_b = X[f"b_{b}"]
            for src, nm in [("occupancy", "occ"), ("temperature", "temp"), ("previous_usage", "prev")]:
                X[f"b_{b}_{nm}"] = is_b * X[src]
        X["temp_sq"] = X["temperature"] ** 2
        X["temp_x_occ"] = X["temperature"] * X["occupancy"]
        X["temp_x_hour_sin"] = X["temperature"] * X["hour_sin"]
        X["temp_x_hour_cos"] = X["temperature"] * X["hour_cos"]
        return X.fillna(0.0)


class SeedMember(BaseEstimator):
    """The four components fitted on one augmented copy of the training data, sharing one imputer."""

    def __init__(self, extra_frame=None, seed=0):
        self.extra_frame = extra_frame
        self.seed = seed

    def fit(self, Xf, y, Xf_lgbm):
        feats = list(Xf.columns)
        self.imputer_ = SubModelImputer(extra_frame=self.extra_frame, seed=RANDOM_STATE + self.seed).fit(Xf)
        F = self.imputer_.transform(Xf)
        ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3.5, 30)))
        self.cols_ = {"ridge": feats, "ridge_no_prev": without_prev(feats), "mlp": DENSE}
        self.ridge_ = ridge().fit(F[self.cols_["ridge"]], y)
        self.ridge_no_prev_ = ridge().fit(F[self.cols_["ridge_no_prev"]], y)
        self.mlps_ = [make_pipeline(StandardScaler(), MLPRegressor(
                          hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=600, early_stopping=True,
                          random_state=100 * self.seed + s)).fit(F[DENSE], y) for s in range(3)]
        self.lgbm_ = lgb.LGBMRegressor(
            n_estimators=2500, learning_rate=0.01, num_leaves=15, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=RANDOM_STATE + self.seed, verbose=-1, n_jobs=-1,
        ).fit(compact_view(Xf_lgbm), y, categorical_feature=["building_code"])
        return self

    def predict_components(self, Xf):
        F = self.imputer_.transform(Xf)
        return {
            "ridge": self.ridge_.predict(F[self.cols_["ridge"]]),
            "ridge_no_prev": self.ridge_no_prev_.predict(F[self.cols_["ridge_no_prev"]]),
            "mlp": np.mean([m.predict(F[DENSE]) for m in self.mlps_], axis=0),
            "lgbm": self.lgbm_.predict(compact_view(Xf)),
        }


class BlendedEnergyModel(BaseEstimator):
    """Seed-averaged components, blended with fixed weights; rows missing previous_usage use their own
    weight set, which leans on the Ridge that never saw previous_usage."""

    def __init__(self, members, weights, feature_names):
        self.members = members
        self.weights = weights
        self.feature_names = feature_names

    def predict(self, Xf):
        Xf = Xf[self.feature_names]
        parts = [m.predict_components(Xf) for m in self.members]
        comp = {k: np.mean([p[k] for p in parts], axis=0) for k in parts[0]}
        prev_missing = Xf["previous_usage_missing"].to_numpy() == 1
        pred = np.empty(len(Xf))
        for group, mask in [("prev_observed", ~prev_missing), ("prev_missing", prev_missing)]:
            pred[mask] = sum(w * comp[k][mask] for k, w in self.weights[group].items())
        return pred


# ---------------------------------------------------------------------------
# Test-matched missingness augmentation
# ---------------------------------------------------------------------------
def gap_count_probs(test_df):
    """P(row has k gaps) in the test file. Test gaps are clustered (6.9% of rows have two) and the
    gapped columns are uniform -- independent per-column injection under-represents multi-gap rows 3x."""
    k = test_df[NUMERIC].isna().sum(axis=1).value_counts(normalize=True)
    return np.array([k.get(i, 0.0) for i in range(len(NUMERIC) + 1)])


def inject_test_like(raw, probs, rng, scale=1.0):
    out = raw.copy()
    p = probs.copy()
    p[1:] *= scale
    p[0] = 1 - p[1:].sum()
    k = rng.choice(len(p), size=len(out), p=p)
    vals = out[NUMERIC].to_numpy(dtype=float, copy=True)
    for i in np.where(k > 0)[0]:
        vals[i, rng.choice(len(NUMERIC), size=k[i], replace=False)] = np.nan
    out[NUMERIC] = vals
    return out


def fit_blended(train_df, y, test_features, probs, n_seeds=N_SEEDS, verbose=True):
    feature_names = list(test_features.columns)
    members = []
    for s in range(n_seeds):
        rng = np.random.default_rng(RANDOM_STATE + 100 * s)
        X_aug = make_features(inject_test_like(train_df, probs, rng))[feature_names]
        X_lgbm = make_features(inject_test_like(train_df, probs, rng, scale=2.0))[feature_names]
        members.append(SeedMember(extra_frame=test_features, seed=s).fit(X_aug, y, X_lgbm))
        if verbose:
            print(f"  seed {s + 1}/{n_seeds} fitted (ridge alpha={members[-1].ridge_[-1].alpha_:.3f})", flush=True)
    return BlendedEnergyModel(members, BLEND_WEIGHTS, feature_names)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA = "Track 1 Dataset"
    train = pd.read_csv(os.path.join(DATA, "Track 1 Training Dataset.csv"))
    test = pd.read_csv(os.path.join(DATA, "Track 1 Testing Dataset.csv"))
    y = train["energy_usage"].to_numpy()
    probs = gap_count_probs(test)
    print("test gap-count distribution:", {k: f"{p * 100:.2f}%" for k, p in enumerate(probs)})
    test_features = make_features(test)          # unlabelled, for the transductive imputer

    if "--cv" in sys.argv:
        # Test-matched protocol: 5-fold CV, validation folds get test-like gap patterns.
        from sklearn.model_selection import KFold
        n_seeds = int(sys.argv[sys.argv.index("--cv") + 1]) if len(sys.argv) > sys.argv.index("--cv") + 1 else 1
        kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
        rng_v = np.random.default_rng(1000)
        oof = np.zeros(len(train))
        for f, (tr_i, va_i) in enumerate(kf.split(train)):
            raw_va = inject_test_like(train.iloc[va_i], probs, rng_v)
            model = fit_blended(train.iloc[tr_i].reset_index(drop=True), y[tr_i], test_features, probs,
                                n_seeds=n_seeds, verbose=False)
            oof[va_i] = model.predict(make_features(raw_va))
            print(f"  fold {f}: RMSE {np.sqrt(np.mean((oof[va_i] - y[va_i]) ** 2)):.4f}", flush=True)
        rmse = np.sqrt(np.mean((oof - y) ** 2))
        mae = np.mean(np.abs(oof - y))
        r2 = 1 - np.sum((oof - y) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"test-matched CV ({n_seeds} seed/s): RMSE {rmse:.4f}  MAE {mae:.4f}  R2 {r2:.4f}")
        sys.exit(0)

    model = fit_blended(train, y, test_features, probs)
    joblib.dump((model, model.feature_names), "model.pkl")
    print(f"\nsaved model.pkl ({os.path.getsize('model.pkl') / 1e6:.2f} MB), "
          f"{len(model.members)} seed members x {len(model.feature_names)} features")

    preds = model.predict(test_features)
    print(f"sanity: {len(preds)} predictions, mean {preds.mean():.2f}, "
          f"range [{preds.min():.2f}, {preds.max():.2f}], finite={np.isfinite(preds).all()}")
