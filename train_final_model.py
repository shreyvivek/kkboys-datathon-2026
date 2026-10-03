"""
Train the final Track 1 model and save model.pkl
================================================
Run this with the PLATFORM library versions (requirements-image.txt) so the saved
pickle loads cleanly in the evaluation sandbox:

    python -m venv .venv_platform
    .venv_platform/bin/pip install -r requirements-image.txt
    .venv_platform/bin/python train_final_model.py

Model: Ridge on 370 building-aware features, with sub-model imputation
(LightGBM per gap-prone column, fitted transductively on train+test FEATURES),
trained on missingness-augmented data, averaged over 5 seeds.

Saves (models, feature_names) as a plain joblib pickle. No cloudpickle: the
prediction notebook defines SubModelImputer and make_features itself, which is
how the official template expects feature engineering to be carried across.
"""
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42
N_SEEDS = 5

BUILDINGS = ["ADM_A", "BUS_A", "BUS_B", "ENG_A", "ENG_B", "LEC_A",
             "LIB_A", "RES_A", "RES_B", "SCI_A", "SCI_B", "SPT_A"]
DOW_MAP = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
NUMERIC = ["temperature", "humidity", "occupancy", "previous_usage"]
HELPER = ["hour", "month", "dow", "weekend", "building_code"]


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


class SubModelImputer(BaseEstimator, TransformerMixin):
    """Predict each missing value with a dedicated LightGBM trained on the rows
    where that column was observed, then regenerate the derived interaction terms.

    `extra_frame` lets the per-column models additionally see unlabelled rows
    (the test features). Only FEATURES are used -- never the target -- so this is
    legitimate and gives the imputer the test feature distribution.
    """

    def __init__(self, extra_frame=None):
        self.extra_frame = extra_frame

    def fit(self, X, y=None):
        src = X if self.extra_frame is None else pd.concat([X, self.extra_frame], ignore_index=True)
        self.models_, self.median_ = {}, {}
        for c in NUMERIC:
            self.median_[c] = src[c].median()
            observed = src[c].notna()
            feats = HELPER + [o for o in NUMERIC if o != c]
            if observed.sum() < 50:
                self.models_[c] = None
                continue
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
            m.fit(src.loc[observed, feats], src.loc[observed, c])
            self.models_[c] = (m, feats)
        return self

    def transform(self, X):
        X = X.copy()
        for c in NUMERIC:
            gaps = X[c].isna()
            if not gaps.any():
                continue
            if self.models_[c] is None:
                X.loc[gaps, c] = self.median_[c]
            else:
                m, feats = self.models_[c]
                tmp = X.loc[gaps, feats].copy()
                for col in feats:
                    if tmp[col].isna().any():
                        tmp[col] = tmp[col].fillna(self.median_.get(col, 0.0))
                X.loc[gaps, c] = m.predict(tmp)

        # derived terms were built from the raw (gappy) columns -- rebuild them
        for b in BUILDINGS:
            if f"b_{b}" not in X.columns:
                continue
            is_b = X[f"b_{b}"]
            for src, nm in [("occupancy", "occ"), ("temperature", "temp"), ("previous_usage", "prev")]:
                if f"b_{b}_{nm}" in X.columns:
                    X[f"b_{b}_{nm}"] = is_b * X[src]
        if "temp_sq" in X.columns:
            X["temp_sq"] = X["temperature"] ** 2
            X["temp_x_occ"] = X["temperature"] * X["occupancy"]
            X["temp_x_hour_sin"] = X["temperature"] * X["hour_sin"]
            X["temp_x_hour_cos"] = X["temperature"] * X["hour_cos"]
        return X.fillna(0.0)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA = "Track 1 Dataset"
    train = pd.read_csv(os.path.join(DATA, "Track 1 Training Dataset.csv"))
    test = pd.read_csv(os.path.join(DATA, "Track 1 Testing Dataset.csv"))
    TARGET = "energy_usage"
    y = train[TARGET].to_numpy()

    # Training-time missingness augmentation at the observed test rates
    EXTRA_NA_RATE = {
        c: max(0.0, (test[c].isna().mean() - train[c].isna().mean()) / (1 - train[c].isna().mean()))
        for c in NUMERIC
    }
    print("augmentation rates:", {k: f"{v*100:.2f}%" for k, v in EXTRA_NA_RATE.items()})

    def inject(raw, rng):
        out = raw.copy()
        for c, r in EXTRA_NA_RATE.items():
            if r <= 0:
                continue
            idx = rng.choice(len(out), size=int(round(r * len(out))), replace=False)
            out.iloc[idx, out.columns.get_loc(c)] = np.nan
        return out

    test_features = make_features(test)          # unlabelled, for the transductive imputer
    feature_names = list(test_features.columns)

    models = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(RANDOM_STATE + 100 * s)
        X_aug = make_features(inject(train, rng))
        pipe = make_pipeline(
            SubModelImputer(extra_frame=test_features),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-2, 3, 30)),
        )
        pipe.fit(X_aug[feature_names], y)
        models.append(pipe)
        print(f"  seed {s + 1}/{N_SEEDS} fitted (alpha={pipe[-1].alpha_:.3f})")

    joblib.dump((models, feature_names), "model.pkl")
    print(f"\nsaved model.pkl ({os.path.getsize('model.pkl') / 1e6:.2f} MB), "
          f"{len(models)} models x {len(feature_names)} features")

    preds = np.mean([m.predict(test_features[feature_names]) for m in models], axis=0)
    print(f"sanity: {len(preds)} predictions, mean {preds.mean():.2f}, "
          f"range [{preds.min():.2f}, {preds.max():.2f}], finite={np.isfinite(preds).all()}")
