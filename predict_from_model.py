"""
Inference script — Track 1: Smart Campus Analytics
==================================================
Loads model.pkl and produces the submission CSV.

This mirrors exactly what the official upload instructions describe: load the saved
model in a fresh environment, apply the SAME feature engineering used in training,
predict on the test set, write the CSV.

The custom transformer inside model.pkl was serialised BY VALUE, so you do NOT need
to redefine any model classes here — joblib.load gives you a ready-to-use object.
You only need `make_features` below, which is the pure feature-engineering function.

Usage:
    python predict_from_model.py
"""
import os
import numpy as np
import pandas as pd
import joblib

# --------------------------------------------------------------------------
# 1. Feature engineering — must match training EXACTLY
# --------------------------------------------------------------------------
BUILDINGS = ["ADM_A", "BUS_A", "BUS_B", "ENG_A", "ENG_B", "LEC_A",
             "LIB_A", "RES_A", "RES_B", "SCI_A", "SCI_B", "SPT_A"]
DOW_MAP = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
NUMERIC = ["temperature", "humidity", "occupancy", "previous_usage"]


def make_features(df):
    """Pure feature construction — learns nothing from the data, so train and test
    are transformed identically. All learned steps live inside the pipeline."""
    df = df.copy()
    X = pd.DataFrame(index=df.index)

    # time features
    X["hour"] = df["hour"]
    X["month"] = df["month"]
    X["dow"] = df["day_of_week"].map(DOW_MAP)
    X["weekend"] = (X["dow"] >= 5).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    X["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # raw drivers + explicit missingness signals
    for c in NUMERIC:
        X[c] = df[c]
        X[c + "_missing"] = df[c].isna().astype(int)
    X["n_missing"] = df[NUMERIC].isna().sum(axis=1)

    X["building_code"] = df["building_id"].map({b: i for i, b in enumerate(BUILDINGS)})

    # per-building intercept, weekend offset, driver slopes, and 24h profile
    extra = {}
    for b in BUILDINGS:
        is_b = (df["building_id"] == b).astype(float)
        extra[f"b_{b}"] = is_b
        extra[f"b_{b}_weekend"] = is_b * X["weekend"]
        for src, nm in [("occupancy", "occ"), ("temperature", "temp"), ("previous_usage", "prev")]:
            extra[f"b_{b}_{nm}"] = is_b * df[src]
        for h in range(24):
            extra[f"b_{b}_h{h}"] = is_b * (df["hour"] == h).astype(float)

    return pd.concat([X, pd.DataFrame(extra, index=df.index)], axis=1)


# --------------------------------------------------------------------------
# 2. Load model and test data
# --------------------------------------------------------------------------
TEAM_NAME = "kkboys"

def locate(fname, extra_dirs=()):
    for d in list(extra_dirs) + ["Track 1 Dataset", ".", "/content"]:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{fname} not found")

model, feature_names = joblib.load(locate("model.pkl", extra_dirs=["."]))

input_path = os.environ.get("DATATHON_INPUT_PATH") or locate("Track 1 Testing Dataset.csv")
test_df = pd.read_csv(input_path)
print(f"loaded {len(test_df)} test rows from {input_path}")

# --------------------------------------------------------------------------
# 3. Predict
# --------------------------------------------------------------------------
X_test = make_features(test_df)
missing_cols = set(feature_names) - set(X_test.columns)
if missing_cols:
    raise RuntimeError(f"feature mismatch: {sorted(missing_cols)[:5]}")

preds = model.predict(X_test[feature_names])

# --------------------------------------------------------------------------
# 4. Validate and save
# --------------------------------------------------------------------------
assert len(preds) == len(test_df), "row count mismatch"
assert np.isfinite(preds).all(), "non-finite predictions"

out = pd.DataFrame({"prediction": preds})
out_name = f"{TEAM_NAME}_Datathon 2026_Track 1_Prediction.csv"
out.to_csv(out_name, index=False)

print(f"\nwrote {out_name}  ({out.shape[0]} rows)")
print(out["prediction"].describe().round(3).to_string())
