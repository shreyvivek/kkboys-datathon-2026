"""
Generate the leaderboard submission from model.pkl, and audit it.
Loads the model fresh from disk, re-predicts, and validates the output against the
organisers' sample_submission.csv before writing.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# imported so the pickle can resolve the classes it references
from train_final_model import make_features, SubModelImputer, SeedMember, BlendedEnergyModel  # noqa: F401

DATA = "Track 1 Dataset"
OUT = "kkboys_Datathon 2026_Track 1_Prediction.csv"
SAMPLE = "/Users/shreyv/Desktop/track2/submissions/sample_submission.csv"

test = pd.read_csv(os.path.join(DATA, "Track 1 Testing Dataset.csv"))
model, features = joblib.load("model.pkl")
print(f"model.pkl : {type(model).__name__}, "
      f"{len(model.members)} seed members x {len(features)} features")

X = make_features(test)
assert not set(features) - set(X.columns), "feature mismatch"
preds = np.clip(model.predict(X), 0, None)

sub = pd.DataFrame({"id": test["id"], "prediction": preds})

# ---- audit before writing ---------------------------------------------------
checks = {
    "no NaN": not pd.Series(preds).isna().any(),
    "all finite": bool(np.isfinite(preds).all()),
    "no negatives": not (preds < 0).any(),
    "no duplicate ids": not sub["id"].duplicated().any(),
    "ids match test order": bool((sub["id"].values == test["id"].values).all()),
    "row count 3000": len(sub) == 3000,
}
if os.path.exists(SAMPLE):
    samp = pd.read_csv(SAMPLE)
    checks["columns match sample"] = list(sub.columns) == list(samp.columns)
    checks["ids match sample exactly"] = bool((sub["id"].values == samp["id"].values).all())

print("\n--- audit ---")
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
assert all(checks.values()), "AUDIT FAILED - not writing"

# ---- compare against whatever is already committed --------------------------
if os.path.exists(OUT):
    old = pd.read_csv(OUT)
    if len(old) == len(sub) and (old["id"].values == sub["id"].values).all():
        diff = np.abs(old["prediction"].to_numpy() - preds)
        print(f"\nvs existing CSV: max abs diff {diff.max():.8f} "
              f"({'identical' if diff.max() < 1e-6 else 'CHANGED'})")

sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  ({len(sub):,} rows)")

tr = pd.read_csv(os.path.join(DATA, "Track 1 Training Dataset.csv"))
print("\n--- distribution ---")
print(pd.DataFrame({"train target": tr["energy_usage"].describe(),
                    "predictions": sub["prediction"].describe()}).round(2).to_string())
