"""
Generate the leaderboard submission from model.pkl, and audit it.
Loads the model fresh from disk (not from memory), re-predicts, and validates the
output against the organisers' sample_submission.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_final_model import make_features, SubModelImputer  # noqa: F401 (needed to unpickle)

DATA = "Track 1 Dataset"
OUT = "kkboys_Datathon 2026_Track 1_Prediction.csv"
SAMPLE = "/Users/shreyv/Desktop/track2/submissions/sample_submission.csv"

test = pd.read_csv(os.path.join(DATA, "Track 1 Testing Dataset.csv"))
bundle = joblib.load("model.pkl")
models, features = bundle["models"], bundle["features"]
models_np, features_np = bundle["models_no_prev"], bundle["features_no_prev"]
print(f"main       : {len(models)} models x {len(features)} features")
print(f"specialist : {len(models_np)} models x {len(features_np)} features")

X = make_features(test)
assert not set(features) - set(X.columns), "feature mismatch"
preds = np.mean([m.predict(X[features]) for m in models], axis=0)

# Route rows that never had previous_usage to the model trained without it
gap = test["previous_usage"].isna().to_numpy()
if gap.any():
    X_np = make_features(test, drop_prev=True)
    spec = np.mean([m.predict(X_np[features_np]) for m in models_np], axis=0)
    preds[gap] = spec[gap]
print(f"routed     : {gap.sum()} rows ({gap.mean()*100:.1f}%) to the specialist")

preds = np.clip(preds, 0, None)

sub = pd.DataFrame({"id": test["id"], "prediction": preds})
sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  ({len(sub):,} rows)")

print("\n--- audit ---")
checks = {
    "no NaN": not sub["prediction"].isna().any(),
    "all finite": bool(np.isfinite(sub["prediction"]).all()),
    "no negatives": not (sub["prediction"] < 0).any(),
    "no duplicate ids": not sub["id"].duplicated().any(),
    "ids match test order": bool((sub["id"].values == test["id"].values).all()),
    "row count 3000": len(sub) == 3000,
}
if os.path.exists(SAMPLE):
    samp = pd.read_csv(SAMPLE)
    checks["columns match sample"] = list(sub.columns) == list(samp.columns)
    checks["ids match sample exactly"] = bool((sub["id"].values == samp["id"].values).all())
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
assert all(checks.values()), "AUDIT FAILED"

print("\n--- distribution ---")
tr = pd.read_csv(os.path.join(DATA, "Track 1 Training Dataset.csv"))
print(pd.DataFrame({
    "train target": tr["energy_usage"].describe(),
    "predictions": sub["prediction"].describe(),
}).round(2).to_string())
print(f"\nfirst 3 rows:\n{sub.head(3).to_string(index=False)}")
