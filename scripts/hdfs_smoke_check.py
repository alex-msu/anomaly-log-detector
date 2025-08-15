from pathlib import Path
import json, joblib, numpy as np, pandas as pd

required = [
    "data/raw/hdfs/HDFS.log",
    "data/raw/hdfs/anomaly_label.csv",
    "data/processed/hdfs_processed.csv",
    "data/processed/X_tfidf.joblib",
    "data/processed/y.joblib",
    "models/autoencoder_hdfs.h5",
    "outputs/mse_scores.npy",
    "outputs/anomaly_predictions.npy",
    "outputs/detect_meta.json",
]
missing = [p for p in required if not Path(p).exists()]
assert not missing, f"Missing: {missing}"

df = pd.read_csv("data/processed/hdfs_processed.csv")
assert len(df) > 0 and {"message","label"}.issubset(df.columns)

X = joblib.load("data/processed/X_tfidf.joblib")
y = joblib.load("data/processed/y.joblib")
if hasattr(X, "shape"):
    assert X.shape[0] == len(y), "X/y length mismatch"

mse = np.load("outputs/mse_scores.npy")
pred = np.load("outputs/anomaly_predictions.npy")
assert len(mse) == len(pred) == len(y)

with open("outputs/detect_meta.json","r") as f:
    meta = json.load(f)
assert "computed_threshold" in meta
print("✅ Todo está correcto!")
