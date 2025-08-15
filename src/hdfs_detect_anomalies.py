# src/hdfs_detect_anomalies.py
from pathlib import Path
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from typing import Literal, Tuple

__all__ = ["detect_anomalies_hdfs"]

def _predict_in_batches(model, X, batch_size: int = 1024) -> np.ndarray:
    """Run model(X) in batches to avoid OOM; returns dense array."""
    n = X.shape[0]
    preds = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X[start:end]
        preds.append(model.predict(batch, verbose=0))
    return np.vstack(preds)

def detect_anomalies_hdfs(
    model_path: str | Path = "models/autoencoder_hdfs.h5",
    x_path: str | Path = "data/processed/X_tfidf.joblib",
    y_path: str | Path = "data/processed/y.joblib",
    out_dir: str | Path = "outputs",
    *,
    threshold_method: Literal["percentile", "mean_std", "fixed"] = "percentile",
    percentile: float = 0.99,
    k_std: float = 3.0,
    fixed_threshold: float | None = None,
    batch_size: int = 1024,
    return_arrays: bool = False,
) -> Tuple[str, str, str]:
    """
    Load the trained autoencoder and TF-IDF data, compute reconstruction MSE,
    choose a threshold from *normal* samples (y==0), and save predictions.

    Returns
    -------
    (mse_path, preds_path, meta_path): tuple of str paths
    """
    model_path = Path(model_path)
    x_path = Path(x_path)
    y_path = Path(y_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    autoencoder = load_model(model_path)
    X = joblib.load(x_path)
    if hasattr(X, "toarray"):  # handle scipy sparse
        X = X.toarray()
    X = X.astype(np.float32, copy=False)
    y = joblib.load(y_path)

    # Reconstructions (batched)
    recon = _predict_in_batches(autoencoder, X, batch_size=batch_size)
    mse = np.mean((X - recon) ** 2, axis=1)

    # Determine threshold on NORMALS
    normal_mse = mse[y == 0]
    if threshold_method == "percentile":
        threshold = float(np.quantile(normal_mse, percentile))
    elif threshold_method == "mean_std":
        threshold = float(normal_mse.mean() + k_std * normal_mse.std(ddof=0))
    elif threshold_method == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed_threshold must be provided when threshold_method='fixed'")
        threshold = float(fixed_threshold)
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")

    # Predictions
    preds = (mse > threshold).astype(int)

    # Save outputs
    mse_path = out_dir / "mse_scores.npy"
    preds_path = out_dir / "anomaly_predictions.npy"
    meta_path = out_dir / "detect_meta.json"

    np.save(mse_path, mse)
    np.save(preds_path, preds)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": str(model_path),
                "x_path": str(x_path),
                "y_path": str(y_path),
                "threshold_method": threshold_method,
                "percentile": percentile,
                "k_std": k_std,
                "fixed_threshold": fixed_threshold,
                "computed_threshold": threshold,
                "batch_size": batch_size,
                "counts": {
                    "total": int(len(mse)),
                    "normals_in_y": int((y == 0).sum()),
                    "anomalies_predicted": int(preds.sum()),
                },
            },
            f,
            indent=2,
        )

    if return_arrays:
        # You can return arrays for immediate analysis if you want
        return mse, preds, {"threshold": threshold}  # type: ignore[return-value]

    return str(mse_path), str(preds_path), str(meta_path)