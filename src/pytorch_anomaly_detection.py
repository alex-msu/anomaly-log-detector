import torch
import numpy as np
import joblib
from pathlib import Path
from .pytorch_autoencoder import Autoencoder, SparseTensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, average_precision_score, roc_auc_score
import pandas as pd

def _best_f1_threshold(scores, y):
    pr, rc, th = precision_recall_curve(y, scores)
    f1 = 2*pr*rc/(pr+rc+1e-12)
    i = int(np.nanargmax(f1))
    thr = float(th[i] if i < len(th) else 0.5)
    return thr, float(pr[i]), float(rc[i]), float(f1[i])

def detect_anomalies_hdfs(
    model_path: str | Path,
    x_path: str | Path = "data/processed/X_tfidf_sparse.joblib",
    y_path: str | Path = "data/processed/y.joblib",
    threshold_method: str = "percentile",      # "percentile" | "mean_std" | "fixed" | "val_f1"
    percentile: float = 0.99,
    batch_size: int = 8192,
    mse_out_path: str | Path = "data/results/mse_scores.joblib",
    preds_out_path: str | Path = "data/results/anomaly_preds.joblib",
    meta_out_path: str | Path = "data/results/anomaly_meta.joblib",
    # evaluación por bloque (opcional)
    aggregate_by_block: bool = False,
    processed_csv_path: str | Path = "data/processed/hdfs_processed.csv",
    agg: str = "max",   # "max" o "mean"
    num_workers: int = 2,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device} para detección de anomalías")

    X_sparse = joblib.load(x_path)

    # Cargar modelo (misma arquitectura que en train)
    input_dim = X_sparse.shape[1]
    model = Autoencoder(input_dim, encoding_dim=24, hidden_dim=48).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # DataLoader sin shuffle para preservar orden  :contentReference[oaicite:5]{index=5}
    dataset = SparseTensorDataset(X_sparse)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type=='cuda'),
                        persistent_workers=(num_workers>0))

    mse_scores = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            recon = model(inputs)
            mse = torch.mean((recon - inputs)**2, dim=1).cpu().numpy()
            mse_scores.append(mse)
    mse_scores = np.concatenate(mse_scores)
    Path(mse_out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mse_scores, mse_out_path)

    # --- Selección de umbral ---
    threshold = None
    extra_meta = {}
    if threshold_method == "percentile":
        threshold = np.percentile(mse_scores, percentile * 100)   # :contentReference[oaicite:6]{index=6}
    elif threshold_method == "mean_std":
        threshold = float(np.mean(mse_scores) + 3*np.std(mse_scores))
    elif threshold_method == "fixed":
        threshold = 0.1
    elif threshold_method == "val_f1":
        y = joblib.load(y_path).astype(int)
        threshold, p, r, f1 = _best_f1_threshold(mse_scores, y)
        extra_meta.update({"val_best_precision": p, "val_best_recall": r, "val_best_f1": f1})
    else:
        raise ValueError(f"threshold_method desconocido: {threshold_method}")

    preds = (mse_scores > threshold).astype(int)
    joblib.dump(preds, preds_out_path)

    # --- Métricas a nivel línea (si hay y) ---
    metrics = {}
    if Path(y_path).exists():
        y = joblib.load(y_path).astype(int)
        try:
            ap = average_precision_score(y, mse_scores)
            roc = roc_auc_score(y, mse_scores)
            rep = classification_report(y, preds, digits=4)
            cm = confusion_matrix(y, preds)
            metrics.update({"line_ap": float(ap), "line_roc": float(roc), "line_report": rep, "line_cm": cm})
        except Exception:
            pass

    # --- (Opcional) Agregar por bloque y evaluar a nivel de bloque ---
    if aggregate_by_block:
        print("Agregando por block_id…")
        block_scores, block_labels = {}, {}
        off = 0
        for chunk in pd.read_csv(processed_csv_path, usecols=["block_id","label"], chunksize=200_000):
            n = len(chunk)
            ms = mse_scores[off:off+n]; off += n
            bids = chunk["block_id"].astype(str).values
            labs = chunk["label"].astype(int).values
            for b, s, yb in zip(bids, ms, labs):
                if b not in block_scores:
                    block_scores[b] = s
                else:
                    if agg == "max":  block_scores[b] = max(block_scores[b], s)
                    else:             block_scores[b] += s  # suma para luego promediar
                block_labels[b] = yb
        scores = np.array([block_scores[b] if agg=="max" else block_scores[b]/1.0  for b in block_scores.keys()])
        yb = np.array([block_labels[b] for b in block_scores.keys()], dtype=int)

        # si el umbral vino de val_f1 a nivel línea, recalcula a nivel bloque:
        if threshold_method == "val_f1":
            thr_b, p, r, f1 = _best_f1_threshold(scores, yb)
            threshold = thr_b
            extra_meta.update({"block_best_precision": p, "block_best_recall": r, "block_best_f1": f1})

        preds_b = (scores > threshold).astype(int)
        ap_b = average_precision_score(yb, scores)
        roc_b = roc_auc_score(yb, scores)
        rep_b = classification_report(yb, preds_b, digits=4)
        cm_b  = confusion_matrix(yb, preds_b)
        metrics.update({
            "block_ap": float(ap_b), "block_roc": float(roc_b),
            "block_report": rep_b, "block_cm": cm_b
        })

    # Guardar metadatos
    joblib.dump({
        "threshold_method": threshold_method,
        "threshold": float(threshold),
        "percentile": percentile,
        "mse_mean": float(np.mean(mse_scores)),
        "mse_std": float(np.std(mse_scores)),
        "num_anomalies": int(preds.sum()),
        "anomaly_ratio": float(preds.mean()),
        **extra_meta, **metrics,
    }, meta_out_path)

    print(f"Detección completada. Umbral={threshold:.6g}  Anomalías={int(preds.sum())}")
    if "block_report" in metrics:
        print("\n[Métricas BLOQUE]\n", metrics["block_report"])
    elif "line_report" in metrics:
        print("\n[Métricas LÍNEA]\n", metrics["line_report"])

    return str(mse_out_path), str(preds_out_path), str(meta_out_path)