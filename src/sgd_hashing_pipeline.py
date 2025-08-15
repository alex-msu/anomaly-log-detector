# src/sgd_hashing_pipeline.py
from pathlib import Path
import pandas as pd, numpy as np, joblib, gc, hashlib
from typing import Optional, Tuple
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, classification_report, confusion_matrix)

# -----------------------
# Helpers
# -----------------------
def _block_val_mask(block_ids: np.ndarray, *, mod=10, pick=0):
    """Split estable por bloque usando hash MD5 % mod."""
    def is_val(b):
        if not isinstance(b, str): b = str(b)
        h = int(hashlib.md5(b.encode()).hexdigest(), 16)
        return (h % mod) == pick
    return np.fromiter((is_val(b) for b in block_ids), dtype=bool, count=len(block_ids))

def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def _threshold_for_precision(y, scores, min_precision=0.20) -> float:
    P, R, T = precision_recall_curve(y, scores)
    ok = np.where(P[:-1] >= min_precision)[0]  # T tiene len-1 respecto a P/R
    if len(ok) == 0:
        # si no hay punto con esa precisión, usa el de mejor F1
        f1 = 2 * P * R / (P + R + 1e-12)
        i = int(np.nanargmax(f1))
        return float(T[i]) if i < len(T) else 0.5
    i = ok[np.argmax(R[ok])]
    return float(T[i]) if i < len(T) else 0.5

def _threshold_for_posrate(scores, target_rate=0.03) -> float:
    q = 1.0 - float(target_rate)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(scores, q))

def _best_f1_threshold(y, scores) -> Tuple[float, float, float, float]:
    P, R, T = precision_recall_curve(y, scores)
    f1 = 2 * P * R / (P + R + 1e-12)
    i = int(np.nanargmax(f1))
    thr = float(T[i]) if i < len(T) else 0.5
    return thr, float(P[i]), float(R[i]), float(f1[i])

# -----------------------
# Entrenamiento (streaming) + Calibración Platt
# -----------------------
def train_sgd_hash(
    input_csv="data/processed/hdfs_processed.csv",
    model_out="models/sgd_hash.joblib",
    vectorizer_out="models/hash_vectorizer.joblib",
    meta_out="models/sgd_hash.meta.joblib",
    calibrator_out="models/sgd_hash.cal.joblib",
    chunksize=200_000,
    n_features=2**18,
    ngram_range=(1, 2),
    val_fold=10, val_pick=0,
    random_state=42,
    calibrate=True,
):
    vec = HashingVectorizer(
        n_features=n_features, alternate_sign=False, norm="l2",
        ngram_range=ngram_range, dtype=np.float32
    )
    clf = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-5,
        max_iter=1, tol=None, learning_rate="optimal",
        random_state=random_state
    )

    first = True
    y_val_all, p_val_all = [], []
    n_seen0 = n_seen1 = 0  # para sample_weight equilibrado progresivo

    for chunk in pd.read_csv(input_csv, usecols=["message","block_id","label"], chunksize=chunksize):
        y = chunk["label"].astype(int).values
        vmask = _block_val_mask(chunk["block_id"].astype(str).values, mod=val_fold, pick=val_pick)
        X = vec.transform(chunk["message"])

        Xtr, ytr = X[~vmask], y[~vmask]
        Xva, yva = X[vmask], y[vmask]

        # Pesado "balanced" en streaming
        n_seen0 += int((ytr == 0).sum()); n_seen1 += int((ytr == 1).sum())
        total = n_seen0 + n_seen1
        w0 = total / (2 * max(n_seen0, 1)); w1 = total / (2 * max(n_seen1, 1))
        sw = np.where(ytr == 1, w1, w0).astype(np.float32)

        if first:
            clf.partial_fit(Xtr, ytr, classes=np.array([0,1]), sample_weight=sw)
            first = False
        else:
            clf.partial_fit(Xtr, ytr, sample_weight=sw)

        if Xva.shape[0] > 0:
            try:
                p = clf.predict_proba(Xva)[:, 1]
            except AttributeError:
                z = clf.decision_function(Xva); p = 1.0 / (1.0 + np.exp(-z))
            y_val_all.append(yva); p_val_all.append(p)

        del chunk, X, Xtr, Xva; gc.collect()

    yv = np.concatenate(y_val_all); pv = np.concatenate(p_val_all)

    # Calibración (Platt: y ~ logistic(logit(p)))
    cal_meta = {}
    if calibrate:
        cal = LogisticRegression(max_iter=200, solver="lbfgs")
        z = _safe_logit(pv).reshape(-1,1)
        cal.fit(z, yv)
        Path(calibrator_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cal, calibrator_out)
        cal_meta = {"calibrated": True, "calibrator_path": calibrator_out}
    else:
        cal = None
        cal_meta = {"calibrated": False}

    # Métricas de validación (sin y con calibración)
    pv_use = pv if cal is None else cal.predict_proba(_safe_logit(pv).reshape(-1,1))[:,1]
    thr_f1, p_f1, r_f1, f1 = _best_f1_threshold(yv, pv_use)
    meta = {
        "val_ap": float(average_precision_score(yv, pv_use)),
        "val_roc_auc": float(roc_auc_score(yv, pv_use)),
        "best_f1": float(f1), "best_precision": float(p_f1), "best_recall": float(r_f1),
        "threshold_f1": float(thr_f1),
        "n_features": int(n_features), "ngram_range": ngram_range,
        "val_fold": val_fold, "val_pick": val_pick,
        **cal_meta
    }

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    joblib.dump(vec, vectorizer_out)
    joblib.dump(meta, meta_out)
    return model_out, vectorizer_out, meta_out, (calibrator_out if calibrate else None)

# -----------------------
# Evaluación a nivel BLOQUE con agregadores y políticas de umbral
# -----------------------
def eval_block_level(
    input_csv="data/processed/hdfs_processed.csv",
    model_path="models/sgd_hash.joblib",
    vectorizer_path="models/hash_vectorizer.joblib",
    calibrator_path: Optional[str] = "models/sgd_hash.cal.joblib",
    threshold_mode: str = "f1",   # "f1" | "posrate" | "min_precision" | "fixed"
    target_posrate: float = 0.03,
    min_precision: float = 0.25,
    fixed_threshold: float = 0.5,
    agg: str = "noisy_or",        # "noisy_or" | "max" | "mean"
    chunksize=200_000,
    out_prefix="data/results/sgd_hash",
):
    vec = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
    cal = None
    if calibrator_path and Path(calibrator_path).exists():
        try:
            cal = joblib.load(calibrator_path)
        except Exception:
            cal = None

    # Acumuladores por bloque según agregador
    if agg == "noisy_or":
        block_acc = {}   # suma de log(1-p)
    elif agg == "max":
        block_acc = {}   # max(p)
    elif agg == "mean":
        block_acc = {}   # suma(p)
        block_cnt = {}   # cuenta
    else:
        raise ValueError(f"agg desconocido: {agg}")
    block_label = {}

    for chunk in pd.read_csv(input_csv, usecols=["message","block_id","label"], chunksize=chunksize):
        X = vec.transform(chunk["message"])
        bids = chunk["block_id"].astype(str).values
        y = chunk["label"].astype(int).values

        # Probabilidad por línea
        try:
            p = clf.predict_proba(X)[:, 1]
        except AttributeError:
            z = clf.decision_function(X); p = 1.0 / (1.0 + np.exp(-z))

        # Calibración (si existe)
        if cal is not None:
            z = _safe_logit(p).reshape(-1,1)
            p = cal.predict_proba(z)[:, 1]

        # Agregación por bloque
        if agg == "noisy_or":
            for b, pi, yi in zip(bids, p, y):
                block_acc[b] = block_acc.get(b, 0.0) + np.log1p(-float(pi))
                if b not in block_label: block_label[b] = int(yi)
        elif agg == "max":
            for b, pi, yi in zip(bids, p, y):
                cur = block_acc.get(b, 0.0)
                if pi > cur: block_acc[b] = float(pi)
                if b not in block_label: block_label[b] = int(yi)
        elif agg == "mean":
            for b, pi, yi in zip(bids, p, y):
                block_acc[b] = block_acc.get(b, 0.0) + float(pi)
                block_cnt[b] = block_cnt.get(b, 0) + 1
                if b not in block_label: block_label[b] = int(yi)

        del X, chunk; gc.collect()

    # Probabilidad a nivel bloque
    probs, y_true = [], []
    if agg == "noisy_or":
        for b, s in block_acc.items():
            p_block = 1.0 - np.exp(s)  # 1 - prod(1 - p_i)
            probs.append(p_block); y_true.append(block_label[b])
    elif agg == "max":
        for b, m in block_acc.items():
            probs.append(m); y_true.append(block_label[b])
    elif agg == "mean":
        for b, s in block_acc.items():
            p_block = s / max(block_cnt[b], 1)
            probs.append(p_block); y_true.append(block_label[b])

    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    # Selección de umbral
    if threshold_mode == "f1":
        thr, p_f1, r_f1, f1 = _best_f1_threshold(y_true, probs)
    elif threshold_mode == "posrate":
        thr = _threshold_for_posrate(probs, target_rate=target_posrate)
        p_f1 = r_f1 = f1 = np.nan
    elif threshold_mode == "min_precision":
        thr = _threshold_for_precision(y_true, probs, min_precision=min_precision)
        p_f1 = r_f1 = f1 = np.nan
    elif threshold_mode == "fixed":
        thr = float(fixed_threshold)
        p_f1 = r_f1 = f1 = np.nan
    else:
        raise ValueError(f"threshold_mode desconocido: {threshold_mode}")

    y_hat = (probs >= thr).astype(int)

    # Métricas
    cm = confusion_matrix(y_true, y_hat)
    rep = classification_report(y_true, y_hat, digits=4)
    ap = average_precision_score(y_true, probs)
    roc = roc_auc_score(y_true, probs)

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "threshold": float(thr),
            "threshold_mode": threshold_mode,
            "target_posrate": float(target_posrate),
            "min_precision": float(min_precision),
            "agg": agg,
            "cm": cm, "report": rep, "ap": float(ap), "roc": float(roc),
            "best_f1_thr": (float(thr) if threshold_mode=="f1" else None),
            "best_f1": (float(f1) if threshold_mode=="f1" else None),
            "best_precision": (float(p_f1) if threshold_mode=="f1" else None),
            "best_recall": (float(r_f1) if threshold_mode=="f1" else None),
        },
        out_prefix + ".metrics.joblib"
    )
    joblib.dump(probs, out_prefix + ".block_probs.joblib")

    print(f"Threshold ({threshold_mode}): {thr}")
    print("Confusion matrix:\n", cm)
    print(rep)
    print("AUC-PR:", ap, "ROC-AUC:", roc)
    return thr, cm, rep, ap, roc