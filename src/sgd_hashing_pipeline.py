# src/sgd_hashing_pipeline.py
from pathlib import Path
import pandas as pd, numpy as np, joblib, gc, hashlib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, classification_report, confusion_matrix)

def _block_val_mask(block_ids: np.ndarray, *, mod=10, pick=0):
    # Split estable por bloque (sin sets en RAM)
    def is_val(b):
        if not isinstance(b, str): b = str(b)
        h = int(hashlib.md5(b.encode()).hexdigest(), 16)
        return (h % mod) == pick
    return np.fromiter((is_val(b) for b in block_ids), dtype=bool, count=len(block_ids))

def train_sgd_hash(
    input_csv="data/processed/hdfs_processed.csv",
    model_out="models/sgd_hash.joblib",
    vectorizer_out="models/hash_vectorizer.joblib",
    meta_out="models/sgd_hash.meta.joblib",
    chunksize=200_000,
    n_features=2**18,
    ngram_range=(1, 2),
    val_fold=10, val_pick=0,
    random_state=42,
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
    n_seen0 = n_seen1 = 0  # para class_weight='balanced' vía sample_weight

    for chunk in pd.read_csv(input_csv, usecols=["message","block_id","label"], chunksize=chunksize):
        y = chunk["label"].astype(int).values
        vmask = _block_val_mask(chunk["block_id"].astype(str).values, mod=val_fold, pick=val_pick)
        X = vec.transform(chunk["message"])

        Xtr, ytr = X[~vmask], y[~vmask]
        Xva, yva = X[vmask], y[vmask]

        # Pesado "balanced" sin cargar todo el dataset
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
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.nanargmax(f1))
    best_thr = float(th[i] if i < len(th) else 0.5)

    meta = {
        "val_ap": float(average_precision_score(yv, pv)),
        "val_roc_auc": float(roc_auc_score(yv, pv)),
        "best_f1": float(f1[i]), "best_precision": float(pr[i]), "best_recall": float(rc[i]),
        "threshold": best_thr,
        "n_features": int(n_features), "ngram_range": ngram_range,
        "val_fold": val_fold, "val_pick": val_pick
    }

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    joblib.dump(vec, vectorizer_out)
    joblib.dump(meta, meta_out)
    return model_out, vectorizer_out, meta_out

def eval_block_level(
    input_csv="data/processed/hdfs_processed.csv",
    model_path="models/sgd_hash.joblib",
    vectorizer_path="models/hash_vectorizer.joblib",
    threshold=None,
    chunksize=200_000,
    out_prefix="data/results/sgd_hash",
):
    vec = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)

    block_log1m = {}   # acumulamos sum(log(1-p)) por bloque
    block_label = {}

    for chunk in pd.read_csv(input_csv, usecols=["message","block_id","label"], chunksize=chunksize):
        X = vec.transform(chunk["message"])
        try:
            p = clf.predict_proba(X)[:, 1]
        except AttributeError:
            z = clf.decision_function(X); p = 1.0 / (1.0 + np.exp(-z))

        bids = chunk["block_id"].astype(str).values
        y = chunk["label"].astype(int).values

        for b, pi, yi in zip(bids, p, y):
            # Prob. de que el bloque sea anómalo = 1 - prod(1 - p_i)
            log1m = np.log1p(-float(pi))  # log(1 - p)
            block_log1m[b] = block_log1m.get(b, 0.0) + log1m
            if b not in block_label: block_label[b] = int(yi)

        del X, chunk; gc.collect()

    # Convertir a arrays
    probs, y_true = [], []
    for b, s in block_log1m.items():
        p_block = 1.0 - np.exp(s)  # 1 - prod(1 - p_i)
        probs.append(p_block)
        y_true.append(block_label[b])
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    # Si no pasaron threshold, elegimos por F1 a nivel bloque
    from sklearn.metrics import precision_recall_curve
    if threshold is None:
        pr, rc, th = precision_recall_curve(y_true, probs)
        f1 = 2 * pr * rc / (pr + rc + 1e-12)
        j = int(np.nanargmax(f1))
        threshold = float(th[j] if j < len(th) else 0.5)

    y_hat = (probs >= threshold).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
    cm = confusion_matrix(y_true, y_hat)
    rep = classification_report(y_true, y_hat, digits=4)
    ap = average_precision_score(y_true, probs)
    roc = roc_auc_score(y_true, probs)

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"threshold": threshold, "cm": cm, "report": rep, "ap": ap, "roc": roc},
                out_prefix + ".metrics.joblib")
    joblib.dump(probs, out_prefix + ".block_probs.joblib")

    print("Threshold:", threshold)
    print("Confusion matrix:\n", cm)
    print(rep)
    print("AUC-PR:", ap, "ROC-AUC:", roc)
    return threshold, cm, rep, ap, roc
