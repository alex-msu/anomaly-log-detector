#!/usr/bin/env python3
"""
Smoke test para CI / desarrollo local.

Estrategia:
- Si existe un CSV de muestra en `tests/data/hdfs_sample.csv` (o en la ruta dada por
  la variable de entorno HDFS_SAMPLE_CSV), lo usa directamente.
- Si NO existe, descarga el ZIP oficial (Zenodo), extrae los archivos necesarios,
  preprocesa a CSV y recorta a N filas para que el test sea rápido.

Luego entrena (Hashing+SGD en streaming, con calibración) y evalúa a nivel de bloque
con parámetros "chicos" adecuados para CI.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Rutas y helpers del proyecto
from src.hdfs_download import download_file, extract_selected
from src.hdfs_preprocess import preprocess_hdfs
from src.sgd_hashing_pipeline import train_sgd_hash, eval_block_level

# ---------------------------
# Config por variables de entorno (con defaults seguros para CI)
# ---------------------------
def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

def _get_env_tuple2_int(name: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = os.environ.get(name, f"{default[0]},{default[1]}")
    try:
        a, b = raw.split(",")
        return int(a), int(b)
    except Exception:
        return default

# Dataset remoto (fallback)
HDFS_ZIP_URL = os.environ.get(
    "HDFS_ZIP_URL",
    "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
)
ZIP_PATH = os.environ.get("HDFS_ZIP_PATH", "data/HDFS_v1.zip")
EXTRACT_DIR = os.environ.get("HDFS_EXTRACT_DIR", "data/raw/hdfs")
FILES_TO_EXTRACT = os.environ.get(
    "HDFS_FILES_TO_EXTRACT",
    "HDFS.log,preprocessed/anomaly_label.csv"
).split(",")

# Sample local opcional
SAMPLE_CSV = os.environ.get("HDFS_SAMPLE_CSV", "tests/data/hdfs_sample.csv")

# CSV final para este smoke
PROCESSED_CSV = os.environ.get("HDFS_PROCESSED_CSV", "data/processed/hdfs_processed.csv")
CI_CSV = os.environ.get("HDFS_CI_CSV", "data/processed/hdfs_processed_ci.csv")
CI_HEAD_ROWS = _get_env_int("HDFS_CI_HEAD_ROWS", 200_000)

# Parámetros de entrenamiento/evaluación (chicos para CI)
HDFS_CHUNKSIZE = _get_env_int("HDFS_CHUNKSIZE", 20_000)
HDFS_NFEATURES = _get_env_int("HDFS_NFEATURES", 131_072)   # 2**17 para CI
HDFS_NGRAM = _get_env_tuple2_int("HDFS_NGRAM", (1, 2))
HDFS_THRESHOLD_MODE = os.environ.get("HDFS_THRESHOLD_MODE", "posrate")  # posrate|min_precision|f1|fixed
HDFS_TARGET_POSRATE = _get_env_float("HDFS_TARGET_POSRATE", 0.03)
HDFS_MIN_PREC = _get_env_float("HDFS_MIN_PREC", 0.25)
HDFS_AGG = os.environ.get("HDFS_AGG", "max")                # max|mean|noisy_or

# Limitar hilos BLAS para runners
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def ensure_dirs():
    for p in ["data/raw", "data/processed", "models", "docs/assets", "tests/data", "data/results"]:
        Path(p).mkdir(parents=True, exist_ok=True)

def prepare_dataset() -> str:
    """
    Devuelve la ruta al CSV que usará el smoke test:
    - Si existe SAMPLE_CSV → lo usa tal cual.
    - Si no existe → descarga, extrae, preprocesa y recorta (guardando CI_CSV).
    """
    ensure_dirs()
    sample = Path(SAMPLE_CSV)
    if sample.exists():
        print(f"[SMOKE] Usando sample local: {sample}")
        return str(sample)

    # Fallback: dataset real (descargar → extraer → preprocesar → recortar)
    print(f"[SMOKE] Sample no encontrado. Descargando ZIP a: {ZIP_PATH}")
    zip_path = download_file(HDFS_ZIP_URL, ZIP_PATH, skip_if_exists=True)
    print(f"[SMOKE] Extrayendo archivos necesarios a: {EXTRACT_DIR}")
    extract_selected(zip_path, EXTRACT_DIR, FILES_TO_EXTRACT, flatten=True, overwrite=False)

    log_path = os.path.join(EXTRACT_DIR, "HDFS.log")
    labels_path = os.path.join(EXTRACT_DIR, "anomaly_label.csv")
    print(f"[SMOKE] Preprocesando → {PROCESSED_CSV}")
    processed_csv = preprocess_hdfs(
        log_path=log_path,
        labels_path=labels_path,
        out_path=PROCESSED_CSV
    )

    print(f"[SMOKE] Recortando a {CI_HEAD_ROWS} filas para CI → {CI_CSV}")
    df = pd.read_csv(processed_csv)
    df.head(CI_HEAD_ROWS).to_csv(CI_CSV, index=False)
    return CI_CSV

def main():
    csv_path = prepare_dataset()
    print(f"[SMOKE] CSV de trabajo: {csv_path}")

    print("[SMOKE] Entrenando Hashing+SGD (streaming, calibración)…")
    model_path, vec_path, meta_path, cal_path = train_sgd_hash(
        input_csv=csv_path,
        chunksize=HDFS_CHUNKSIZE,
        n_features=HDFS_NFEATURES,
        ngram_range=HDFS_NGRAM,
        val_fold=10, val_pick=0,
        calibrate=True
    )
    print(f"[SMOKE] Modelo: {model_path}\nVectorizador: {vec_path}\nMeta: {meta_path}\nCalibrador: {cal_path}")

    print("[SMOKE] Evaluando a nivel de bloque…")
    thr, cm, rep, ap, roc = eval_block_level(
        input_csv=csv_path,
        model_path=model_path,
        vectorizer_path=vec_path,
        calibrator_path=cal_path,
        threshold_mode=HDFS_THRESHOLD_MODE,
        target_posrate=HDFS_TARGET_POSRATE,
        min_precision=HDFS_MIN_PREC,
        agg=HDFS_AGG,
        chunksize=HDFS_CHUNKSIZE,
        out_prefix="data/results/ci_check"
    )

    # Resumen y chequeo básico
    print("\n[SMOKE] === Resumen ===")
    print("Threshold:", thr)
    print("AUC-PR:", ap, "ROC-AUC:", roc)
    print(rep)

    # Asegura que generó algo sensato (siempre > 0 con dataset válido)
    assert ap >= 0.01, "AUC-PR inesperadamente bajo en el smoke test."
    print("[SMOKE] OK ✅")

if __name__ == "__main__":
    main()