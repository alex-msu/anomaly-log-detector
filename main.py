import os
import sys
from pathlib import Path
import gc
import torch

from src.sgd_hashing_pipeline import train_sgd_hash, eval_block_level
MODE = os.environ.get("HDFS_MODE", "sgd")  # "sgd" | "ae"

from src.hdfs_download import download_file, extract_selected
from src.hdfs_preprocess import preprocess_hdfs
from src.hdfs_vectorize import vectorize_hdfs
from src.pytorch_autoencoder import train_autoencoder_hdfs
from src.pytorch_anomaly_detection import detect_anomalies_hdfs

# Configuración central
URL = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
ZIP_PATH = os.path.join("data", "HDFS_v1.zip")
EXTRACT_DIR = os.path.join("data", "raw", "hdfs")
FILES_TO_EXTRACT = ["HDFS.log", "preprocessed/anomaly_label.csv"]

# Configuración inicial eval_block_level
HDFS_CHUNKSIZE = int(os.environ.get("HDFS_CHUNKSIZE", "200000"))
HDFS_NFEATURES = int(os.environ.get("HDFS_NFEATURES", str(2**18)))
HDFS_NGRAM = tuple(map(int, os.environ.get("HDFS_NGRAM", "1,2").split(",")))
HDFS_AGG = os.environ.get("HDFS_AGG", "max")  # "noisy_or" | "max" | "mean"
HDFS_THRESHOLD_MODE = os.environ.get("HDFS_THRESHOLD_MODE", "posrate")  # "f1" | "posrate" | "min_precision" | "fixed"
HDFS_TARGET_POSRATE = float(os.environ.get("HDFS_TARGET_POSRATE", "0.03"))
HDFS_MIN_PREC = float(os.environ.get("HDFS_MIN_PREC", "0.25"))

# Configuración para Colab
if 'COLAB_GPU' in os.environ:
    print("Entorno de Google Colab detectado. Ajustando configuraciones...")
    BATCH_SIZE_TRAIN = 2048
    BATCH_SIZE_DETECT = 8192
    VECTORIZE_FEATURES = 2000
    VECTORIZE_CHUNKSIZE = 50000
    EPOCHS = 10
else:
    BATCH_SIZE_TRAIN = 1024
    BATCH_SIZE_DETECT = 4096
    VECTORIZE_FEATURES = 3000
    VECTORIZE_CHUNKSIZE = 20000
    EPOCHS = 15

def main():
    # Crear estructura de directorios
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)
    
    # 1. DESCARGA DE DATASET
    print("\n" + "="*50)
    print("Paso 1: Descarga del dataset HDFS")
    download_file(URL, ZIP_PATH, skip_if_exists=True)
    
    # Liberar memoria
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. EXTRACCIÓN
    print("\n" + "="*50)
    print("Paso 2: Extracción de archivos seleccionados")
    extracted = extract_selected(ZIP_PATH, EXTRACT_DIR, FILES_TO_EXTRACT, flatten=True, overwrite=False)
    print(f"[SUCCESS] Archivos extraídos: {extracted}")
    
    # Liberar memoria
    gc.collect()

    # 3. PREPROCESAMIENTO
    print("\n" + "="*50)
    print("Paso 3: Preprocesamiento de logs HDFS")
    out_path = preprocess_hdfs()
    print(f"[SUCCESS] Archivo preprocesado guardado en: {out_path}")
    
    # Liberar memoria
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. VECTORIZACIÓN OPTIMIZADA
    print("\n" + "="*50)
    print("Paso 4: Vectorización optimizada con TF-IDF")
    print(f"Configuración: max_features={VECTORIZE_FEATURES}, chunksize={VECTORIZE_CHUNKSIZE}")

    if MODE != "sgd":
        x_path, y_path, vec_path = vectorize_hdfs(
            input_csv=out_path,
            max_features=VECTORIZE_FEATURES,
            ngram_range=(1, 1),
            chunksize=VECTORIZE_CHUNKSIZE,
            verbose=2
        )
        print(f"[SUCCESS] Vectorización completada. Rutas:\n - X: {x_path}\n - y: {y_path}\n - Vectorizador: {vec_path}")
    else:
        print("Modo 'sgd' detectado: se omite TF-IDF (usaremos HashingVectorizer en streaming).")
        x_path = y_path = vec_path = None

    # Liberar memoria agresivamente
    print("Liberando memoria después de vectorización...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if MODE == "sgd":
        print("\n" + "="*50)
        print("Paso 5: Entrenamiento rápido (Hashing + SGDClassifier, streaming)")
        model_path, hash_vec_path, meta_path, cal_path = train_sgd_hash(
            input_csv=out_path,
            chunksize=HDFS_CHUNKSIZE,
            n_features=HDFS_NFEATURES,
            ngram_range=HDFS_NGRAM,
            val_fold=10, val_pick=0,
            calibrate=True
        )
        print(f"[OK] Modelo: {model_path}\nVectorizador: {hash_vec_path}\nMeta: {meta_path}\nCalibrador: {cal_path}")

        print("\n" + "="*50)
        print("Paso 6: Evaluación a nivel de BLOQUE (umbral según política)")
        thr, cm, rep, ap, roc = eval_block_level(
            input_csv=out_path,
            model_path=model_path,
            vectorizer_path=hash_vec_path,
            calibrator_path=cal_path,
            threshold_mode=HDFS_THRESHOLD_MODE,
            target_posrate=HDFS_TARGET_POSRATE,
            min_precision=HDFS_MIN_PREC,
            agg=HDFS_AGG,
            chunksize=HDFS_CHUNKSIZE,
            out_prefix="data/results/sgd_hash"
        )
    else:
        # Autoencoder (no recomendable)
        history, model_path = train_autoencoder_hdfs(
            x_path=x_path, y_path=y_path,
            batch_size=BATCH_SIZE_TRAIN, epochs=EPOCHS,
            encoding_dim=24, hidden_dim=48
        )  # :contentReference[oaicite:2]{index=2}
        mse_path, preds_path, meta_path = detect_anomalies_hdfs(
            model_path=model_path,
            x_path=x_path,
            y_path=y_path,
            batch_size=BATCH_SIZE_DETECT,
            threshold_method="val_f1", # validación por etiquetas
            aggregate_by_block=True,
            processed_csv_path=out_path,
        )  # :contentReference[oaicite:3]{index=3}

if __name__ == "__main__":
    # Configuración automática de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Usando dispositivo: {device} ===")
    
    # Monitoreo inicial de memoria
    if device.type == 'cuda':
        print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    main()
    
    # Liberación final de memoria
    print("Limpiando memoria...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()