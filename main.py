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
    
    x_path, y_path, vec_path = vectorize_hdfs(
        input_csv=out_path,
        max_features=VECTORIZE_FEATURES,
        ngram_range=(1, 1),
        chunksize=VECTORIZE_CHUNKSIZE,
        verbose=2
    )
    
    print(f"[SUCCESS] Vectorización completada. Rutas:\n - X: {x_path}\n - y: {y_path}\n - Vectorizador: {vec_path}")
    
    # Liberar memoria agresivamente
    print("Liberando memoria después de vectorización...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ... dentro de main(), reemplaza las etapas 5 y 6:
    if MODE == "sgd":
        print("\n" + "="*50)
        print("Paso 5: Entrenamiento rápido (Hashing + SGDClassifier, streaming)")
        model_path, vec_path, meta_path = train_sgd_hash(
            input_csv=out_path,
            chunksize=200_000,      # súbelo/bájalo según RAM
            n_features=2**18,       # 262k; prueba 2**19 si hay RAM
            ngram_range=(1, 2),     # uni + bi-gramas
            val_fold=10, val_pick=0
        )
        print(f"[OK] Modelo: {model_path}\nVectorizador: {vec_path}\nMeta: {meta_path}")
        print("\n" + "="*50)
        print("Paso 6: Evaluación a nivel de BLOQUE + selección de umbral por F1")
        thr, cm, rep, ap, roc = eval_block_level(
            input_csv=out_path,
            model_path=model_path,
            vectorizer_path=vec_path,
            threshold=None,         # elige por F1 a nivel bloque
            chunksize=200_000
        )
    else:
        # Tu ruta actual con autoencoder (menos recomendable con tus recursos)
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