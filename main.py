import os
import sys
from pathlib import Path
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
    BATCH_SIZE_TRAIN = 512
    BATCH_SIZE_DETECT = 4096
    # Cambiar rutas si es necesario para Colab
    BASE_DIR = "/content/anomaly-log-detector"
    if os.path.exists(BASE_DIR):
        os.chdir(BASE_DIR)
else:
    BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_DETECT = 2048

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

    # 2. EXTRACCIÓN
    print("\n" + "="*50)
    print("Paso 2: Extracción de archivos seleccionados")
    extracted = extract_selected(ZIP_PATH, EXTRACT_DIR, FILES_TO_EXTRACT, flatten=True, overwrite=False)
    print(f"[SUCCESS] Archivos extraídos: {extracted}")

    # 3. PREPROCESAMIENTO
    print("\n" + "="*50)
    print("Paso 3: Preprocesamiento de logs HDFS")
    out_path = preprocess_hdfs()
    print(f"[SUCCESS] Archivo preprocesado guardado en: {out_path}")

    # 4. VECTORIZACIÓN
    print("\n" + "="*50)
    print("Paso 4: Vectorización con TF-IDF")
    x_path, y_path, vec_path = vectorize_hdfs(return_data=False)
    print(f"[SUCCESS] Vectorización completada. Rutas:\n - X: {x_path}\n - y: {y_path}\n - Vectorizador: {vec_path}")

    # 5. ENTRENAR AUTOENCODER
    print("\n" + "="*50)
    print("Paso 5: Entrenamiento del autoencoder")
    history, model_path = train_autoencoder_hdfs(
        x_path=x_path,
        y_path=y_path,
        batch_size=BATCH_SIZE_TRAIN
    )
    print(f"Entrenamiento completo. Modelo guardado en: {model_path}")

    # 6. DETECTAR ANOMALÍAS
    print("\n" + "="*50)
    print("Paso 6: Detección de anomalías")
    mse_path, preds_path, meta_path = detect_anomalies_hdfs(
        model_path=model_path,
        x_path=x_path,
        y_path=y_path,
        batch_size=BATCH_SIZE_DETECT,
        threshold_method="percentile",
        percentile=0.99
    )
    print(f"Detección completa.\n - MSE: {mse_path}\n - Predicciones: {preds_path}\n - Metadatos: {meta_path}")

    print("\n" + "="*50)
    print("¡Pipeline completado con éxito!")

if __name__ == "__main__":
    main()