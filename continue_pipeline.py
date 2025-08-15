import os
import joblib
import torch
import numpy as np
from pathlib import Path
from src.pytorch_anomaly_detection import detect_anomalies_hdfs
from src.hdfs_vectorize import vectorize_hdfs  # Solo si necesitas regenerar X,y

# Configurar rutas (ajusta según tu estructura)
MODEL_PATH = "models/autoencoder_hdfs.pt"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
PROCESSED_CSV = "data/processed/hdfs_processed.csv"
Y_PATH = "data/processed/y.joblib"

def main():
    # Crear estructura de directorios
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)
    
    # Paso 1: Verificar o regenerar datos vectorizados
    if not os.path.exists("data/processed/X_tfidf_sparse.joblib"):
        print("Generando datos vectorizados...")
        _, x_path, y_path = vectorize_hdfs(
            input_csv=PROCESSED_CSV,
            vectorizer_out_path=VECTORIZER_PATH,
            return_data=False
        )
        # Si tenemos y.joblib subido, usarlo
        if os.path.exists(Y_PATH):
            joblib.dump(joblib.load(Y_PATH), y_path)
    else:
        x_path = "data/processed/X_tfidf_sparse.joblib"
        y_path = "data/processed/y.joblib"
    
    # Paso 2: Detección de anomalías
    print("\n" + "="*50)
    print("Ejecutando detección de anomalías con modelo pre-entrenado")
    
    mse_path, preds_path, meta_path = detect_anomalies_hdfs(
        model_path=MODEL_PATH,
        x_path=x_path,
        y_path=y_path,
        batch_size=8192,
        threshold_method="percentile",
        percentile=0.99
    )
    
    print(f"Detección completada.\n - MSE: {mse_path}\n - Predicciones: {preds_path}\n - Metadatos: {meta_path}")
    
    # Paso 3: Visualización de resultados
    print("\n" + "="*50)
    print("Visualizando resultados...")
    
    try:
        import matplotlib.pyplot as plt
        mse_scores = joblib.load(mse_path)
        meta = joblib.load(meta_path)
        
        plt.figure(figsize=(10, 6))
        plt.hist(mse_scores, bins=50, alpha=0.7, color='blue')
        plt.axvline(meta['threshold'], color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribución de Errores de Reconstrucción (MSE)')
        plt.xlabel('MSE')
        plt.ylabel('Frecuencia')
        plt.legend(['Umbral de anomalía', 'Distribución MSE'])
        plt.grid(True)
        plt.show()
        
        print(f"Anomalías detectadas: {meta['num_anomalies']} ({meta['anomaly_ratio']*100:.2f}%)")
        
    except Exception as e:
        print(f"Error en visualización: {e}")

if __name__ == "__main__":
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Usando dispositivo: {device} ===")
    
    main()