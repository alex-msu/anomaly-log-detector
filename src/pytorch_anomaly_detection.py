# src/pytorch_anomaly_detection.py
import torch
import numpy as np
import joblib
from pathlib import Path
from .pytorch_autoencoder import Autoencoder

def detect_anomalies_hdfs(
    model_path: str | Path,
    x_path: str | Path = "data/processed/X_tfidf.joblib",
    y_path: str | Path = "data/processed/y.joblib",
    threshold_method: str = "percentile",
    percentile: float = 0.99,
    batch_size: int = 2048,
    mse_out_path: str | Path = "data/results/mse_scores.joblib",
    preds_out_path: str | Path = "data/results/anomaly_preds.joblib",
    meta_out_path: str | Path = "data/results/anomaly_meta.joblib",
):
    """
    Detectar anomalías usando un autoencoder entrenado con PyTorch
    """
    # Configurar dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device} para anomaly detection")
    
    # Cargar datos
    X = joblib.load(x_path)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype("float32", copy=False)
    y = joblib.load(y_path)
    
    # Cargar modelo
    input_dim = X.shape[1]
    model = Autoencoder(input_dim, encoding_dim=64, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Calcular MSE para todas las muestras
    dataset = torch.utils.data.TensorDataset(torch.tensor(X).float().to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    mse_scores = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0]
            reconstructions = model(inputs)
            mse = torch.mean((reconstructions - inputs) ** 2, dim=1)
            mse_scores.append(mse.cpu().numpy())  # Mover a CPU para numpy
    
    mse_scores = np.concatenate(mse_scores)
    
    # Determinar umbral
    if threshold_method == "percentile":
        threshold = np.percentile(mse_scores, percentile * 100)
    elif threshold_method == "mean_std":
        mean = np.mean(mse_scores)
        std = np.std(mse_scores)
        threshold = mean + 3 * std
    else:  # fixed
        threshold = 0.1  # valor por defecto
    
    # Predecir anomalías
    preds = (mse_scores > threshold).astype(int)
    
    # Crear directorios si no existen
    Path(mse_out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preds_out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar resultados
    joblib.dump(mse_scores, mse_out_path)
    joblib.dump(preds, preds_out_path)
    joblib.dump({
        "threshold_method": threshold_method,
        "threshold": threshold,
        "percentile": percentile if threshold_method=="percentile" else None,
        "mse_mean": float(np.mean(mse_scores)),
        "mse_std": float(np.std(mse_scores)),
        "num_anomalies": int(np.sum(preds)),
        "anomaly_ratio": float(np.mean(preds))
    }, meta_out_path)
    
    print(f"Detección de anomalías completada. Se encontraron {np.sum(preds)} anomalías")
    
    return str(mse_out_path), str(preds_out_path), str(meta_out_path)