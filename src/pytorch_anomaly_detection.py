import torch
import numpy as np
import joblib
from pathlib import Path
from .pytorch_autoencoder import Autoencoder, SparseTensorDataset
from torch.utils.data import DataLoader
import gc

def detect_anomalies_hdfs(
    model_path: str | Path,
    x_path: str | Path = "data/processed/X_tfidf_sparse.joblib",
    y_path: str | Path = "data/processed/y.joblib",
    threshold_method: str = "percentile",
    percentile: float = 0.99,
    batch_size: int = 8192,
    mse_out_path: str | Path = "data/results/mse_scores.joblib",
    preds_out_path: str | Path = "data/results/anomaly_preds.joblib",
    meta_out_path: str | Path = "data/results/anomaly_meta.joblib",
):
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device} para detección de anomalías")
    
    # Cargar datos dispersos
    X_sparse = joblib.load(x_path)
    y = joblib.load(y_path)
    
    # Cargar modelo
    input_dim = X_sparse.shape[1]
    model = Autoencoder(input_dim, encoding_dim=24, hidden_dim=48).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dataset para carga incremental
    dataset = SparseTensorDataset(X_sparse, device)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Calcular MSE en bloques
    mse_scores = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            reconstructions = model(inputs)
            mse = torch.mean((reconstructions - inputs) ** 2, dim=1).cpu().numpy()
            mse_scores.append(mse)
    
    mse_scores = np.concatenate(mse_scores)
    
    # Determinar umbral
    if threshold_method == "percentile":
        threshold = np.percentile(mse_scores, percentile * 100)
    elif threshold_method == "mean_std":
        mean = np.mean(mse_scores)
        std = np.std(mse_scores)
        threshold = mean + 3 * std
    else:  # fixed
        threshold = 0.1
    
    # Predecir anomalías
    preds = (mse_scores > threshold).astype(int)
    
    # Guardar resultados
    Path(mse_out_path).parent.mkdir(parents=True, exist_ok=True)
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
    
    print(f"Detección completada. Anomalías encontradas: {np.sum(preds)}")
    
    return str(mse_out_path), str(preds_out_path), str(meta_out_path)