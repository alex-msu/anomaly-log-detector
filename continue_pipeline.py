import os
import joblib
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import gc

# Configurar rutas (ajusta según tu estructura)
MODEL_PATH = "models/autoencoder_hdfs.pt"
X_PATH = "data/processed/X_tfidf_sparse.joblib"
Y_PATH = "data/processed/y.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

# Crear estructura de directorios
Path("data/results").mkdir(parents=True, exist_ok=True)

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=== Usando dispositivo: {device} ===")

# Paso 1: Verificar archivos
print("\nVerificando archivos necesarios...")
required_files = [MODEL_PATH, X_PATH, Y_PATH]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Archivo requerido no encontrado: {file}")
    print(f" - {file} encontrado")

# Paso 2: Cargar datos vectorizados
print("\nCargando datos vectorizados...")
X = joblib.load(X_PATH)
y = joblib.load(Y_PATH)
print(f"Datos cargados: X.shape = {X.shape}, y.shape = {y.shape}")

# Paso 3: Cargar modelo
print("\nCargando modelo pre-entrenado...")
vectorizer = joblib.load(VECTORIZER_PATH)
input_dim = X.shape[1]
print(f"Dimensionalidad de entrada: {input_dim}")

# Definir arquitectura del modelo
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, encoding_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Crear e inicializar modelo
model = Autoencoder(input_dim, encoding_dim=24, hidden_dim=48).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Modelo cargado correctamente")

# Paso 4: Detección de anomalías
print("\n" + "="*50)
print("Ejecutando detección de anomalías")

def detect_anomalies(model, X, batch_size=8192):
    """Calcula MSE para todo el dataset en bloques"""
    mse_scores = []
    num_batches = (X.shape[0] + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X.shape[0])
        
        # Convertir batch a denso
        if hasattr(X, "toarray"):
            batch = X[start_idx:end_idx].toarray().astype("float32")
        else:
            batch = X[start_idx:end_idx].astype("float32")
        
        # Convertir a tensor
        inputs = torch.tensor(batch).float().to(device)
        
        # Calcular reconstrucción
        with torch.no_grad():
            reconstructions = model(inputs)
            mse = torch.mean((reconstructions - inputs) ** 2, dim=1).cpu().numpy()
        
        mse_scores.append(mse)
        
        # Liberar memoria
        del batch, inputs, reconstructions
        if i % 10 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return np.concatenate(mse_scores)

# Calcular MSE
mse_scores = detect_anomalies(model, X)

# Determinar umbral (percentil 99%)
threshold = np.percentile(mse_scores, 99)
print(f"Umbral de anomalía (percentil 99%): {threshold:.6f}")

# Predecir anomalías
preds = (mse_scores > threshold).astype(int)
num_anomalies = np.sum(preds)
anomaly_ratio = num_anomalies / len(preds)
print(f"Anomalías detectadas: {num_anomalies} ({anomaly_ratio*100:.2f}%)")

# Paso 5: Guardar resultados
print("\nGuardando resultados...")
mse_out_path = "data/results/mse_scores.joblib"
preds_out_path = "data/results/anomaly_preds.joblib"
meta_out_path = "data/results/anomaly_meta.joblib"

joblib.dump(mse_scores, mse_out_path)
joblib.dump(preds, preds_out_path)
joblib.dump({
    "threshold_method": "percentile",
    "threshold": threshold,
    "percentile": 99,
    "mse_mean": float(np.mean(mse_scores)),
    "mse_std": float(np.std(mse_scores)),
    "num_anomalies": int(num_anomalies),
    "anomaly_ratio": float(anomaly_ratio)
}, meta_out_path)

print(f"Resultados guardados en:\n - {mse_out_path}\n - {preds_out_path}\n - {meta_out_path}")

# Paso 6: Visualización de resultados
print("\n" + "="*50)
print("Visualizando resultados...")

plt.figure(figsize=(12, 6))

# Histograma de MSE
plt.subplot(1, 2, 1)
plt.hist(mse_scores, bins=100, alpha=0.7, color='blue', log=True)
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2)
plt.title('Distribución de MSE (escala log)')
plt.xlabel('MSE')
plt.ylabel('Frecuencia (log)')
plt.legend(['Umbral de anomalía', 'Distribución MSE'])
plt.grid(True)

# Distribución de anomalías
plt.subplot(1, 2, 2)
labels = ['Normales', 'Anomalías']
sizes = [len(preds) - num_anomalies, num_anomalies]
colors = ['#66b3ff', '#ff9999']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title(f'Detección de Anomalías: {num_anomalies} encontradas')

plt.tight_layout()
plt.savefig('data/results/anomaly_detection_results.png', dpi=150)
plt.show()

print("\n" + "="*50)
print("¡Pipeline completado con éxito!")
print(f"Resultados guardados en 'data/results/'")
print(f"Imagen de resultados: data/results/anomaly_detection_results.png")