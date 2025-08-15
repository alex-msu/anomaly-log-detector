import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
from pathlib import Path
import os
import time
import gc

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder_hdfs(
    x_path: str | Path = "data/processed/X_tfidf_sparse.joblib",
    y_path: str | Path = "data/processed/y.joblib",
    model_out_path: str | Path = "models/autoencoder_hdfs.pt",
    encoding_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 50,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    learning_rate: float = 0.001,
    patience: int = 5,
    seed: int = 42,
    verbose: int = 1,
):
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Usando dispositivo: {device}")
    
    # Configurar semilla
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Cargar datos
    X = joblib.load(x_path)
    if hasattr(X, "toarray"):
        X = X.toarray().astype("float32")
    else:
        X = X.astype("float32")
    
    y = joblib.load(y_path)

    # Filtrar solo muestras normales
    normal_mask = (y == 0)
    X_normal = X[normal_mask]
    
    # Liberar memoria temprana
    del X, y
    gc.collect()

    # Shuffle y split
    idx = np.random.permutation(len(X_normal))
    X_normal = X_normal[idx]
    split = int((1 - val_ratio) * len(X_normal))
    X_train, X_val = X_normal[:split], X_normal[split:]
    
    # Convertir a tensores y mover a dispositivo
    X_train_tensor = torch.tensor(X_train).float().to(device)
    X_val_tensor = torch.tensor(X_val).float().to(device)
    
    # Liberar más memoria
    del X_normal, X_train, X_val
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Preparar datasets y dataloaders
    train_data = TensorDataset(X_train_tensor, X_train_tensor)
    val_data = TensorDataset(X_val_tensor, X_val_tensor)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Inicializar modelo
    input_dim = X_train_tensor.shape[1]
    model = Autoencoder(input_dim, encoding_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Crear directorio para el modelo
    model_out_path = Path(model_out_path)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Entrenamiento
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)

        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_out_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break

    training_time = time.time() - start_time
    if verbose:
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        print(f"Mejor pérdida de validación: {best_val_loss:.6f}")
        print(f"Modelo guardado en: {model_out_path}")

    return history, str(model_out_path)