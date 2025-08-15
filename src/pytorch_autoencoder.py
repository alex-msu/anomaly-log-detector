import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
from pathlib import Path
import time
import gc
from tqdm import tqdm
import scipy.sparse

class SparseTensorDataset(Dataset):
    """Dataset para cargar matrices dispersas eficientemente"""
    def __init__(self, data, device='cpu'):
        self.data = data
        self.device = device
        self.shape = data.shape
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Convertir a denso solo la fila necesaria
        sample = torch.tensor(self.data[idx].toarray().squeeze(), dtype=torch.float32)
        return sample.to(self.device), sample.to(self.device)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # Arquitectura optimizada para bajo consumo de memoria
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
        return self.decoder(self.encoder(x))

def train_autoencoder_hdfs(
    x_path: str | Path = "data/processed/X_tfidf_sparse.joblib",
    y_path: str | Path = "data/processed/y.joblib",
    model_out_path: str | Path = "models/autoencoder_hdfs.pt",
    encoding_dim: int = 24,
    hidden_dim: int = 48,
    epochs: int = 10,
    batch_size: int = 2048,
    val_ratio: float = 0.1,
    learning_rate: float = 0.001,
    patience: int = 2,
    seed: int = 42,
    verbose: int = 1,
):
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Usando dispositivo: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Habilitar optimizaciones
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Configurar semilla
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    if verbose:
        print("Cargando datos...")
    
    # Cargar solo las etiquetas para filtrar
    y = joblib.load(y_path)
    normal_mask = (y == 0)
    del y
    gc.collect()
    
    # Cargar datos dispersos directamente
    X_sparse = joblib.load(x_path)
    
    # Filtrar solo muestras normales
    X_normal = X_sparse[normal_mask]
    del X_sparse, normal_mask
    gc.collect()
    
    # Dividir índices en lugar de datos
    num_samples = X_normal.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int((1 - val_ratio) * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]
    
    # Crear datasets que cargan bajo demanda
    train_dataset = SparseTensorDataset(X_normal[train_indices], device)
    val_dataset = SparseTensorDataset(X_normal[val_indices], device)
    
    del X_normal
    gc.collect()
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Inicializar modelo
    input_dim = train_dataset.shape[1]
    model = Autoencoder(input_dim, encoding_dim, hidden_dim).to(device)
    
    # Mixed precision para GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
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
        train_batches = 0
        
        # Barra de progreso
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=verbose<2)
        
        for inputs, targets in batch_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            batch_iter.set_postfix(loss=loss.item())
        
        train_loss /= train_batches
        history['train_loss'].append(train_loss)
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        history['val_loss'].append(val_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_out_path)
            if verbose:
                print(f"Modelo guardado (val_loss={val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping en epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    if verbose:
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        print(f"Mejor val_loss: {best_val_loss:.6f}")
    
    return history, str(model_out_path)