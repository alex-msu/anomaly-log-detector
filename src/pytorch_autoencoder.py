import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
from pathlib import Path
import time
import gc
from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # Capas más eficientes con inicialización adecuada
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
        
        # Inicialización de pesos
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder_hdfs(
    x_path: str | Path = "data/processed/X_tfidf_sparse.joblib",
    y_path: str | Path = "data/processed/y.joblib",
    model_out_path: str | Path = "models/autoencoder_hdfs.pt",
    encoding_dim: int = 32,
    hidden_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 1024,
    val_ratio: float = 0.1,
    learning_rate: float = 0.001,
    patience: int = 3,
    seed: int = 42,
    verbose: int = 1,
):
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Usando dispositivo: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Habilitar optimizaciones de cuDNN
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Configurar semilla
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Cargar solo los datos necesarios
    if verbose:
        print("Cargando datos...")
    
    X = joblib.load(x_path)
    if hasattr(X, "toarray"):
        X = X.toarray().astype("float32")
    else:
        X = X.astype("float32")
    
    y = joblib.load(y_path)
    
    # Filtrar solo muestras normales (label == 0)
    normal_mask = (y == 0)
    X_normal = X[normal_mask]
    
    # Liberar memoria inmediatamente
    del X, y
    gc.collect()
    
    # Usar solo una muestra del 50% para acelerar (en Colab)
    if 'COLAB_GPU' in os.environ:
        sample_size = min(500000, len(X_normal))
        idx = np.random.choice(len(X_normal), sample_size, replace=False)
        X_normal = X_normal[idx]
        if verbose:
            print(f"Usando muestra de {sample_size} registros para entrenamiento")
    
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
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=True)
    
    # Inicializar modelo
    input_dim = X_train_tensor.shape[1]
    model = Autoencoder(input_dim, encoding_dim, hidden_dim).to(device)
    
    # Usar mixed precision para acelerar entrenamiento en GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
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
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
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
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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