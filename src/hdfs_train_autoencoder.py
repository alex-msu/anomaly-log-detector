# src/train_autoencoder_hdfs.py
from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

__all__ = ["train_autoencoder_hdfs"]

def train_autoencoder_hdfs(
    x_path: str | Path = "data/processed/X_tfidf.joblib",
    y_path: str | Path = "data/processed/y.joblib",
    model_out_path: str | Path = "models/autoencoder_hdfs.h5",
    encoding_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 50,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    optimizer: str = "adam",
    loss: str = "mse",
    patience: int = 5,
    seed: int = 42,
    verbose: int = 1,
):
    """
    Entrenar un autoencoder sobre los mensajes HDFS normales (label==0)

    Retorna
    -------
    history_dict : dict
        Keras History.history
    saved_path : str
        Path to the saved best model (.h5)
    """
    x_path = Path(x_path)
    y_path = Path(y_path)
    model_out_path = Path(model_out_path)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cargar data
    X = joblib.load(x_path)
    if hasattr(X, "toarray"):  # manejar matrices dispersas
        X = X.toarray()
    X = X.astype("float32", copy=False)

    y = joblib.load(y_path)

    # Quedarse solo con las samples normales para el entrenamiento
    normal_mask = (y == 0)
    X_normal = X[normal_mask]

    # Shuffle + split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_normal))
    X_normal = X_normal[idx]

    split = int((1 - val_ratio) * len(X_normal))
    X_train, X_val = X_normal[:split], X_normal[split:]

    # Modelo
    input_dim = X.shape[1]
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dense(encoding_dim, activation="relu")(x)
    x = layers.Dense(hidden_dim, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(filepath=str(model_out_path), monitor="val_loss", save_best_only=True)
    ]

    # Entrenamiento
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        verbose=verbose,
        callbacks=callbacks,
    )

    return history.history, str(model_out_path)
