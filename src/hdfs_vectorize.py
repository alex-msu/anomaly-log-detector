from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
import gc
import scipy.sparse

__all__ = ["vectorize_hdfs"]

def vectorize_hdfs(
    input_csv: str | Path = "data/processed/hdfs_processed.csv",
    max_features: int = 5000,
    stop_words: str = "english",
    ngram_range: tuple = (1, 1),
    x_out_path: str | Path = "data/processed/X_tfidf_sparse.joblib",
    y_out_path: str | Path = "data/processed/y.joblib",
    vectorizer_out_path: str | Path = "models/tfidf_vectorizer.joblib",
    return_data: bool = False,
    chunksize: int = 20000,
    verbose: int = 1
):
    if verbose:
        print(f"Configuración de vectorización:")
        print(f"- max_features: {max_features}")
        print(f"- ngram_range: {ngram_range}")
        print(f"- chunksize: {chunksize}")
        print(f"- output: {x_out_path}")

    input_csv = Path(input_csv)
    x_out_path = Path(x_out_path)
    y_out_path = Path(y_out_path)
    vectorizer_out_path = Path(vectorizer_out_path)

    # Asegurar directorios
    x_out_path.parent.mkdir(parents=True, exist_ok=True)
    y_out_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Aprender vocabulario en bloques
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range,
        dtype=np.float32  # Usar float32 para ahorrar memoria
    )
    
    if verbose:
        print("Fase 1/2: Aprendiendo vocabulario...")
    
    # Solo necesitamos la columna de texto para el aprendizaje
    text_chunks = pd.read_csv(input_csv, usecols=['message'], chunksize=chunksize)
    
    for chunk in tqdm(text_chunks, disable=verbose<2):
        vectorizer.partial_fit(chunk["message"])
        del chunk
        gc.collect()

    # 2. Transformar datos en bloques
    if verbose:
        print("Fase 2/2: Transformando datos...")
    
    chunks = pd.read_csv(input_csv, usecols=['message', 'label'], chunksize=chunksize)
    X_chunks = []
    y_chunks = []
    
    for chunk in tqdm(chunks, disable=verbose<2):
        X_chunk = vectorizer.transform(chunk["message"])
        X_chunks.append(X_chunk)
        y_chunks.append(chunk["label"].values)
        del chunk, X_chunk
        gc.collect()

    # Combinar resultados
    X = scipy.sparse.vstack(X_chunks) if len(X_chunks) > 0 else None
    y = np.concatenate(y_chunks) if len(y_chunks) > 0 else np.array([])
    
    # Guardar en formato eficiente
    if verbose:
        print("Guardando resultados...")
    
    joblib.dump(X, x_out_path, compress=3)
    joblib.dump(y, y_out_path, compress=3)
    joblib.dump(vectorizer, vectorizer_out_path, compress=3)
    
    # Liberar memoria antes de salir
    del X_chunks, y_chunks
    gc.collect()

    if return_data:
        return X, y, vectorizer
        
    return str(x_out_path), str(y_out_path), str(vectorizer_out_path)