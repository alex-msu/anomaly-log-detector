from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
import gc
import scipy.sparse
import os

__all__ = ["vectorize_hdfs"]

def vectorize_hdfs(
    input_csv: str | Path = "data/processed/hdfs_processed.csv",
    max_features: int = 3000,
    stop_words: str = "english",
    ngram_range: tuple = (1, 1),
    x_out_path: str | Path = "data/processed/X_tfidf_sparse.joblib",
    y_out_path: str | Path = "data/processed/y.joblib",
    vectorizer_out_path: str | Path = "models/tfidf_vectorizer.joblib",
    chunksize: int = 20000,
    verbose: int = 1
):
    input_csv = Path(input_csv)
    x_out_path = Path(x_out_path)
    y_out_path = Path(y_out_path)
    vectorizer_out_path = Path(vectorizer_out_path)

    # Asegurar directorios
    x_out_path.parent.mkdir(parents=True, exist_ok=True)
    y_out_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Primera pasada: Construir vocabulario con muestra
    if verbose:
        print("Fase 1/3: Construyendo vocabulario con muestra...")
    
    # Leer una muestra representativa (500k registros)
    sample_size = min(500000, sum(1 for _ in open(input_csv)) - 1)
    sample = pd.read_csv(input_csv, usecols=['message'], nrows=sample_size)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range,
        dtype=np.float32
    )
    vectorizer.fit(sample['message'])
    
    del sample
    gc.collect()

    # 2. Segunda pasada: Transformar datos en bloques
    if verbose:
        print("Fase 2/3: Transformando datos en bloques...")
    
    chunks = pd.read_csv(input_csv, usecols=['message', 'label'], chunksize=chunksize)
    X_chunks = []
    y_chunks = []
    
    for chunk in tqdm(chunks, disable=verbose<2):
        X_chunk = vectorizer.transform(chunk["message"])
        X_chunks.append(X_chunk)
        y_chunks.append(chunk["label"].values)
        del chunk
        gc.collect()

    # 3. Combinar resultados y guardar
    if verbose:
        print("Fase 3/3: Guardando resultados...")
    
    # Guardar y como array numpy
    y = np.concatenate(y_chunks)
    joblib.dump(y, y_out_path, compress=3)
    
    # Guardar X como matriz dispersa
    X = scipy.sparse.vstack(X_chunks) if len(X_chunks) > 0 else None
    joblib.dump(X, x_out_path, compress=3)
    
    # Guardar vectorizador
    joblib.dump(vectorizer, vectorizer_out_path, compress=3)
    
    # Liberar memoria
    del X, y, X_chunks, y_chunks
    gc.collect()

    return str(x_out_path), str(y_out_path), str(vectorizer_out_path)