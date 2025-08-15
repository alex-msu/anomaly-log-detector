# src/vectorize.py
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

__all__ = ["vectorize_hdfs"]

def vectorize_hdfs(
    input_csv: str | Path = "data/processed/hdfs_processed.csv",
    max_features: int = 1000,
    stop_words: str = "english",
    ngram_range: tuple = (1, 2),
    x_out_path: str | Path = "data/processed/X_tfidf.joblib",
    y_out_path: str | Path = "data/processed/y.joblib",
    vectorizer_out_path: str | Path = "models/tfidf_vectorizer.joblib",
    return_data: bool = False
):
    """
    Vectoriza el texto en HDFS usando TF-IDF y guarda X, y y el vectorizador.

    Params
    ------
    input_csv : ruta al CSV procesado
    max_features : número máximo de características TF-IDF
    stop_words : idioma para palabras vacías
    ngram_range : rango de n-gramas
    x_out_path : ruta de salida para la matriz X
    y_out_path : ruta de salida para el vector y
    vectorizer_out_path : ruta para guardar el vectorizador
    return_data : si True, devuelve (X, y, vectorizer) además de guardarlos
    """
    input_csv = Path(input_csv)
    x_out_path = Path(x_out_path)
    y_out_path = Path(y_out_path)
    vectorizer_out_path = Path(vectorizer_out_path)

    # Ensure directories exist
    x_out_path.parent.mkdir(parents=True, exist_ok=True)
    y_out_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_csv)

    assert "message" in df.columns and len(df) > 0

    # Vectorización
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(df["message"])
    y = df["label"].values

    # Guardar artefactos
    joblib.dump(X, x_out_path)
    joblib.dump(y, y_out_path)
    joblib.dump(vectorizer, vectorizer_out_path)

    if return_data:
        return X, y, vectorizer
    else:
        return str(x_out_path), str(y_out_path), str(vectorizer_out_path)
