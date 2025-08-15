from pathlib import Path
import pandas as pd
import re

__all__ = ["preprocess_hdfs"]

PATTERN = re.compile(r"(blk_-?\d+)", re.IGNORECASE)

def preprocess_hdfs(
    log_path: str | Path = "data/raw/hdfs/HDFS.log",
    labels_path: str | Path = "data/raw/hdfs/anomaly_label.csv",
    out_path: str | Path = "data/processed/hdfs_processed.csv",
    return_df: bool = False,
):
    """
    Lee los logs HDFS y etiquetas, une por block_id y guarda el CSV procesado.

    Params
    ------
    log_path : ruta al HDFS.log
    labels_path : ruta al anomaly_label.csv
    out_path : ruta de salida del CSV procesado
    return_df : si True, devuelve el DataFrame además de guardar el archivo
    """
    log_path, labels_path, out_path = map(Path, (log_path, labels_path, out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Read raw lines
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]
    logs = pd.DataFrame({"message": lines})
    logs["block_id"] = logs["message"].str.extract(PATTERN, expand=False).str.lower()

    found = logs["block_id"].notna().sum()
    if found == 0:
        sample = logs["message"].head(5).tolist()
        raise ValueError("No block_id extracted. Sample: " + repr(sample))

    # 2) Load labels and normalize
    labels = pd.read_csv(labels_path, header=None, names=["block_id", "label"])
    labels["block_id"] = labels["block_id"].astype(str).str.lower()

    # Map string labels -> ints; fall back to numeric if already numbers
    # Accepted strings (case-insensitive): normal=0, anomaly=1, anomalous=1, abnormal=1
    map_dict = {"normal": 0, "anomaly": 1, "anomalous": 1, "abnormal": 1}
    if labels["label"].dtype == object:
        lab_str = labels["label"].astype(str).str.strip().str.lower()
        lab_num = lab_str.map(map_dict)
        # If still NaN (e.g., already "0"/"1" as strings), try numeric
        lab_num = lab_num.fillna(pd.to_numeric(lab_str, errors="coerce"))
        labels["label"] = lab_num
    else:
        labels["label"] = pd.to_numeric(labels["label"], errors="coerce")

    # Any unknowns -> assume normal (0); then cast to int
    labels["label"] = labels["label"].fillna(0).astype(int)

    # 3) Merge + finalize
    df = pd.merge(logs, labels, on="block_id", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    print(
        f"[preprocess] lines={len(df)}  with_blk={found}  "
        f"unique_blk_in_logs={df['block_id'].nunique(dropna=True)}  "
        f"unique_blk_in_labels={labels['block_id'].nunique()}  "
        f"anomaly_count={int((df['label']==1).sum())}"
    )

    df.to_csv(out_path, index=False)
    return df if return_df else str(out_path)