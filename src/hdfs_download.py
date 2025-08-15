from pathlib import Path
import zipfile, shutil, requests

__all__ = ["download_file", "extract_selected"]

def download_file(url: str, dest_path: str | Path, *, skip_if_exists: bool = True, chunk_size: int = 8192) -> str:
    """
    Descarga un archivo en tiempo real desde `url` a `dest_path`.
    Devuelve la ruta de destino como un string.
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        print(f"[SKIP] ZIP ya existe en {dest} (usa skip_if_exists=False para forzar una re-descarga)")
        return str(dest)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print(f"[SUCCESS] Descargado -> {dest}")
    return str(dest)

def extract_selected(
    zip_path: str | Path,
    extract_dir: str | Path,
    files_to_extract: list[str],
    *,
    flatten: bool = True,
    overwrite: bool = False,   # default to NOT overwrite
) -> list[str]:
    """
    Extrae solo `files_to_extract` de `zip_path` a `extract_dir`.
    Si `flatten` es verdadero, elimina las carpetas ZIP internas y coloca los archivos directamente en `extract_dir`.
    Devuelve la lista de rutas de archivo escritas.
    """
    zip_path, extract_dir = Path(zip_path), Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())
        for member in files_to_extract:
            if member not in names:
                print(f"[ADVERTENCIA] '{member}' no se encuentra en el archivo")
                continue

            target_name = Path(member).name if flatten else member
            dest = extract_dir / target_name
            if dest.exists() and not overwrite:
                print(f"[SKIP] {dest} ya existe (overwrite=False)")
                written.append(str(dest))
                continue

            if flatten:
                with z.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                z.extract(member, extract_dir)
            print(f"[SUCCESS] Extracción de '{member}' satisfactoria -> '{dest}'")
            written.append(str(dest))
    return written