import os
from pathlib import Path
from urllib.parse import urlparse, unquote
import pandas as pd

def _uri_dir(uri: str) -> str:
    p = urlparse(uri)
    path = f"//{p.netloc}{p.path}" if p.netloc else p.path  # handles file:// and UNC
    return os.path.dirname(unquote(path))

def check_image_csv(csv_path: Path):
    df = pd.read_csv(csv_path, low_memory=False)
    # find the URL_DAPI column (case/spacing tolerant)
    col = next(c for c in df.columns if "url" in c.lower() and "dapi" in c.lower())
    dirs = df[col].dropna().map(_uri_dir).unique()
    ok = (len(dirs) == 1)
    print(f"{'OK ' if ok else 'PROBLEM'} | {csv_path} | DAPI_DIRS={list(dirs)}")
    return ok, dirs

def scan_output_root(root: str):
    for csv in Path(root).rglob("Image.csv"):
        check_image_csv(csv)

# --- usage ---
scan_output_root("file:///home/projects/hornsteinlab/Collaboration/NOVA/cell_profiler/outputs/filtered_by_brenner_post_rescale_outputs/OPERA_dNLS_6_batches_NOVA_sorted/batch2/dNLS/panelH/Untreated/rep1/DAPI")
