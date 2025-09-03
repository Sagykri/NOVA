#!/usr/bin/env python3
# Split each TIFF into 1080x1080 tiles; save tiles to a mirrored tree under dst_root.
# Requirements: pip install tifffile numpy

import os
from pathlib import Path
import numpy as np
import tifffile as tiff

TILE = 1080  # tile size

def split_all(src_root: str, dst_root: str) -> None:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    for dirpath, _, files in os.walk(src_root):
        # find the first .tif/.tiff (assumes one per subfolder)
        tifs = [f for f in files if Path(f).suffix.lower() in {".tif", ".tiff"}]
        if not tifs:
            continue
        img_path = Path(dirpath) / tifs[0]

        # mirror subfolder under dst_root
        rel = Path(dirpath).relative_to(src_root)
        out_dir = (dst_root / rel)
        out_dir.mkdir(parents=True, exist_ok=True)

        # read image (single page)
        img = tiff.imread(str(img_path))              # (H,W) or (H,W,C)
        H, W = img.shape[:2]
        y_steps = range(0, H - (H % TILE), TILE)      # drop remainders
        x_steps = range(0, W - (W % TILE), TILE)

        idx = 0
        stem = img_path.stem
        for y in y_steps:
            for x in x_steps:
                idx += 1
                tile = img[y:y+TILE, x:x+TILE, ...]
                out_name = f"{stem}_s{idx}.tif"
                print(f"From {img_path} to {out_dir / out_name}, shape={tile.shape}")
                tiff.imwrite(str(out_dir / out_name), tile)
        print(f"[{rel}] {img_path.name} -> {idx} tiles")

if __name__ == "__main__":
    SRC = r"/home/projects/hornsteinlab/Collaboration/NOVA/input/images/raw/U2OS"
    DST = r"/home/projects/hornsteinlab/Collaboration/NOVA/input/images/raw/U2OS_sorted"
    split_all(SRC, DST)
