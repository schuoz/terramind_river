"""
create_hdf5.py
==============
Preprocesses the TIF images and annotation CSV into a single HDF5 file
(dataset.h5) for fast training I/O.

Matching: The filename convention is <FID>_<...>.tif  →  the FID is extracted
from everything BEFORE the first underscore.

Usage:
    python create_hdf5.py \\
        --annotation  data/Annotation.csv \\
        --img_dirs    /path/to/train_tifs /path/to/val_tifs \\
        --output      dataset.h5 \\
        --img_size    224
"""

import argparse
import os
import glob
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import h5py
from tqdm import tqdm


# ── CLI ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation", default="data/Annotation.csv")
    p.add_argument("--img_dirs",   nargs="+", required=True,
                   help="One or more directories containing *.tif named <FID>_*.tif")
    p.add_argument("--output",     default="dataset.h5")
    p.add_argument("--img_size",   type=int, default=224,
                   help="H=W to resize every image to")
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────
def build_fid_map(img_dirs):
    """Return {FID (int): absolute path} by parsing filenames."""
    fid_map = {}
    for d in img_dirs:
        for path in glob.glob(os.path.join(d, "*.tif")):
            basename = os.path.basename(path)
            # FID is the part before the first '_'
            fid_str = basename.split("_")[0]
            try:
                fid = int(fid_str)
                fid_map[fid] = path
            except ValueError:
                pass  # skip malformed names
    return fid_map


def read_and_resize(path, size):
    """Read a TIF and return a (C, H, W) float32 array, resized to (size, size)."""
    with rasterio.open(path) as src:
        data = src.read(
            out_shape=(src.count, size, size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    return data          # (C, H, W)


def per_band_normalize(arr):
    """Normalise each band independently to [0, 1] using 2nd–98th percentile."""
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        band = arr[c]
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi > lo:
            out[c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        else:
            out[c] = 0.0
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    # 1. Load annotation
    df = pd.read_csv(args.annotation)
    print(f"Loaded annotation with {len(df)} rows.")
    print(f"Columns: {list(df.columns)}")

    # Normalise TRUESKILL_MEAN from its empirical range to [0, 1]
    ts_min, ts_max = df["TRUESKILL_MEAN"].min(), df["TRUESKILL_MEAN"].max()
    df["SCORE_NORM"] = (df["TRUESKILL_MEAN"] - ts_min) / (ts_max - ts_min)
    # Keep STD as-is (it's already a small positive number)
    df["SCORE_STD"]  = df["TRUESKILL_STD"]

    print(f"TRUESKILL_MEAN range: [{ts_min:.4f}, {ts_max:.4f}]  →  normalised to [0,1]")

    # 2. Build FID → path map
    fid_map = build_fid_map(args.img_dirs)
    print(f"Found {len(fid_map)} TIF files across all image directories.")

    # 3. Filter annotation to rows that have a matching TIF
    df = df[df["FID"].isin(fid_map)].reset_index(drop=True)
    print(f"Matched {len(df)} annotation rows to TIF files.")

    if len(df) == 0:
        raise RuntimeError("No annotation rows matched any TIF file. "
                           "Check that --img_dirs and --annotation paths are correct.")

    # Probe one image to know # bands
    sample_path = fid_map[df["FID"].iloc[0]]
    with rasterio.open(sample_path) as src:
        n_bands = src.count
    print(f"TIF band count: {n_bands}, resizing to {args.img_size}×{args.img_size}")

    # 4. Write HDF5
    N = len(df)
    S = args.img_size
    C = n_bands

    with h5py.File(args.output, "w") as f:
        # Metadata attributes so downstream code can read them back
        f.attrs["ts_min"]   = ts_min
        f.attrs["ts_max"]   = ts_max
        f.attrs["n_bands"]  = C
        f.attrs["img_size"] = S

        # Pre-allocate datasets
        imgs   = f.create_dataset("images",    shape=(N, C, S, S), dtype=np.float32,
                                  chunks=(1, C, S, S), compression="lzf")
        scores = f.create_dataset("score_norm", shape=(N,),          dtype=np.float32)
        stds   = f.create_dataset("score_std",  shape=(N,),          dtype=np.float32)
        lats   = f.create_dataset("lat",        shape=(N,),          dtype=np.float32)
        lons   = f.create_dataset("lon",        shape=(N,),          dtype=np.float32)
        fids   = f.create_dataset("fid",        shape=(N,),          dtype=np.int32)

        for i, row in tqdm(df.iterrows(), total=N, desc="Writing HDF5"):
            path = fid_map[int(row["FID"])]
            img  = read_and_resize(path, S)
            img  = per_band_normalize(img)

            imgs[i]   = img
            scores[i] = float(row["SCORE_NORM"])
            stds[i]   = float(row["SCORE_STD"])
            lats[i]   = float(row["LAT"])
            lons[i]   = float(row["LON"])
            fids[i]   = int(row["FID"])

    print(f"\nSaved {N} samples to '{args.output}'.")


if __name__ == "__main__":
    main()
