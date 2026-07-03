from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

def check_image_shapes(dir1, dir2):
    """
    Compares image shapes between two directories (e.g. S1 vs S2)

    Assumes matching filenames.
    """

    dir1 = Path(dir1)
    dir2 = Path(dir2)

    mismatches = []
    missing = []

    files1 = {f.name for f in dir1.glob("*.tif")}

    for fname in files1:
        path1 = dir1 / fname
        path2 = dir2 / fname

        if not path2.exists():
            missing.append(fname)
            continue

        with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
            shape1 = (src1.height, src1.width)
            shape2 = (src2.height, src2.width)

        if shape1 != shape2:
            mismatches.append((fname, shape1, shape2))

    print("\nShape check results:")
    print(f"Checked: {len(files1)} files")
    print(f"Missing in dir2: {len(missing)}")
    print(f"Mismatched shapes: {len(mismatches)}")

    return {
        "missing": missing,
        "mismatches": mismatches
    }

def get_band_min_max(dir_path):
    """Compute the global per-band minimum and maximum across all samples"""
    dir_path = Path(dir_path)

    band_mins = None
    band_maxs = None

    # iterate over all tif in the dir
    for tif_path in dir_path.glob("*.tif"):
        with rasterio.open(tif_path) as src:
            data = src.read()  

            # init accumulators on first file
            if band_mins is None:
                band_mins = np.full(data.shape[0], np.inf)
                band_maxs = np.full(data.shape[0], -np.inf)

            # update running min/max per band
            for b in range(data.shape[0]):
                band_mins[b] = min(band_mins[b], np.min(data[b]))
                band_maxs[b] = max(band_maxs[b], np.max(data[b]))

    print("\nPer-band stats:")
    for i, (mn, mx) in enumerate(zip(band_mins, band_maxs)):
        print(f"Band {i+1}: min={mn:.3f}, max={mx:.3f}")

    return band_mins, band_maxs

def get_band_percentiles(dir_path, percentiles=[1, 5, 50, 95, 99]):
    """
    Compute global per-band percentiles across all samples
    """
    dir_path = Path(dir_path)

    band_values = None

    # iterate over all tif in the dir
    for tif_path in dir_path.glob("*.tif"):
        with rasterio.open(tif_path) as src:
            data = src.read()  # [C, H, W]

            # init one value list per band on first file
            if band_values is None:
                band_values = [[] for _ in range(data.shape[0])]

            for b in range(data.shape[0]):
                # flatten
                vals = data[b].ravel()
                # pool this file's values into the band
                band_values[b].append(vals)

    print("\nPer-band percentiles:")

    results = {}

    for b, vals_list in enumerate(band_values):
        # combine all files' values for this band
        all_vals = np.concatenate(vals_list) 

        p_vals = np.percentile(all_vals, percentiles)

        results[b] = dict(zip(percentiles, p_vals))

        print(f"\nBand {b+1}:")
        for p, v in zip(percentiles, p_vals):
            print(f"  P{p}: {v:.3f}")

    return results

def get_max_time_difference_with_row(csv_path, sentinel_timestamp):
    """
    Find the row with the largest time gap between the flood map date and the
    Sentinel acquisition timestamp.
    """
    df = pd.read_csv(csv_path)

    df["floodmap_date"] = pd.to_datetime(df["floodmap_date"], dayfirst=True, errors="coerce")
    df[sentinel_timestamp] = pd.to_datetime(df[sentinel_timestamp], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["floodmap_date", sentinel_timestamp])

    df["time_diff_hours"] = (
        (df[sentinel_timestamp] - df["floodmap_date"])
        .abs()
        .dt.total_seconds() / 3600
    )

    idx = df["time_diff_hours"].idxmax()
    row = df.loc[idx]

    print(f"Max time difference: {row['time_diff_hours']:.2f} hours")
    print(row)

    return row
