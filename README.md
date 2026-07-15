# STURM-fusion — Instruction Manual

This repository builds the **STURM-fusion** dataset: it fuses the original
**STURM-Flood** dataset (Sentinel-2 optical tiles + flood masks) with newly-exported
**Sentinel-1** SAR imagery, matched to each Sentinel-2 tile via Google Earth Engine (GEE),
then preprocesses, validates, and packages the result for upload to Hugging Face.

This is the *dataset-building* repository. The companion
[Flood-Mapping](../Flood-Mapping) repository trains and evaluates segmentation models
against the dataset this repo produces.

The entire pipeline runs from one notebook, [Create_Dataset.ipynb](Create_Dataset.ipynb):
match S1↔S2, export S1 from GEE, assemble, preprocess, validate, push to Hugging Face.

---
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/YOUR-USERNAME/STURM-Fusion-24)
---

## 1. Requirements

Before running anything, you need:

- **A Google account with Google Drive.** GEE's batch export
  (`ee.batch.Export.image.toDrive`, in [src/gee/export.py](src/gee/export.py)) is a
  **server-side** export — it can only write to *your* Google Drive, Google Cloud
  Storage, or an EE Asset. There is no "export straight to this machine" option. The
  pipeline therefore needs your Drive mounted so it can read the exported Sentinel-1
  GeoTIFFs back after each export job finishes — mounting Drive is mandatory in the
  Setup cell (see §4).
- **A Google Earth Engine account, with a registered Cloud project.** Needed for the
  Sentinel-1 search and export steps
  ([src/gee/s1_collection.py](src/gee/s1_collection.py),
  [src/gee/matching.py](src/gee/matching.py),
  [src/gee/export.py](src/gee/export.py)). See §2.
- **Python packages**: `earthengine-api`, `rasterio`, `pandas`, `numpy`, `scipy`,
  `huggingface_hub`. (There's no `requirements.txt` yet — install these directly, e.g.
  `!pip install earthengine-api rasterio pandas numpy scipy huggingface_hub`.)
- **A Hugging Face account and access token** — only needed if you intend to push the
  finished dataset (`push_to_HF=True` in the Setup cell; see §5/§8).

---

## 2. Setting up a Google Earth Engine account

The Sentinel-1 search and export steps ([src/gee/s1_collection.py](src/gee/s1_collection.py),
[src/gee/matching.py](src/gee/matching.py), [src/gee/export.py](src/gee/export.py)) all run
against Earth Engine, which now requires a registered **Cloud project** — the old
standalone-account model has been phased out. You set the project's ID as `gee_project`
in the Setup cell (§4), which becomes `cfg.GEE_PROJECT`.

1. **Sign up for Google Earth Engine.** Go to
   [https://earthengine.google.com/](https://earthengine.google.com/), click **Get
   Started** (top right), and sign in with your Google account (the same account whose
   Drive you'll mount in §4 — GEE exports land there, per §1).

2. **Register a Cloud project.** After signing in you'll be prompted to register a Cloud
   project. Choose either **Register an existing Cloud project** or **Create a new Cloud
   project** (the latter is easiest if you don't already have one).

3. **Choose the project use type.** Select **commercial** or **noncommercial/unpaid**.
   Academic, research, and personal use qualify as noncommercial and are free; commercial
   use requires a linked billing account.

4. **Create the project** (if making a new one). Enter a globally-unique **Project ID**
   (e.g. `my-ee-project-2026`) and a display **Project name**, link a **billing account**
   if commercial, then accept the terms and click **Continue / Register**.

5. **Enable the Earth Engine API.** If this isn't done automatically, open the
   [Google Cloud Console](https://console.cloud.google.com/), select your project, search
   for **Earth Engine API**, and click **Enable**.

6. **Set `gee_project`.** Put your **Project ID** (the alphanumeric ID from step 4, *not*
   the numeric project number) into the Setup cell's `gee_project` variable (§4). That's
   the value the pipeline authenticates and exports with.

> Registration is usually approved instantly for noncommercial use, but can occasionally
> take a day or two. You need a Cloud project attached to use Earth Engine at all — there
> is no account-only mode anymore.

---

## 3. Repository layout

```text
STURM-fusion/
├── Create_Dataset.ipynb       # main pipeline notebook
├── src/
│   ├── config.py               # CFG dataclass — every path, threshold, and step list
│   ├── data/
│   │   └── sturm_flood.py      # download_and_extract(): fetch + unzip the original STURM-Flood dataset
│   ├── gee/
│   │   ├── aoi.py              # AOI extraction + export-grid alignment from a reference S2 tile
│   │   ├── s1_collection.py    # Sentinel-1 GRD collection query (IW mode, VV+VH)
│   │   ├── matching.py         # best S1 image selection + AOI-coverage check
│   │   ├── export.py           # ee.batch.Export.image.toDrive() submission
│   │   └── tasks.py            # list/cancel/wait on GEE batch tasks
│   ├── pipeline/
│   │   ├── matching.py         # process_sample/process_csv: S1<->S2 matching, builds fusion metadata
│   │   ├── export.py           # export_all_s1_images: submit export tasks for matched images
│   │   ├── assemble.py         # assemble_dataset: copy matched S1/S2/mask files into place
│   │   ├── preprocessing.py    # S1/S2 preprocessing pipeline (resumable, tag-based)
│   │   └── validation.py       # completeness / NaN-ratio checks, remove_bad_nan_files
│   ├── preprocess/
│   │   └── operations.py       # the actual per-step image transforms (crop, Lee filter, clip, normalise, ...)
│   ├── hugging_face/
│   │   └── push_dataset.py     # push_zip_to_hf(): upload Dataset.zip to a HF dataset repo
│   └── util/
│       ├── io.py                # dataset folder structure, file copying, zipping
│       ├── time_utils.py        # timestamp parsing, time-window helpers
│       └── metrics.py           # post-hoc dataset stats (shapes, band min/max/percentiles, time gaps)
```

---

## 4. Setup

`Create_Dataset.ipynb` starts with a **Setup** cell:

```python
root_path = "/content/drive/MyDrive/MSc/STURM-fusion"
mount_point = "/content/drive"
clone_repo = False
reset_export = False
gee_export = False
push_to_HF = False
cancle_gee_tasks = False
gee_project = "..."
```

| Variable | Effect |
|---|---|
| Drive mount | **Always happens** — `drive.mount(mount_point)` runs unconditionally. Google Drive is a hard requirement (§1), not a toggle. |
| `root_path` | Where the project (code + `STURM-flood/` + `STURM-fusion-<hours>h/`) lives, inside your mounted Drive. |
| `clone_repo` | `True` clones `https://github.com/TAX2310/STURM-fusion.git` into `root_path/STURM-fusion` if it doesn't already exist there. `False` assumes the repo is already present at `root_path`. |
| `gee_export` | Gates whether this run authenticates with Earth Engine and actually submits new S1 export tasks (cell 7). Leave `False` to just re-run validation/preprocessing on already-exported data. |
| `gee_project` | Your GEE Cloud project ID (§2). |
| `reset_export` | `True` clears `cfg.EXPORT_PATH` (the Drive folder GEE exports land in) before running. Use with care — this deletes files. |
| `push_to_HF` | `True` zips the finished dataset and uploads it to `cfg.HF_REPO_ID` on Hugging Face (cell 11). |
| `cancle_gee_tasks` | `True` cancels all active/queued GEE batch tasks (cell 4) — useful if a previous run left tasks stuck. |

After this cell, `cfg` is a `CFG()` instance ([src/config.py](src/config.py)) with
`cfg.ROOT`, `cfg.DRIVE_ROOT`, and `cfg.GEE_PROJECT` set.

---

## 5. Running the pipeline (`Create_Dataset.ipynb`)

1. **Setup** (§4).
2. **Cancel stale GEE tasks** (optional) — `cancel_all_tasks()` if `cancle_gee_tasks=True`.
3. **Create the dataset folder structure** — `create_dataset_structure(cfg)` builds
   `STURM-flood/` and `STURM-fusion-<hours>h/Dataset/{S1,S2,floodmaps,metadata}` under
   `cfg.ROOT`.
4. **Download + extract STURM-Flood** —
   `download_and_extract(cfg)` ([src/data/sturm_flood.py](src/data/sturm_flood.py))
   fetches the original dataset zip and extracts the Sentinel-2 tiles, flood masks, and
   metadata CSV.
5. **Match S1 to S2 and export** (only if `gee_export=True` and no tasks are already
   running) — `process_csv(cfg.OLD_S2_METADATA_CSV, cfg)`
   ([src/pipeline/matching.py](src/pipeline/matching.py)) finds, for every S2 tile, the
   closest-in-time Sentinel-1 image that fully covers the tile's AOI within
   `cfg.S1_TIME_THRESHOLD_HOURS`, then `export_all_s1_images(images, cfg)`
   ([src/pipeline/export.py](src/pipeline/export.py)) submits the GEE export tasks
   (landing in your Drive, per §1).
6. **Assemble + preprocess** — looped until complete: `assemble_dataset(cfg)`
   ([src/pipeline/assemble.py](src/pipeline/assemble.py)) copies matched S1/S2/mask
   files into the new dataset layout, then `preprocessing_s1_pipeline`/
   `preprocessing_s2_pipeline` ([src/pipeline/preprocessing.py](src/pipeline/preprocessing.py))
   run the configured step lists (`cfg.S1_PREPROCESSING_STEPS`/
   `cfg.S2_PREPROCESSING_STEPS`). This step is resumable — already-completed steps are
   tracked in each GeoTIFF's tags, and orphaned temp files from an interrupted run are
   cleaned up automatically on the next pass.
7. **Remove bad NaN files** — `remove_bad_nan_files(cfg)`
   ([src/pipeline/validation.py](src/pipeline/validation.py)) drops the S1/S2/mask
   files and metadata row for any tile whose S1 NaN/zero-pixel ratio exceeds
   `cfg.NAN_RATIO_THRESHOLD`.
8. **Inspect** — if `validate_dataset(cfg)` passes (files complete, all preprocessed,
   NaN ratio acceptable), runs `check_image_shapes`, `get_band_min_max`,
   `get_band_percentiles`, and `get_max_time_difference_with_row`
   ([src/util/metrics.py](src/util/metrics.py)) as a sanity check over the finished
   dataset.
9. **Push to Hugging Face** (only if `push_to_HF=True`) — zips the dataset
   (`zip_dataset(cfg)`) and uploads it via `push_zip_to_hf(...)`
   ([src/hugging_face/push_dataset.py](src/hugging_face/push_dataset.py)) to
   `cfg.HF_REPO_ID`.

**Resulting dataset structure**, under `cfg.NEW_DATA_PATH`:

```text
STURM-fusion-<hours>h/Dataset/
├── S1/            # Sentinel-1 GeoTIFFs, 2 bands (VV, VH), cropped/filtered/clipped/normalised
├── S2/             # Sentinel-2 GeoTIFFs (from STURM-Flood, NaN-cleaned)
├── floodmaps/      # ground-truth mask GeoTIFFs (same tile_id filenames)
└── metadata/
    └── metadata.csv   # one row per tile_id, with both sentinel1_timestamp and sentinel2_timestamp
```

---

## 6. Configuration (`src/config.py`)

`CFG` is a single dataclass holding every path, threshold, and the preprocessing step
lists. Notable fields you may want to change:

| Field | Meaning |
|---|---|
| `S1_TIME_THRESHOLD_HOURS` | ± window (hours) used when searching for a matching S1 image around the flood date |
| `S2_TIME_THRESHOLD_HOURS` | Max allowed gap (hours) between the S2 acquisition and the flood date — tiles outside this are skipped |
| `S1_COVERAGE_THRESHOLD` | Minimum fraction of the AOI the candidate S1 footprint must cover |
| `S1_CROP_SIZE` | S1 crop size in pixels (square) |
| `LEE_FILTER_SIZE` | Lee speckle filter window size in pixels |
| `S1_BAND_MINS` / `S1_BAND_MAXS` | Per-band (VV, VH) clip range in dB |
| `NAN_RATIO_THRESHOLD` | Max allowed fraction of NaN/zero pixels before a file is dropped |
| `S1_PREPROCESSING_STEPS` / `S2_PREPROCESSING_STEPS` | Ordered `(tag_name, step_fn)` lists run by [src/pipeline/preprocessing.py](src/pipeline/preprocessing.py); each `step_fn` is `(data, profile, cfg) -> (data, profile)` from [src/preprocess/operations.py](src/preprocess/operations.py) |
| `GEE_PROJECT` / `GEE_EXPORT_FOLDER` | Your GEE Cloud project, and the Drive folder GEE exports land in |
| `HF_REPO_ID` | Hugging Face dataset repo the finished dataset is pushed to |

Each config also exposes derived `Path` properties for every dataset location
(`NEW_S1_PATH`, `NEW_S2_PATH`, `NEW_MASK_PATH`, `NEW_METADATA_CSV`, `EXPORT_PATH`,
etc.) — use these instead of hardcoding paths.

---

## 7. End-to-end checklist

1. Set up a Google Earth Engine account and Cloud project (§2).
2. Run the **Setup** cell — Drive will mount, and the repo will be cloned if
   `clone_repo=True`.
3. Run **create dataset structure** and **download + extract** STURM-Flood.
4. Set `gee_export=True` and run the **match + export** cell. Re-run later with
   `gee_export=False` while waiting for GEE tasks to finish, then `True` again once
   they're done (or `cancle_gee_tasks=True` to cancel a stuck batch).
5. Run the **assemble + preprocess** loop until `validate_files(cfg)` reports complete.
6. Run **remove bad NaN files**.
7. Run the **inspection** cell as a final sanity check.
8. Set `push_to_HF=True` and run the final cell to publish the dataset.

---

## 8. Key implementation notes

- **Resumable preprocessing**: each step in `S1_PREPROCESSING_STEPS`/
  `S2_PREPROCESSING_STEPS` writes its tag name into the GeoTIFF's `steps` metadata tag
  once applied, so an interrupted run picks up exactly where it left off rather than
  reprocessing from scratch.
- **Crash safety**: preprocessing writes to a `*.tif.tmp` file and only swaps it in
  (`os.replace`) after a full successful write — and any orphaned temp file from a
  prior crash is automatically deleted at the start of the next run, since the original
  `.tif` is guaranteed to still be present whenever a temp file is.
- **GEE exports require Drive**: see §1 — there's no way around mounting Drive, because
  `Export.image.toDrive` is the only export destination GEE offers that this pipeline
  can read back from.
- **Order matters in `S1_PREPROCESSING_STEPS`**: `remove_nana` runs *first*, before the
  Lee filter — the Lee filter is not NaN-safe, and a single real NaN left in the data
  would otherwise poison its global noise-variance estimate and turn an entire band to
  NaN, not just the area around the bad pixel.
- **HF push needs a token**: `push_to_HF=True` reads `HF_TOKEN` from Colab's secrets
  manager (`google.colab.userdata`) — set that up in Colab's secrets panel before
  pushing.
