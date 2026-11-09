# hyperstac_tool (prototype 0.1)

CPU preprocessing for HyPERSTAC (Farndale et al. 2025, bioRxiv 10.1101/2025.10.16.682563)
that accepts our image formats directly. Produces 224 px tiles ready for the
self-supervised training step. Training, embedding and clustering are not wired yet.

Methods guide (paper value vs our value for every step):
`akoya_webinar_0622-main/notebooks/docs/help/hyperstac_reproduction_guide.html`.
Design: `notebooks/Planning/superpowers/specs/2026-11-09-hyperstac-standalone-gui-design.md`.

## Inputs

| Input | Detected by | Channel names from | Pixel size from |
|---|---|---|---|
| Akoya / PerkinElmer QPTIFF | `.qptiff` | `<Biomarker>` in each page's XML description | TIFF XResolution |
| OME-TIFF | `.ome.tif(f)` | OME `Channel/@Name` | OME `PhysicalSizeX` |
| 10x Xenium bundle | folder with `experiment.xenium` | `morphology_focus/chNNNN_<name>.ome.tif` | `experiment.xenium: pixel_size` |
| Phenotyped `.h5ad` | refused | - | - (no pixels; use for interpretation only) |

## Environment

Own venv, never the akoya analysis venv:

```
cd E:\Downloads_6-13-25\HyPERSTAC
C:\Users\thomasc8\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv_tool
.venv_tool\Scripts\python.exe -m pip install tifffile "zarr>=3" imagecodecs scikit-image scipy numpy pandas matplotlib pyyaml
```

## Use

```
.venv_tool\Scripts\python.exe -m hyperstac_tool inspect E:\Tonsil31plex\Tonsil31plex_Scan1.qptiff
bash run_prep_retry.sh E:/Tonsil31plex/Tonsil31plex_Scan1.qptiff --out runs/tonsil31_all32
bash run_prep_retry.sh E:/Tonsil31plex/Tonsil31plex_Scan1.qptiff --out runs/tonsil31_paper8 --channels paper8
```

`--channels`: `all` (default; drops Xenium background and segmentation-stain channels),
`paper8` (DAPI, Pan-Cytokeratin, CD68, SMA, CD4, CD8, Ki67, CD3e: the paper's IO panel
plus CD3e), or a comma-separated list. `--max-rows N` limits tile rows for smoke tests.

`run_prep_retry.sh` reruns after a crash and resumes from verified shards. It stops when
the same error repeats twice. This matters on LMSK56386, which throws sporadic
`SystemError: unknown opcode` faults (crash compendium CD-001 / OS-002).

## What each stage does

| Stage | Paper | This tool |
|---|---|---|
| Tile size | 224 px at 0.497 um/px = 111.3 um | Cut 111.3 um tiles at native resolution (219 px for this QPTIFF, 524 px for Xenium), resize to 224 px |
| Tissue mask | DAPI: downsample x10, dilate 50 px, median 20 px, upsample, Otsu (reference-code order) | Same, with the downsample factor converted to source pixels |
| Tile filter | keep tiles > 50% tissue; background pixels set to 0 | Same |
| Region filter | reject cores with < 20 tiles | Same (`region_passes_min_tiles`) |
| Scaling | not stated; augmentations assume 0-255 | uint8 used as-is; uint16 mapped per channel from [1st, 99.8th] percentile to [0, 255] |

## Outputs (one run folder)

`inspect_report.json`, `channels.csv`, `panel.json`, `scaling_params.json`, `mask_params.json`,
`tissue_mask_small.npy`, `tiles/shard_NNNNN.npy` (N, 224, 224, C) uint8, `tiles/tile_index.csv`
(one row per tile: grid row/col, pixel and um origin, tissue fraction, shard and position),
`tiles_summary.json`, `qc/mask_and_tiles.png`, `qc/tile_montage.png`, `qc/channel_tile_stats.csv`,
`DECISION_LOG.md`, `attempts.log`.

## Not done yet

- Shard loader for `ssl-training.py` (the reference code expects one `.npy` per tile).
- TMA dearraying and core -> patient table.
- Training / embedding / Leiden / interpretation stages; GPU host decision.
- Xenium and OME-TIFF adapters are written but only QPTIFF has been run end to end.
