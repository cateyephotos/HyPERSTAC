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

## GPU stages (WSL2)

Training, embedding and clustering run in WSL2 with GPU TensorFlow 2.21 and the legacy
Keras 2 API (`tf_keras`), which the reference `VICReg/` code was written for.

One-time setup inside WSL (no sudo needed):

```
python3 -m venv --without-pip ~/hyperstac_tf
curl -sSLo /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && ~/hyperstac_tf/bin/python /tmp/get-pip.py
~/hyperstac_tf/bin/python -m pip install "tensorflow[and-cuda]" tf-keras pandas scanpy leidenalg igraph h5py matplotlib scikit-image
```

`wsl_env.sh` puts the pip-installed NVIDIA libraries on `LD_LIBRARY_PATH`. Without it,
TensorFlow 2.21 silently runs on CPU (it cannot dlopen `libcusolver.so.11`).

```
# single-slide demo: both channel sets, both folds, detached, crash-resuming
python -m hyperstac_tool subset runs/tonsil31_all32 --out runs/tonsil31_paper8 --channels paper8   # Windows venv
setsid nohup bash run_train_queue.sh runs/tonsil31_paper8:0 runs/tonsil31_all32:0 runs/tonsil31_paper8:1 runs/tonsil31_all32:1 > runs/train_queue.log 2>&1 &
setsid nohup bash run_post.sh runs/tonsil31_paper8 runs/tonsil31_all32 > runs/post.log 2>&1 &
```

| Module | What it does | Outputs (`<run>/ssl/fold-<k>/`) |
|---|---|---|
| `folds.py` | Paper splits patients 2-fold. One slide has one patient, so `stripes` mode alternates 10-tile (1.1 mm) vertical stripes. Demonstration only. | `<run>/folds.csv`, `folds_note.txt` |
| `train_vicreg.py` | Reference `VICReg` loss, `custom_augment_multiplex`, `WarmUpCosine` unchanged; ResNet-50 from scratch; checkpoint every epoch, resumes with optimizer state | `train_params.json`, `training_log.csv`, `ckpt/`, `encoder_weights_0.weights.h5`, `DONE` |
| `embed.py` | Encoder + global average pooling -> 2048-d per tile | `embeddings.npz`, `embeddings_info.json` |
| `cluster.py` | PCA 50 -> kNN graph -> Leiden (res 1) on training tiles; `ingest` for held-out tiles; marker intensity per SPC; near-blank channel check; figures | `spc/tile_spc.csv`, `spc_sizes.csv`, `spc_marker_intensity(_z).csv`, `channel_coverage.csv`, `spc_qc.json` (incl. `near_blank_channels`), `umap_spc.png`, `spatial_spc_map.png` (single panel, legend), `fold_stripes_map.png`, `spc_marker_heatmap.png` (near-blank channels left out), `spc_gallery.png` |

Training also writes `training_summary.json` (training and validation loss first / last / min,
and `validation_improved`), and the queue writes `wall_time.json` (attempts, total minutes
across every attempt). Present both losses, and the total time, never the last attempt's time.

**Figure conventions (fixed 2026-11-11 after the tonsil deck review):** every figure says
"Cluster n" (never "SPC n"); the slide map has a legend; a channel above zero in fewer than 5%
of tissue pixels is listed as near blank and left out of the heatmap, because cluster z-scores
make empty channels look enriched (CD21 on the tonsil: means 0.0-0.1 of 255, z = 2.7).

### Deviations from the paper (recorded in `train_params.json` / `spc_qc.json`)

- **Batch size 48** (paper 256). Batch 64 runs out of memory on the 16 GB RTX 5000 Ada:
  the 3 x 8192 projector and its 8192 x 8192 covariance terms dominate.
- **Encoder** from `tf.keras.applications.ResNet50(weights=None)` rather than the
  `image-classifiers` package (same ResNet-50 v1 topology; that package predates Keras 3).
- **Validation loss** on 512 random held-out tiles, not all of them.
- **Leiden** from scanpy/igraph instead of rapids_singlecell on GPU.
- **Held-out assignment**: `ingest`, as in the reference. On synthetic, perfectly separable
  embeddings it mislabeled 7.5% of held-out points while an exact 15-NN vote in the same PCA
  space was 100% correct. Both are saved (`spc`, `spc_exact_knn`) and their agreement is in
  `spc_qc.json` as `heldout_ingest_vs_exact_knn_agreement`.
- **Interpretation** by marker intensity per SPC (the paper's Hallmarks approach); a bare
  QPTIFF has no cell segmentation for cell-type densities.

### Known issues

- In WSL, TensorFlow 2.21 aborts (`free(): double free`) whenever the GPU is hidden
  (`CUDA_VISIBLE_DEVICES=""`, `-1`, or an invalid index). Embedding/clustering cannot run on
  CPU while training holds the GPU; `run_post.sh` waits for the training queue instead.
- Fixed 2026-11-10: `run_train_queue.sh` used to match TensorFlow's harmless oneDNN banner
  ("round-off errors") as its "identical error", so two crashes of any kind could stop the
  queue. Do not edit a queue script while it runs: bash reads scripts incrementally.
- Results of the first full run: `DEMO_TONSIL31.md`.
- Speed: about 50 s per epoch at batch 48 (2,764 training tiles); CPU-side augmentation is
  the bottleneck (GPU mostly idle). 100 epochs is about 85 minutes per encoder.

## Not done yet

- TMA dearraying and core -> patient table; `patient` fold mode needs a patient column.
- Xenium and OME-TIFF adapters are written; only QPTIFF has been run end to end.
- Cell-segmentation import (QuPath / inForm / Xenium cells) for cell-type density per SPC.
