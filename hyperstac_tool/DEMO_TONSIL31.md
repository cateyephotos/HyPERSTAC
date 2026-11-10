# Demo: HyPERSTAC on one Akoya QPTIFF (Tonsil31plex), 2026-11-09/10

**What this is:** a check that the full workflow runs on a multichannel QPTIFF and that its
clusters look like known tonsil anatomy. **What it is not:** an evaluation of the method.
One slide is one patient, so the paper's patient-level 2-fold split was replaced with
alternating 1.1 mm vertical stripes of tiles, and there are no outcomes.

## Input and runs

- `E:\Tonsil31plex\Tonsil31plex_Scan1.qptiff`: 32 channels, uint8, 0.507 um/px, 12.8 x 7.8 mm.
- 5,526 tiles of 111 um (219 px native -> 224 px), tissue > 50%.
- Two channel sets: `paper8` (DAPI, Pan-Cytokeratin, CD68, SMA, CD4, CD8, Ki67, CD3e; the
  paper's IO panel + CD3e) and `all32`. Pixels of `paper8` are an exact slice of `all32`.
- Four VICReg encoders (2 channel sets x 2 folds), 100 epochs each, batch 48, RTX 5000 Ada
  16 GB, ~50 s/epoch. Training loss fell from ~216-231 to ~38.8-38.9 in every run.
- Results: `runs/tonsil31_<set>/ssl/fold-<k>/spc/` (not in git; regenerate with the commands in README.md).

## QC

| Run | SPCs | SPCs receiving held-out tiles | Top-3 share train / held-out | ingest vs exact 15-NN |
|---|---|---|---|---|
| paper8 fold 0 | 24 | 24 | 0.21 / 0.23 | 0.975 |
| paper8 fold 1 | 24 | 24 | 0.23 / 0.25 | 0.971 |
| all32 fold 0 | 22 | 22 | 0.23 / 0.24 | 0.960 |
| all32 fold 1 | 21 | 21 | 0.26 / 0.31 | 0.955 |

No held-out collapse (the DINOv2 failure mode in the paper). Agreement between encoders
(`runs/tonsil31_demo_agreement.csv`): same channels, different folds AMI 0.53-0.59 (ARI 0.31-0.33);
8 vs 32 channels AMI 0.45-0.47 (ARI 0.21-0.25).

## Does it find tonsil anatomy? (all32 fold 0, by marker heatmap and tile gallery)

| SPC | Reads as | Evidence |
|---|---|---|
| 0 | B-cell follicle | dense CD20+ tiles; HLA-DR, PD-1 high |
| 15 | germinal-center-like | CD20, CD21, PD-1, ICOS, Ki67 high |
| 21 | T-cell zone | dense CD3e+ tiles; CD4, CD8, ICOS, TCF-1 high |
| 1, 13 | epithelium with neutrophils | Pan-Cytokeratin with CD66, MPO; HIF1A, PD-L1, IDO1 |
| 19, 20 | stroma / vessels | SMA, Collagen IV, CD31 |
| 5, 14 | sparse tissue / edges | low in every channel |

Spatially the SPCs form coherent domains (`spatial_spc_map.png`), not salt-and-pepper noise.
This is one reviewer's read of summary figures, not a pathologist annotation.

## Caveat that must be resolved before reading biology into SPCs

SPC membership explains 69-84% of the variance in tile brightness (mean over channels) and
in DAPI intensity (eta squared; shuffled labels 0.2-0.5%;
`runs/tonsil31_demo_brightness_eta2.json`). Part of this is real (follicles are denser than
stroma), but the slide also has a top-to-bottom brightness gradient. The paper checked that
its SPCs did not simply sort by marker intensity; that check has not been done here.
Next tests: (1) compare SPC composition within matched anatomy between the top and bottom
halves of the slide; (2) rerun with per-tile or per-region intensity normalization and see
which clusters survive (a deviation from the paper that needs the defensibility gate).

CD21 and CD31 are nonzero in only ~2-3% of tile pixels on this slide, so z-scores for those
two channels are driven by very small signals.

## Machine notes

Native crashes during the run (this workstation's known instability): 1 startup segfault on
paper8 fold 0, 4 failed attempts on paper8 fold 1 (segfault, abort, two
`SystemError: error return without exception set`), and 3 clustering failures (a numba
internal TypeError and two segfaults). The retry loops completed every job; nothing had to
be redone by hand.
