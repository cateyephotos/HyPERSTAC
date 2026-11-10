"""Fold assignment. The paper splits PATIENTS into two folds (Methods 4.3.2, 4.5.3).

- patient mode: needs a patient column (multi-sample cohorts) - paper-faithful.
- stripes mode: a single slide has one patient, so the patient split does not exist.
  For demonstrations we alternate vertical stripes of tiles (default 10 tiles = 1.1 mm)
  between two groups. Neighboring tiles inside a stripe stay together, which limits
  leakage between train and held-out tiles, but this is NOT a patient-level evaluation
  and must be labeled as a demonstration.

Group convention follows ssl-training.py: fold k trains on group != k and holds out group == k.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_folds(run_dir: Path, mode: str = "stripes", stripe_tiles: int = 10,
               patient_col: str = "patient_id", n_folds: int = 2) -> pd.DataFrame:
    idx = pd.read_csv(run_dir / "tiles" / "tile_index.csv")
    if mode == "stripes":
        group = (idx["grid_col"] // stripe_tiles) % n_folds
        note = f"single-slide demo: vertical stripes of {stripe_tiles} tiles alternate between groups"
    elif mode == "patient":
        if patient_col not in idx.columns:
            raise KeyError(f"tile_index.csv has no {patient_col!r}; add a patient map first")
        pats = sorted(idx[patient_col].astype(str).unique())
        order = {p: i % n_folds for i, p in enumerate(pats)}
        group = idx[patient_col].astype(str).map(order)
        note = f"patient-level split over {len(pats)} patients (paper mode)"
    else:
        raise ValueError(mode)
    out = pd.DataFrame({"tile_id": range(len(idx)), "group": group.astype(int)})
    out.to_csv(run_dir / "folds.csv", index=False)
    (run_dir / "folds_note.txt").write_text(note + "\n")
    return out
