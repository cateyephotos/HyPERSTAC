"""Embed every tile with a trained fold encoder - runs in WSL2 with GPU TensorFlow.

    python -m hyperstac_tool.embed --run runs/tonsil31_paper8 --fold 0

As in save-representations.py: encoder output (7, 7, 2048) -> global average pooling
-> 2048-d representation per tile. No augmentation. Writes
<run>/ssl/fold-<k>/embeddings.npz with arrays features (N, 2048) float32, tile_id (N,),
group (N,), split (N,) ('train' or 'heldout', per the fold convention in folds.py).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .train_vicreg import load_tiles


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix of the ssl/fold-<k> folder")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tile-cache", default="~/hyperstac_cache")
    args = ap.parse_args(argv)

    run = Path(args.run)
    out = run / "ssl" / f"fold-{args.fold}{args.tag}"
    if not (out / "DONE").exists():
        raise FileNotFoundError(f"{out} has no finished encoder (DONE missing)")
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    folds = pd.read_csv(run / "folds.csv").sort_values("tile_id")
    ids = folds["tile_id"].to_numpy()
    t0 = time.time()
    x = load_tiles(run, ids, Path(args.tile_cache).expanduser() if args.tile_cache else None)
    shape = x.shape[1:]

    encoder = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=shape)
    encoder.load_weights(str(out / "encoder_weights_0.weights.h5"))
    model = tf.keras.Sequential([encoder, tf.keras.layers.GlobalAveragePooling2D()])

    feats = np.empty((len(x), 2048), dtype=np.float32)
    for s in range(0, len(x), args.batch_size):
        feats[s:s + args.batch_size] = model(x[s:s + args.batch_size].astype(np.float32), training=False).numpy()
    split = np.where(folds["group"].to_numpy() == args.fold, "heldout", "train")
    tmp = out / "embeddings.tmp.npz"
    np.savez(tmp, features=feats, tile_id=ids, group=folds["group"].to_numpy(), split=split)
    os.replace(tmp, out / "embeddings.npz")
    info = {"n_tiles": int(len(ids)), "dim": 2048, "pooling": "global average",
            "n_train": int((split == "train").sum()), "n_heldout": int((split == "heldout").sum()),
            "seconds": round(time.time() - t0, 1),
            "any_nan": bool(np.isnan(feats).any()), "feature_std_mean": float(feats.std(axis=0).mean())}
    (out / "embeddings_info.json").write_text(json.dumps(info, indent=2))
    print(f"[embed] {info}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
