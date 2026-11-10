"""VICReg training on prep tiles - runs in WSL2 with GPU TensorFlow.

    source /mnt/e/Downloads_6-13-25/HyPERSTAC/wsl_env.sh
    cd /mnt/e/Downloads_6-13-25/HyPERSTAC
    python -m hyperstac_tool.train_vicreg --run runs/tonsil31_all32 --fold 0

Reuses the reference VICReg loss class, augmentations and warmup-cosine schedule
unchanged (VICReg/ in this repo), under the legacy Keras 2 API (tf_keras) they were
written for. Paper settings (Methods 4.3): ResNet-50 from scratch, 2048-d
representation, 3 x 8192 projector, 100 epochs, warmup-cosine peak 1e-4 with 10-epoch
warmup from 0, Adam (clipnorm 1 as in vicreg_utils.create_adam_opt), VICReg
lambda=25 mu=25 nu=1.

Deviations recorded in train_params.json:
  - batch size (paper 256 on an 80 GB H100; set by --batch-size for a 16 GB GPU)
  - encoder from tf.keras.applications instead of the `image-classifiers` package
    (same ResNet-50 v1 topology; that package predates Keras 3)
  - validation loss on a fixed random subset of held-out tiles (--val-tiles)
Resumes from the last completed epoch (optimizer state included).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must precede the TensorFlow import

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # replaces the reference code's sys.path.insert(0, '/home/')

import tensorflow as tf  # noqa: E402

from VICReg.augmentations import custom_augment_multiplex  # noqa: E402
from VICReg.ssl_base import VICReg  # noqa: E402
from VICReg.vicreg_utils import create_adam_opt  # noqa: E402
from VICReg.warmupcosine import WarmUpCosine  # noqa: E402

from .folds import make_folds  # noqa: E402


def _shard_dir(run: Path, cache: Path | None) -> Path:
    """Shards on /mnt/<drive> read slowly from WSL (9p). Optionally copy them once to a
    Linux-native cache folder; a cached shard is reused only if its size matches."""
    src = run / "tiles"
    if cache is None:
        return src
    dst = cache / run.name / "tiles"
    dst.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in sorted(src.glob("shard_*.npy")):
        if ".tmp" in f.name:
            continue
        g = dst / f.name
        if not g.exists() or g.stat().st_size != f.stat().st_size:
            shutil.copyfile(f, g.with_suffix(".part"))
            os.replace(g.with_suffix(".part"), g)
    return dst


def load_tiles(run: Path, tile_ids: np.ndarray, cache: Path | None = None) -> np.ndarray:
    """Load the requested tiles from the shards into one RAM array (N, 224, 224, C) uint8."""
    idx = pd.read_csv(run / "tiles" / "tile_index.csv")
    sel = idx.iloc[np.sort(tile_ids)]
    sdir = _shard_dir(run, cache)
    out, n = None, 0
    for shard, g in sel.groupby("shard", sort=True):
        a = np.load(sdir / f"shard_{int(shard):05d}.npy", mmap_mode="r")
        block = np.asarray(a[g["shard_pos"].to_numpy()])
        if out is None:
            out = np.empty((len(sel),) + block.shape[1:], dtype=np.uint8)
        out[n:n + len(block)] = block
        n += len(block)
        del a
    return out


def make_ds(arr: np.ndarray, batch: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    shape = arr.shape[1:]
    ds = tf.data.Dataset.range(len(arr))
    if shuffle:
        ds = ds.shuffle(len(arr), seed=seed, reshuffle_each_iteration=True)

    def get(i):
        return arr[i]

    def load(i):
        x = tf.numpy_function(get, [i], tf.uint8)
        x.set_shape(shape)
        return tf.cast(x, tf.float32)

    aug = lambda x: custom_augment_multiplex(x, input_shape=shape, output_shape=shape)  # noqa: E731
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (aug(x), aug(x)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def create_projector(input_shape, output_size=8192):
    """As in ssl-training.py: GAP -> 3 x Dense(8192), BN + ReLU on the first two, linear output."""
    p = tf.keras.models.Sequential()
    p.add(tf.keras.Input(shape=input_shape))
    p.add(tf.keras.layers.GlobalAveragePooling2D())
    for _ in range(2):
        p.add(tf.keras.layers.Dense(output_size))
        p.add(tf.keras.layers.BatchNormalization())
        p.add(tf.keras.layers.Activation("relu"))
    p.add(tf.keras.layers.Dense(output_size))
    return p


class EpochCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, manager, progress_path, log_path):
        super().__init__()
        self.manager, self.progress_path, self.log_path = manager, progress_path, log_path
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.manager.save(checkpoint_number=epoch + 1)
        row = {"epoch": epoch + 1, "seconds": round(time.time() - self.t0, 1),
               **{k: float(v) for k, v in (logs or {}).items()}}
        self.t0 = time.time()
        pd.DataFrame([row]).to_csv(self.log_path, mode="a", index=False,
                                   header=not self.log_path.exists())
        self.progress_path.write_text(json.dumps(row))
        print(f"[epoch {epoch + 1}] " + " ".join(f"{k}={v:.4g}" for k, v in row.items() if k != "epoch"),
              flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-fraction", type=float, default=0.1)
    ap.add_argument("--projector-size", type=int, default=8192)
    ap.add_argument("--val-tiles", type=int, default=512)
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="timing tests only")
    ap.add_argument("--fold-mode", default="stripes", choices=["stripes", "patient"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile-cache", default=None, help="Linux-native folder to cache shards (WSL)")
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. _timing)")
    args = ap.parse_args(argv)

    run = Path(args.run)
    tf.keras.utils.set_random_seed(args.seed)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    folds = pd.read_csv(run / "folds.csv") if (run / "folds.csv").exists() else make_folds(run, args.fold_mode)
    train_ids = folds.loc[folds.group != args.fold, "tile_id"].to_numpy()
    held_ids = folds.loc[folds.group == args.fold, "tile_id"].to_numpy()
    rng = np.random.default_rng(args.seed)
    val_ids = rng.choice(held_ids, size=min(args.val_tiles, len(held_ids)), replace=False)

    t0 = time.time()
    cache = Path(args.tile_cache).expanduser() if args.tile_cache else None
    x_train = load_tiles(run, train_ids, cache)
    x_val = load_tiles(run, val_ids, cache)
    print(f"[data] train {x_train.shape} val {x_val.shape} loaded in {time.time() - t0:.0f}s", flush=True)

    shape = x_train.shape[1:]
    steps = args.steps_per_epoch or len(x_train) // args.batch_size
    total = args.epochs * steps
    warm = int(args.epochs * args.warmup_fraction * steps)

    out = run / "ssl" / f"fold-{args.fold}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    params = {
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "fold": args.fold, "fold_mode": args.fold_mode,
        "n_train_tiles": int(len(x_train)), "n_heldout_tiles": int(len(held_ids)), "n_val_tiles": int(len(x_val)),
        "input_shape": list(shape), "encoder": "tf.keras.applications.ResNet50(weights=None, include_top=False)",
        "representation_dim": 2048, "projector": [args.projector_size] * 3,
        "loss": "VICReg (reference ssl_base.VICReg defaults: lambda=25, mu=25, nu=1, gamma=1)",
        "epochs": args.epochs, "batch_size": args.batch_size, "paper_batch_size": 256,
        "steps_per_epoch": steps, "lr_peak": args.lr, "warmup_steps": warm, "optimizer": "Adam, clipnorm=1",
        "augmentations": "reference custom_augment_multiplex (flips, crop 0.75-1, background p0.3, "
                         "rotation p0.4, solarize p0.3, gaussian noise)",
        "tensorflow": tf.__version__, "seed": args.seed,
    }
    (out / "train_params.json").write_text(json.dumps(params, indent=2))

    encoder = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=shape)
    projector = create_projector(encoder.output_shape[1:], args.projector_size)
    sched = WarmUpCosine(learning_rate_base=args.lr, total_steps=total, warmup_learning_rate=0.0,
                         warmup_steps=warm, start_step=0)
    model = VICReg(encoder_list=[encoder], projector_list=[projector],
                   encoder_indices=[0, 0], projector_indices=[0, 0])
    model.compile(optimizer=create_adam_opt(sched))

    ckpt = tf.train.Checkpoint(encoder=encoder, projector=projector, optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(ckpt, str(out / "ckpt"), max_to_keep=2)
    initial_epoch = 0
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        initial_epoch = int(manager.latest_checkpoint.rsplit("-", 1)[-1])
        print(f"[resume] from epoch {initial_epoch}", flush=True)
    if initial_epoch >= args.epochs:
        print("[train] already complete")
    else:
        ds_train = make_ds(x_train, args.batch_size, True, args.seed)
        if args.steps_per_epoch:
            ds_train = ds_train.repeat()
        ds_val = make_ds(x_val, args.batch_size, False, args.seed)
        model.fit(ds_train, epochs=args.epochs, initial_epoch=initial_epoch, steps_per_epoch=steps,
                  validation_data=ds_val, verbose=2,
                  callbacks=[EpochCheckpoint(manager, out / "progress.json", out / "training_log.csv")])

    encoder.save_weights(str(out / "encoder_weights_0.weights.h5"))
    projector.save_weights(str(out / "projector_weights_0.weights.h5"))
    (out / "DONE").write_text(dt.datetime.now().isoformat(timespec="seconds"))
    print(f"[train] saved encoder to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
