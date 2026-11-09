"""CPU stages of the HyPERSTAC workflow: inspect -> panel -> scaling -> tissue mask -> tiles -> QC.

Paper defaults (Farndale et al. 2025, bioRxiv 10.1101/2025.10.16.682563, Methods 4.3.1)
are expressed in pixels at 0.497 um/px. They are converted to physical units here and
back to pixels at the source resolution, so a QPTIFF at 0.507 um/px and a Xenium image
at 0.2125 um/px get the same physical tile (111.3 um) and the same mask geometry.

Every stage writes its table and a params JSON before the next stage starts, and the
run appends to DECISION_LOG.md (provenance rule in the akoya batch_processing CLAUDE.md).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.transform import resize

from .sources import ImageSource

PAPER = {
    "pixel_size_um": 0.497,
    "tile_px": 224,                 # network input; 224 px * 0.497 = 111.3 um
    "dilate_kernel_px": 50,         # reference code: applied AFTER the x10 downsample
    "downsample": 10,
    "median_px": 20,                # applied on the downsampled image
    "min_tissue_fraction": 0.5,     # tile kept when tissue > 50% (code: strictly greater)
    "min_tiles": 20,                # region rejected below this (code: kept only if > 20)
}
TILE_UM = PAPER["tile_px"] * PAPER["pixel_size_um"]


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _npy_shape(path: Path) -> tuple:
    """Shape from the .npy header, plus a file-size check. No memory map, so Windows
    does not keep the file locked (os.replace on a mapped file fails with WinError 32)."""
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, _, dtype = reader(fh)
        header_end = fh.tell()
    expected = header_end + int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise IOError(f"{path.name}: size {path.stat().st_size} != expected {expected}")
    return tuple(shape)


def log_decision(run_dir: Path, step: str, inputs, outputs, note: str = "") -> None:
    with open(run_dir / "DECISION_LOG.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n## {_now()} - {step}\n- inputs: {inputs}\n- outputs: {outputs}\n")
        if note:
            fh.write(f"- note: {note}\n")


# ---------------------------------------------------------------------------
# Stage 0: inspect
def stage_inspect(src: ImageSource, run_dir: Path) -> dict:
    rep = src.summary()
    rep["paper_tile_um"] = round(TILE_UM, 3)
    rep["native_tile_px"] = int(round(TILE_UM / src.pixel_size_um))
    rep["inspected_at"] = _now()
    _write_json(run_dir / "inspect_report.json", rep)
    pd.DataFrame(src.channels).to_csv(run_dir / "channels.csv", index=False)
    log_decision(run_dir, "inspect", str(src.path), ["inspect_report.json", "channels.csv"],
                 f"{src.kind}, {len(src.channels)} channels, {src.pixel_size_um:.4f} um/px, "
                 f"{src.shape_yx} px, dtype {src.dtype}")
    return rep


# ---------------------------------------------------------------------------
# Stage 1: panel (which channels go to the encoder)
def stage_panel(src: ImageSource, run_dir: Path, nuclear: str, channels: list[str] | None,
                drop_roles=("background", "segmentation")) -> dict:
    nuc_idx = src.channel_index(nuclear)
    if channels:
        idx = [src.channel_index(c) for c in channels]
        if nuc_idx not in idx:
            idx = [nuc_idx] + idx
    else:
        idx = [c["index"] for c in src.channels if c["role"] not in drop_roles]
    panel = {
        "nuclear_channel": src.channels[nuc_idx]["name"],
        "nuclear_index": nuc_idx,
        "encoder_channels": [src.channels[i]["name"] for i in idx],
        "encoder_indices": idx,
        "n_encoder_channels": len(idx),
        "dropped": [c["name"] for c in src.channels if c["index"] not in idx],
        "paper_n_channels": 8,
    }
    _write_json(run_dir / "panel.json", panel)
    log_decision(run_dir, "panel", "channels.csv", "panel.json",
                 f"{len(idx)} encoder channels (paper used 8). Dropped: {panel['dropped'] or 'none'}")
    return panel


# ---------------------------------------------------------------------------
# Stage 2: intensity scaling to the 0-255 range the reference augmentations assume
def stage_scaling(src: ImageSource, run_dir: Path, panel: dict, mode: str = "auto",
                  lo_pct: float = 1.0, hi_pct: float = 99.8) -> dict:
    if mode == "auto":
        mode = "none" if src.dtype == "uint8" else "percentile"
    params = {"mode": mode, "source_dtype": src.dtype, "channels": {}}
    if mode == "percentile":
        lvl = min(range(len(src.level_shapes)), key=lambda l: abs(src.level_shapes[l][2] - 2048))
        sub = src.read(lvl, 0, src.level_shapes[lvl][1], 0, src.level_shapes[lvl][2],
                       panel["encoder_indices"]).astype(np.float32)
        for k, name in enumerate(panel["encoder_channels"]):
            v = sub[k][sub[k] > 0]
            lo, hi = (np.percentile(v, [lo_pct, hi_pct]) if v.size else (0.0, 1.0))
            params["channels"][name] = {"lo": float(lo), "hi": float(max(hi, lo + 1))}
        params.update({"fit_level": lvl, "lo_pct": lo_pct, "hi_pct": hi_pct,
                       "fit_scope": "this sample only - multi-sample runs must refit on training patients"})
    elif mode == "none" and src.dtype != "uint8":
        raise ValueError("scaling 'none' is only valid for uint8 sources")
    _write_json(run_dir / "scaling_params.json", params)
    note = ("uint8 source used as-is (already on the 0-255 scale the reference augmentations assume)"
            if mode == "none" else
            f"per-channel [{lo_pct}, {hi_pct}] percentile -> [0, 255]; paper does not state its scaling")
    log_decision(run_dir, "scaling", "panel.json", "scaling_params.json", note)
    return params


def apply_scaling(block: np.ndarray, params: dict, names: list[str]) -> np.ndarray:
    if params["mode"] == "none":
        return block.astype(np.uint8, copy=False)
    out = np.empty(block.shape, dtype=np.uint8)
    for k, n in enumerate(names):
        lo, hi = params["channels"][n]["lo"], params["channels"][n]["hi"]
        out[k] = np.clip((block[k].astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Stage 3: tissue mask (reference-code order: downsample -> dilate -> median -> Otsu)
def stage_mask(src: ImageSource, run_dir: Path, panel: dict) -> dict:
    scale = PAPER["pixel_size_um"] / src.pixel_size_um          # source px per paper px
    ds = PAPER["downsample"] * scale                             # downsample in source px
    lvl = max([l for l in range(len(src.level_shapes)) if src.level_downsample(l) <= ds] or [0])
    H, W = src.shape_yx
    small_shape = (max(1, int(round(H / ds))), max(1, int(round(W / ds))))
    lv = src.level_shapes[lvl]
    dapi = src.read(lvl, 0, lv[1], 0, lv[2], [panel["nuclear_index"]])[0].astype(np.float32)
    small = resize(dapi, small_shape, order=1, anti_aliasing=True, preserve_range=True)
    k_dil = max(1, int(round(PAPER["dilate_kernel_px"])))   # kernel lives at the downsampled scale
    k_med = max(1, int(round(PAPER["median_px"])))
    dil = ndi.grey_dilation(small, size=(k_dil, k_dil))
    filt = ndi.median_filter(dil, size=k_med)
    thr = float(threshold_otsu(filt))
    mask_small = filt > thr
    np.save(run_dir / "tissue_mask_small.npy", filt.astype(np.float32))
    params = {
        "order": "reference code: downsample, dilate, median filter, Otsu (paper text lists dilate first)",
        "source_pixel_size_um": src.pixel_size_um, "downsample_source_px": ds, "read_level": lvl,
        "small_shape": list(small_shape), "dilate_kernel_small_px": k_dil, "median_small_px": k_med,
        "otsu_threshold": thr, "tissue_fraction_small": float(mask_small.mean()),
    }
    _write_json(run_dir / "mask_params.json", params)
    log_decision(run_dir, "tissue mask", "nuclear channel", ["mask_params.json", "tissue_mask_small.npy"],
                 f"tissue covers {params['tissue_fraction_small']:.1%} of the image; Otsu {thr:.1f}")
    return {"params": params, "filtered_small": filt, "threshold": thr}


def _band_mask(filt_small: np.ndarray, thr: float, y0: int, y1: int, W: int, H: int) -> np.ndarray:
    """Bilinear-upsample the filtered small DAPI to full resolution for rows y0:y1, then threshold."""
    sh, sw = filt_small.shape
    ys = (np.arange(y0, y1) + 0.5) * sh / H - 0.5
    xs = (np.arange(W) + 0.5) * sw / W - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    vals = ndi.map_coordinates(filt_small, [yy.ravel(), xx.ravel()], order=1, mode="nearest")
    return (vals.reshape(len(ys), W) > thr)


# ---------------------------------------------------------------------------
# Stage 4: tiles, written in resumable shards
def stage_tiles(src: ImageSource, run_dir: Path, panel: dict, scaling: dict, mask: dict,
                shard_size: int = 256, max_rows: int | None = None) -> dict:
    tdir = run_dir / "tiles"
    tdir.mkdir(exist_ok=True)
    H, W = src.shape_yx
    n = int(round(TILE_UM / src.pixel_size_um))           # native tile size, same physical width
    out_px = PAPER["tile_px"]
    names, idx = panel["encoder_channels"], panel["encoder_indices"]
    n_rows, n_cols = H // n, W // n
    if max_rows:
        n_rows = min(n_rows, max_rows)

    index_path = tdir / "tile_index.csv"
    # Resume: shards only ever hold whole grid rows (flushed at row ends), so every row
    # found in a shard that passes readback is complete. Anything else is redone.
    done = {}
    if index_path.exists():
        prev = pd.read_csv(index_path)
        for shard, g in prev.groupby("shard"):
            f = tdir / f"shard_{shard:05d}.npy"
            try:
                if _npy_shape(f) == (len(g), out_px, out_px, len(idx)):
                    done[shard] = g
            except Exception:
                pass
        # keep only a contiguous prefix of shards so shard ids stay consistent
        keep, k = {}, 0
        while k in done:
            keep[k] = done[k]; k += 1
        done = keep
    records = pd.concat(done.values()) if done else pd.DataFrame()
    rows_done = set(records["grid_row"].unique().tolist()) if not records.empty else set()
    last_done_row = max(rows_done) if rows_done else -1
    rec_list = records.to_dict("records") if not records.empty else []
    shard_id = (max(done) + 1) if done else 0
    buf, buf_rec = [], []
    n_grid = n_rejected = 0

    def flush():
        nonlocal shard_id, buf, buf_rec
        if not buf:
            return
        arr = np.stack(buf)
        f = tdir / f"shard_{shard_id:05d}.npy"
        tmp = tdir / f"shard_{shard_id:05d}.tmp.npy"
        np.save(tmp, arr)
        if _npy_shape(tmp) != arr.shape:
            raise IOError(f"shard {shard_id} failed readback")
        os.replace(tmp, f)
        for k, r in enumerate(buf_rec):
            r.update(shard=shard_id, shard_pos=k)
        rec_list.extend(buf_rec)
        pd.DataFrame(rec_list).to_csv(index_path, index=False)
        shard_id += 1
        buf, buf_rec = [], []

    for r in range(n_rows):
        if r <= last_done_row:
            continue
        y0, y1 = r * n, (r + 1) * n
        bmask = _band_mask(mask["filtered_small"], mask["threshold"], y0, y1, W, H)
        frac = bmask[:, : n_cols * n].reshape(n, n_cols, n).mean(axis=(0, 2))
        keep = np.where(frac > PAPER["min_tissue_fraction"])[0]
        n_grid += n_cols
        n_rejected += n_cols - len(keep)
        if len(keep) == 0:
            continue
        band = src.read(0, y0, y1, 0, n_cols * n, idx)            # (C, n, W')
        band = apply_scaling(band, scaling, names)
        band = band * bmask[None, :, : n_cols * n]                 # zero background, as in the reference code
        for c in keep:
            tile = band[:, :, c * n:(c + 1) * n].transpose(1, 2, 0)  # (n, n, C), channels last
            if n != out_px:
                tile = resize(tile, (out_px, out_px, tile.shape[2]), order=1,
                              anti_aliasing=n > out_px, preserve_range=True).round().astype(np.uint8)
            buf.append(tile)
            buf_rec.append({
                "sample_id": src.sample_id, "grid_row": r, "grid_col": int(c),
                "y0_px": y0, "x0_px": int(c * n), "size_px": n,
                "y0_um": round(y0 * src.pixel_size_um, 2), "x0_um": round(c * n * src.pixel_size_um, 2),
                "tissue_fraction": round(float(frac[c]), 4),
            })
        if len(buf) >= shard_size:
            flush()                                               # only at row ends
    flush()

    n_tiles = len(rec_list)
    summary = {
        "native_tile_px": n, "output_tile_px": out_px, "tile_um": round(n * src.pixel_size_um, 2),
        "grid": [n_rows, n_cols], "n_tiles_kept": n_tiles, "n_shards": shard_id,
        "region_passes_min_tiles": n_tiles > PAPER["min_tiles"],
        "tile_dtype": "uint8", "tile_layout": "(N, 224, 224, C) channels-last, background = 0",
        "resumed_rows": len(rows_done),
    }
    _write_json(run_dir / "tiles_summary.json", summary)
    log_decision(run_dir, "tiles", "panel/scaling/mask", ["tiles/shard_*.npy", "tiles/tile_index.csv",
                 "tiles_summary.json"],
                 f"{n_tiles} tiles of {n}px ({summary['tile_um']} um) resized to {out_px}px; "
                 f"region {'passes' if summary['region_passes_min_tiles'] else 'FAILS'} the >20-tile rule")
    return summary


# ---------------------------------------------------------------------------
# Stage 5: QC figures and per-channel tile statistics
def stage_qc(src: ImageSource, run_dir: Path, panel: dict, mask: dict, n_montage: int = 24,
             rgb: tuple[str, str, str] | None = None, seed: int = 0) -> dict:
    import matplotlib.pyplot as plt  # MPLBACKEND=Agg is set by the CLI

    qdir = run_dir / "qc"
    qdir.mkdir(exist_ok=True)
    idx = pd.read_csv(run_dir / "tiles" / "tile_index.csv")
    names = panel["encoder_channels"]

    # 1. mask + kept tiles over the downsampled nuclear image
    filt = mask["filtered_small"]
    H, W = src.shape_yx
    sh, sw = filt.shape
    lvl = mask["params"]["read_level"]
    lv = src.level_shapes[lvl]
    nuc = src.read(lvl, 0, lv[1], 0, lv[2], [panel["nuclear_index"]])[0].astype(np.float32)
    nuc = resize(nuc, (sh, sw), anti_aliasing=True, preserve_range=True)
    fig, ax = plt.subplots(figsize=(8, 8 * sh / sw))
    ax.imshow(nuc, cmap="gray", vmax=np.percentile(nuc, 99.5))
    ax.contour(filt > mask["threshold"], levels=[0.5], colors="yellow", linewidths=0.8)
    f = sw / W
    for _, r in idx.iterrows():
        ax.add_patch(plt.Rectangle((r.x0_px * f, r.y0_px * f), r.size_px * f, r.size_px * f,
                                   fill=False, edgecolor="cyan", linewidth=0.2))
    ax.set_title(f"{src.sample_id}: tissue mask (yellow) and {len(idx)} kept tiles (cyan)")
    ax.axis("off")
    fig.savefig(qdir / "mask_and_tiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. per-channel statistics over kept tiles (streamed shard by shard)
    sums = np.zeros(len(names)); sq = np.zeros(len(names)); sat = np.zeros(len(names)); nz = np.zeros(len(names))
    npx = 0
    for shard in sorted(idx["shard"].unique()):
        a = np.load(run_dir / "tiles" / f"shard_{shard:05d}.npy", mmap_mode="r")
        a = np.asarray(a, dtype=np.float64)
        sums += a.sum(axis=(0, 1, 2)); sq += (a ** 2).sum(axis=(0, 1, 2))
        sat += (a >= 255).sum(axis=(0, 1, 2)); nz += (a > 0).sum(axis=(0, 1, 2))
        npx += a.shape[0] * a.shape[1] * a.shape[2]
    mean = sums / npx
    stats = pd.DataFrame({"channel": names, "mean": mean.round(3),
                          "std": np.sqrt(np.maximum(sq / npx - mean ** 2, 0)).round(3),
                          "fraction_saturated_255": (sat / npx).round(5),
                          "fraction_nonzero": (nz / npx).round(4)})
    stats.to_csv(qdir / "channel_tile_stats.csv", index=False)

    # 3. montage of random tiles as an RGB composite
    if rgb is None:
        pick = [n for n in ("CD20", "Pan-Cytokeratin", panel["nuclear_channel"]) if n in names]
        while len(pick) < 3:
            pick.append(names[len(pick)])
        rgb = tuple(pick[:3])
        if "CD3e" in names and "Pan-Cytokeratin" not in names:
            rgb = ("CD20", "CD3e", panel["nuclear_channel"])
    ch = [names.index(c) for c in rgb]
    rng = np.random.default_rng(seed)
    sel = idx.iloc[rng.choice(len(idx), size=min(n_montage, len(idx)), replace=False)]
    cols = 6
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for a in np.ravel(axes):
        a.axis("off")
    # One fixed display limit per channel for every tile (99.5th percentile over the
    # selected tiles). Stretching each tile on its own turns a near-empty channel into
    # amplified noise and makes tiles look different when they are not.
    tiles = []
    for _, r in sel.iterrows():
        sh = np.load(run_dir / "tiles" / f"shard_{r.shard:05d}.npy", mmap_mode="r")
        tiles.append(np.asarray(sh[r.shard_pos][:, :, ch], dtype=np.float32))
        del sh
    hi = np.maximum(np.percentile(np.stack(tiles), 99.5, axis=(0, 1, 2)), 1)
    for a, (_, r), t in zip(np.ravel(axes), sel.iterrows(), tiles):
        a.imshow(np.clip(t / hi, 0, 1))
        a.set_title(f"r{r.grid_row} c{r.grid_col}", fontsize=7)
    fig.suptitle(f"Random tiles (R={rgb[0]}, G={rgb[1]}, B={rgb[2]}), 224 px = {TILE_UM:.0f} um", fontsize=9)
    fig.savefig(qdir / "tile_montage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log_decision(run_dir, "qc", "tiles", ["qc/mask_and_tiles.png", "qc/channel_tile_stats.csv",
                 "qc/tile_montage.png"], f"montage channels {rgb}; seed {seed}")
    return {"rgb": rgb, "n_montage": len(sel)}
