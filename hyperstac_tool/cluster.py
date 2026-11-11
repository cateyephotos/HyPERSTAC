"""Spatial proteomic clusters (SPCs) and their interpretation - runs in WSL (scanpy env).

    python -m hyperstac_tool.cluster --run runs/tonsil31_paper8 --fold 0

Paper (Methods 4.3.2, 4.4.2): Leiden (resolution 1) on the training tiles' representations;
held-out tiles assigned to the same clusters with the learned mapping (reference code:
scanpy ingest); each SPC described by the mean marker content of its tiles, z-scored
across SPCs, and inspected through sample tiles.

Recorded implementation choices:
  - 50-component PCA before the neighbor graph. The reference code calls
    pp.neighbors on the 2048-d features, which runs this PCA implicitly; here it is explicit.
  - Leiden from scanpy/igraph (reference: rapids_singlecell on GPU), random_state 0.
  - No cell segmentation exists for a bare QPTIFF, so SPCs are described by marker
    intensity (the paper's approach for its Hallmarks panel), not cell-type density.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from .train_vicreg import load_tiles


def tile_channel_means(x: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Mean of each channel over tissue pixels (background was set to 0 in every channel)."""
    out = np.zeros((len(x), x.shape[-1]), dtype=np.float32)
    for s in range(0, len(x), chunk):
        b = x[s:s + chunk].astype(np.float32)
        tissue = (b.max(axis=-1, keepdims=True) > 0)
        out[s:s + chunk] = (b * tissue).sum(axis=(1, 2)) / np.maximum(tissue.sum(axis=(1, 2)), 1)
    return out


NEAR_BLANK_FRACTION = 0.05  # channel above zero in < 5% of tissue pixels -> not usable to name clusters


def channel_coverage(x: np.ndarray, names: list, chunk: int = 256) -> pd.DataFrame:
    """Per channel: fraction of tissue pixels above zero, and mean over tissue pixels.

    Z-scores across clusters make a nearly empty channel look strongly enriched
    (tonsil demo 2026-11: CD21 cluster means 0.0-0.1 of 255 scored z = 2.7), so
    near-blank channels are flagged here and left out of the marker heatmap."""
    nz = np.zeros(len(names)); tot = np.zeros(len(names)); n = 0
    for s in range(0, len(x), chunk):
        b = x[s:s + chunk]
        tissue = b.max(axis=-1) > 0
        sel = b[tissue]                      # (pixels, C)
        nz += (sel > 0).sum(axis=0); tot += sel.astype(np.float64).sum(axis=0); n += len(sel)
    cov = pd.DataFrame({"channel": names, "fraction_nonzero": nz / max(n, 1), "mean": tot / max(n, 1)})
    cov["near_blank"] = cov["fraction_nonzero"] < NEAR_BLANK_FRACTION
    return cov


def _palette(n):
    """Up to 40 distinct colors (tab20 + tab20b). The earlier gist_ncar fallback for >20
    clusters faded to near-white at the top end (Cluster 21 invisible on the tonsil map)."""
    cols = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    if n > len(cols):
        cmap = plt.get_cmap("hsv")
        return [cmap(i / n) for i in range(n)]
    return cols[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix of the ssl/fold-<k> folder")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--rgb", default="auto", help="three channel names for galleries, comma-separated")
    ap.add_argument("--gallery-n", type=int, default=8)
    ap.add_argument("--tile-cache", default="~/hyperstac_cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run = Path(args.run)
    fdir = run / "ssl" / f"fold-{args.fold}{args.tag}"
    odir = fdir / "spc"
    odir.mkdir(exist_ok=True)
    emb = np.load(fdir / "embeddings.npz", allow_pickle=False)
    idx = pd.read_csv(run / "tiles" / "tile_index.csv")
    panel = json.loads((run / "panel.json").read_text())
    names = panel["encoder_channels"]

    feats, tile_id, split = emb["features"], emb["tile_id"], emb["split"].astype(str)
    tr, ho = split == "train", split == "heldout"

    # ---- Leiden on training tiles, ingest held-out tiles
    a_tr = ad.AnnData(feats[tr])
    a_tr.obs_names = [f"t{i}" for i in tile_id[tr]]
    sc.pp.pca(a_tr, n_comps=50, random_state=args.seed)
    # use_rep left unset: with >50 features scanpy uses the stored X_pca (as the reference
    # code does), and ingest then projects held-out tiles through the same PCA.
    sc.pp.neighbors(a_tr, random_state=args.seed)
    sc.tl.umap(a_tr, random_state=args.seed)
    sc.tl.leiden(a_tr, resolution=args.resolution, flavor="igraph", n_iterations=2,
                 directed=False, random_state=args.seed)
    # renumber SPCs by size (0 = largest) for stable reading
    order = a_tr.obs["leiden"].value_counts().index.tolist()
    remap = {old: str(new) for new, old in enumerate(order)}
    a_tr.obs["leiden"] = a_tr.obs["leiden"].map(remap).astype("category")
    a_ho = ad.AnnData(feats[ho])
    a_ho.obs_names = [f"t{i}" for i in tile_id[ho]]
    sc.tl.ingest(a_ho, a_tr, obs="leiden", embedding_method="umap")

    # Cross-check: exact 15-NN vote in the same training PCA space. scanpy ingest uses an
    # approximate neighbor index; on synthetic, perfectly separable data it mislabeled ~7%
    # of held-out points (2026-11-09 test), so its agreement with an exact vote is reported.
    from sklearn.neighbors import KNeighborsClassifier
    pcs = a_tr.varm["PCs"]
    x_ho_pca = (feats[ho] - feats[tr].mean(axis=0)) @ pcs
    knn = KNeighborsClassifier(n_neighbors=15).fit(a_tr.obsm["X_pca"], a_tr.obs["leiden"].astype(int))
    exact = knn.predict(x_ho_pca)
    ingest_agreement = float((exact == a_ho.obs["leiden"].astype(int).to_numpy()).mean())

    spc = np.empty(len(tile_id), dtype=int)
    spc[tr] = a_tr.obs["leiden"].astype(int).to_numpy()
    spc[ho] = a_ho.obs["leiden"].astype(int).to_numpy()
    um = np.zeros((len(tile_id), 2))
    um[tr] = a_tr.obsm["X_umap"]
    um[ho] = a_ho.obsm["X_umap"]
    n_spc = int(spc[tr].max()) + 1

    t = idx.loc[tile_id, ["grid_row", "grid_col", "y0_um", "x0_um"]].reset_index(drop=True)
    tiles = pd.DataFrame({"tile_id": tile_id, "split": split, "spc": spc,
                          "umap1": um[:, 0], "umap2": um[:, 1]}).join(t)
    alt = spc.copy()
    alt[ho] = exact
    tiles["spc_exact_knn"] = alt  # sensitivity column; 'spc' (ingest) stays the primary label
    tiles.to_csv(odir / "tile_spc.csv", index=False)

    # ---- QC: sizes, held-out coverage, collapse check
    cnt = pd.crosstab(tiles["spc"], tiles["split"]).reindex(range(n_spc), fill_value=0)
    for col in ("train", "heldout"):
        if col not in cnt:
            cnt[col] = 0
    cnt["train_frac"] = (cnt["train"] / cnt["train"].sum()).round(4)
    cnt["heldout_frac"] = (cnt["heldout"] / max(cnt["heldout"].sum(), 1)).round(4)
    cnt.to_csv(odir / "spc_sizes.csv")
    top3_tr = float(np.sort(cnt["train_frac"].to_numpy())[::-1][:3].sum())
    top3_ho = float(np.sort(cnt["heldout_frac"].to_numpy())[::-1][:3].sum())
    qc = {"n_spc": n_spc, "n_train_tiles": int(tr.sum()), "n_heldout_tiles": int(ho.sum()),
          "spcs_receiving_heldout": int((cnt["heldout"] > 0).sum()),
          "top3_share_train": round(top3_tr, 3), "top3_share_heldout": round(top3_ho, 3),
          "collapse_flag": bool(top3_ho > top3_tr + 0.25),
          "heldout_ingest_vs_exact_knn_agreement": round(ingest_agreement, 4),
          "resolution": args.resolution, "pca_components": 50, "leiden": "scanpy igraph, n_iterations=2",
          "assignment": "scanpy ingest (kNN label transfer onto training PCA/UMAP)"}
    (odir / "spc_qc.json").write_text(json.dumps(qc, indent=2))

    # ---- interpretation: mean marker intensity per SPC (all tiles)
    x = load_tiles(run, tile_id, Path(args.tile_cache).expanduser() if args.tile_cache else None)
    order_ids = np.argsort(np.argsort(tile_id))      # load_tiles returns rows sorted by tile_id
    x = x[order_ids]
    means = pd.DataFrame(tile_channel_means(x), columns=names)
    means.insert(0, "spc", spc)
    means.insert(0, "tile_id", tile_id)
    means.to_csv(odir / "tile_marker_means.csv", index=False)
    per_spc = means.groupby("spc")[names].mean()
    per_spc.round(3).to_csv(odir / "spc_marker_intensity.csv")
    z = (per_spc - per_spc.mean()) / per_spc.std(ddof=0).replace(0, 1)
    z.round(3).to_csv(odir / "spc_marker_intensity_z.csv")

    cov = channel_coverage(x, names)
    cov.round(5).to_csv(odir / "channel_coverage.csv", index=False)
    near_blank = cov.loc[cov["near_blank"], "channel"].tolist()
    qc["near_blank_channels"] = near_blank
    qc["near_blank_rule"] = f"fraction of tissue pixels above zero < {NEAR_BLANK_FRACTION}"
    (odir / "spc_qc.json").write_text(json.dumps(qc, indent=2))
    usable = [c for c in names if c not in near_blank]

    pal = _palette(n_spc)
    title = f"{run.name} fold {args.fold}: {n_spc} clusters"

    # UMAP
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for k in range(n_spc):
        m = spc == k
        axs[0].scatter(um[m & tr, 0], um[m & tr, 1], s=3, color=pal[k], label=str(k))
        axs[1].scatter(um[m & ho, 0], um[m & ho, 1], s=3, color=pal[k])
        if (m & tr).any():
            cx, cy = np.median(um[m & tr], axis=0)
            axs[0].text(cx, cy, str(k), fontsize=8, weight="bold")
    axs[0].set_title("training tiles (clustered)")
    axs[1].set_title("held-out tiles (assigned by ingest)")
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title)
    fig.savefig(odir / "umap_spc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # spatial SPC map
    H = int(idx["grid_row"].max()) + 1
    W = int(idx["grid_col"].max()) + 1
    img = np.ones((H, W, 3))
    for r, c, k in zip(tiles["grid_row"], tiles["grid_col"], tiles["spc"]):
        img[r, c] = pal[k][:3]
    stripe = np.ones((H, W, 3))
    for r, c, s in zip(tiles["grid_row"], tiles["grid_col"], tiles["split"]):
        stripe[r, c] = (0.25, 0.45, 0.85) if s == "train" else (0.95, 0.55, 0.15)
    # Single-panel map with a legend (a deck must never need to crop panels or guess colors);
    # the fold-stripe map is a separate file.
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(img, interpolation="nearest"); ax.axis("off")
    ax.set_title(f"{title}: cluster of each 111 um tile")
    ax.legend(handles=[Patch(color=pal[k], label=f"Cluster {k}") for k in range(n_spc)],
              loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=2 if n_spc > 12 else 1,
              fontsize=9, frameon=False)
    fig.savefig(odir / "spatial_spc_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(stripe, interpolation="nearest"); ax.axis("off")
    ax.set_title(f"{run.name} fold {args.fold}: blue = training stripes, orange = held-out stripes", fontsize=9)
    fig.savefig(odir / "fold_stripes_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # marker heatmap (near-blank channels left out; they stay in the CSVs)
    zu = z[usable]
    fig, ax = plt.subplots(figsize=(max(6, 0.38 * len(usable) + 2.5), 0.38 * n_spc + 2))
    im = ax.imshow(zu.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(usable)), usable, rotation=90, fontsize=9)
    ax.set_yticks(range(n_spc), [f"Cluster {k} (n={int(cnt.loc[k, 'train'] + cnt.loc[k, 'heldout'])})"
                                 for k in range(n_spc)], fontsize=9)
    fig.colorbar(im, ax=ax, label="z-score across clusters of mean tile intensity")
    omitted = f" | left out (near blank): {', '.join(near_blank)}" if near_blank else ""
    ax.set_title(title + " - marker content" + omitted, fontsize=10)
    fig.savefig(odir / "spc_marker_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # galleries: held-out tiles per SPC, fixed display limits
    if args.rgb == "auto":
        pref = [("CD20", "CD3e", "DAPI"), ("Pan-Cytokeratin", "CD3e", "DAPI")]
        rgb = next((p for p in pref if all(c in names for c in p)), tuple(names[:3]))
    else:
        rgb = tuple(c.strip() for c in args.rgb.split(","))
    ch = [names.index(c) for c in rgb]
    rng = np.random.default_rng(args.seed)
    picks = {}
    for k in range(n_spc):
        pool = np.where((spc == k) & ho)[0]
        if len(pool) == 0:
            pool = np.where(spc == k)[0]
        picks[k] = rng.choice(pool, size=min(args.gallery_n, len(pool)), replace=False)
    allp = np.concatenate([v for v in picks.values() if len(v)])
    hi = np.maximum(np.percentile(x[allp][..., ch].astype(np.float32), 99.5, axis=(0, 1, 2)), 1)
    fig, axs = plt.subplots(n_spc, args.gallery_n, figsize=(args.gallery_n * 1.4, n_spc * 1.45))
    axs = np.atleast_2d(axs)
    for k in range(n_spc):
        for j in range(args.gallery_n):
            a = axs[k, j]
            a.set_xticks([]); a.set_yticks([])
            if j < len(picks[k]):
                a.imshow(np.clip(x[picks[k][j]][..., ch].astype(np.float32) / hi, 0, 1))
            else:
                a.axis("off")
        axs[k, 0].set_ylabel(f"Cluster {k}", fontsize=8)
    fig.suptitle(f"{title} - held-out tiles (R={rgb[0]}, G={rgb[1]}, B={rgb[2]})", fontsize=9)
    fig.savefig(odir / "spc_gallery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    with open(run / "DECISION_LOG.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n## SPC clustering fold {args.fold}\n- outputs: {odir}\n- qc: {json.dumps(qc)}\n")
    print(f"[cluster] {json.dumps(qc)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
