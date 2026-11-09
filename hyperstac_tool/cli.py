"""Command line: python -m hyperstac_tool {inspect,prep} ...

  inspect  print channels, pixel size and size of any supported input
  prep     inspect -> panel -> scaling -> tissue mask -> tiles -> QC, into one run folder
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # never matplotlib.use() (crash compendium MPL-001)

PAPER_8 = ["DAPI", "Pan-Cytokeratin", "CD68", "SMA", "CD4", "CD8", "Ki67", "CD3e"]


def _parse(argv=None):
    ap = argparse.ArgumentParser(prog="hyperstac_tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inspect", help="describe an input image")
    a.add_argument("input")
    b = sub.add_parser("prep", help="run the CPU preprocessing stages")
    b.add_argument("input", help="QPTIFF, OME-TIFF, or Xenium bundle folder")
    b.add_argument("--out", required=True, help="run folder (created; reruns resume)")
    b.add_argument("--nuclear", default="DAPI", help="nuclear channel name for the tissue mask")
    b.add_argument("--channels", default="all",
                   help="'all', 'paper8', or a comma-separated list of channel names")
    b.add_argument("--scaling", default="auto", choices=["auto", "none", "percentile"])
    b.add_argument("--max-rows", type=int, default=None, help="limit tile rows (smoke tests)")
    b.add_argument("--shard-size", type=int, default=256)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    from .sources import detect_and_open
    from . import pipeline as P

    src = detect_and_open(args.input)
    if args.cmd == "inspect":
        rep = src.summary()
        print(json.dumps({k: v for k, v in rep.items() if k != "channels"}, indent=2, default=str))
        for c in rep["channels"]:
            print(f"  {c['index']:>3}  {c['role']:<12} {c['name']}")
        return 0

    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P.stage_inspect(src, run)
    if args.channels == "all":
        chans = None
    elif args.channels == "paper8":
        chans = PAPER_8
    else:
        chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    panel = P.stage_panel(src, run, args.nuclear, chans)
    print(f"[panel] {panel['n_encoder_channels']} channels", flush=True)
    scaling = P.stage_scaling(src, run, panel, mode=args.scaling)
    print(f"[scaling] {scaling['mode']}", flush=True)
    mask = P.stage_mask(src, run, panel)
    print(f"[mask] tissue {mask['params']['tissue_fraction_small']:.1%}  ({time.time()-t0:.0f}s)", flush=True)
    tiles = P.stage_tiles(src, run, panel, scaling, mask, shard_size=args.shard_size, max_rows=args.max_rows)
    print(f"[tiles] {tiles['n_tiles_kept']} tiles in {tiles['n_shards']} shards ({time.time()-t0:.0f}s)", flush=True)
    P.stage_qc(src, run, panel, mask)
    print(f"[qc] done ({time.time()-t0:.0f}s) -> {run}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
