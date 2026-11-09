"""Input adapters: map any multichannel image source to one common shape.

Every adapter exposes the same thing:
    channels      list of dicts {index, name, role}
    pixel_size_um physical pixel size at full resolution
    shape_yx      full-resolution (height, width)
    dtype         numpy dtype of the pixel data
    levels        list of zarr arrays, level 0 = full resolution, each (C, Y, X)
    read(level, y0, y1, x0, x1, channel_idx) -> (C, y, x) array

Supported: Akoya/PerkinElmer QPTIFF, generic OME-TIFF, 10x Xenium bundle
(morphology_focus/chNNNN_*.ome.tif). Phenotyped h5ad files hold no pixels and
are intentionally NOT an image source (see the reproduction guide).
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
import zarr

NUCLEAR_NAMES = {"dapi", "hoechst", "dna1", "dna2", "nuclear"}


def _role_for(name: str) -> str:
    n = name.strip().lower()
    if n in NUCLEAR_NAMES:
        return "nuclear"
    if "background" in n or "autofluor" in n or n in {"af", "sample af"}:
        return "background"
    return "marker"


@dataclass
class ImageSource:
    path: Path
    kind: str
    sample_id: str
    channels: list = field(default_factory=list)
    pixel_size_um: float = float("nan")
    shape_yx: tuple = (0, 0)
    dtype: str = ""
    level_shapes: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    _tif: object = None
    _levels: list = field(default_factory=list)

    # ---- pixel access -------------------------------------------------
    def read(self, level: int, y0: int, y1: int, x0: int, x1: int, channel_idx=None) -> np.ndarray:
        arr = self._levels[level]
        if channel_idx is None:
            return np.asarray(arr[:, y0:y1, x0:x1])
        # zarr orthogonal selection on the channel axis; contiguous slice is faster
        idx = list(channel_idx)
        if idx == list(range(idx[0], idx[-1] + 1)):
            return np.asarray(arr[idx[0]:idx[-1] + 1, y0:y1, x0:x1])
        return np.stack([np.asarray(arr[i, y0:y1, x0:x1]) for i in idx])

    def level_downsample(self, level: int) -> float:
        return self.shape_yx[1] / self.level_shapes[level][2]

    def channel_index(self, name: str) -> int:
        for c in self.channels:
            if c["name"].lower() == name.lower():
                return c["index"]
        raise KeyError(f"channel {name!r} not found; have {[c['name'] for c in self.channels]}")

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "sample_id": self.sample_id,
            "pixel_size_um": self.pixel_size_um,
            "shape_yx": list(self.shape_yx),
            "size_mm": [round(self.shape_yx[0] * self.pixel_size_um / 1000, 3),
                        round(self.shape_yx[1] * self.pixel_size_um / 1000, 3)],
            "dtype": self.dtype,
            "n_channels": len(self.channels),
            "level_shapes": [list(s) for s in self.level_shapes],
            "channels": self.channels,
            "metadata": self.metadata,
        }


def _open_series(src: ImageSource, series_index: int = 0) -> None:
    tif = tifffile.TiffFile(src.path)
    series = tif.series[series_index]
    if series.axes not in ("CYX", "IYX", "QYX"):
        raise ValueError(f"expected a channel-first series, got axes {series.axes}")
    store = series.aszarr()
    group = zarr.open(store, mode="r")
    if isinstance(group, zarr.Array):
        levels = [group]
    else:
        levels = [group[k] for k in sorted(group.keys(), key=int)]
    src._tif = tif
    src._levels = levels
    src.level_shapes = [tuple(l.shape) for l in levels]
    src.shape_yx = tuple(levels[0].shape[1:])
    src.dtype = str(levels[0].dtype)


def _resolution_um(page) -> float:
    """Pixel size from TIFF XResolution/ResolutionUnit (2 = inch, 3 = cm)."""
    tag = page.tags.get("XResolution")
    unit = page.tags.get("ResolutionUnit")
    if tag is None or unit is None:
        return float("nan")
    num, den = tag.value
    px_per_unit = num / den
    unit_um = {2: 25400.0, 3: 10000.0}.get(int(unit.value), float("nan"))
    return unit_um / px_per_unit


# ---------------------------------------------------------------------------
def open_qptiff(path: Path) -> ImageSource:
    src = ImageSource(path=path, kind="qptiff", sample_id=path.stem)
    _open_series(src, 0)
    pages = src._tif.series[0].pages
    chans, meta = [], {}
    for i, p in enumerate(pages):
        desc = p.description or ""
        try:
            root = ET.fromstring(desc.encode("utf-16") if "utf-16" in desc[:60] else desc)
            name = (root.findtext("Biomarker") or root.findtext("Name") or f"ch{i}").strip()
            if i == 0:
                for key in ("AcquisitionSoftware", "SlideID", "Objective", "StudyName",
                            "AutofluorescenceSubtracted", "SignalUnits", "IsUnmixedComponent"):
                    meta[key] = root.findtext(key)
        except ET.ParseError:
            m = re.search(r"<Biomarker>(.*?)</Biomarker>", desc) or re.search(r"<Name>(.*?)</Name>", desc)
            name = m.group(1).strip() if m else f"ch{i}"
        chans.append({"index": i, "name": name, "role": _role_for(name)})
    src.channels = chans
    src.metadata = meta
    src.pixel_size_um = _resolution_um(pages[0])
    return src


def open_ome_tiff(path: Path) -> ImageSource:
    src = ImageSource(path=path, kind="ome-tiff", sample_id=path.name.split(".")[0])
    _open_series(src, 0)
    ome = src._tif.ome_metadata or ""
    names, px = [], float("nan")
    if ome:
        root = ET.fromstring(ome)
        ns = {"o": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        pix = root.find(".//o:Pixels", ns) if ns else root.find(".//Pixels")
        if pix is not None:
            px = float(pix.get("PhysicalSizeX", "nan"))
            for ch in (pix.findall("o:Channel", ns) if ns else pix.findall("Channel")):
                names.append(ch.get("Name") or ch.get("ID"))
    if not names:
        names = [f"ch{i}" for i in range(src._levels[0].shape[0])]
    src.channels = [{"index": i, "name": n, "role": _role_for(n)} for i, n in enumerate(names)]
    src.pixel_size_um = px if px == px else _resolution_um(src._tif.pages[0])
    return src


def open_xenium_bundle(path: Path) -> ImageSource:
    exp = json.loads((path / "experiment.xenium").read_text())
    files = sorted((path / "morphology_focus").glob("ch*.ome.tif"))
    if not files:
        raise FileNotFoundError(f"no morphology_focus/ch*.ome.tif under {path}")
    src = ImageSource(path=files[0], kind="xenium", sample_id=path.name)
    _open_series(src, 0)  # tifffile joins the per-channel files into one series
    chans = []
    for i, f in enumerate(files):
        m = re.match(r"ch\d{4}_(.+)\.ome\.tif", f.name)
        raw = m.group(1) if m else f.stem
        role = _role_for(raw)
        if role == "marker" and i in (1, 2, 3):
            role = "segmentation"  # Xenium cell-segmentation stain composites
        chans.append({"index": i, "name": raw, "role": role})
    src.channels = chans
    src.pixel_size_um = float(exp["pixel_size"])
    src.metadata = {k: exp.get(k) for k in ("analysis_sw_version", "run_name", "instrument_sn", "slide_id")}
    return src


def detect_and_open(path: str | Path) -> ImageSource:
    p = Path(path)
    if p.is_dir() and (p / "experiment.xenium").exists():
        return open_xenium_bundle(p)
    name = p.name.lower()
    if name.endswith(".qptiff"):
        return open_qptiff(p)
    if name.endswith((".ome.tif", ".ome.tiff")):
        return open_ome_tiff(p)
    if name.endswith(".h5ad"):
        raise ValueError(
            "h5ad files hold per-cell values, not pixels. HyPERSTAC needs the image "
            "(raw Xenium bundle, QPTIFF or OME-TIFF). Use the h5ad only for SPC "
            "interpretation; see docs/help/hyperstac_reproduction_guide.html.")
    if name.endswith((".tif", ".tiff")):
        return open_ome_tiff(p)
    raise ValueError(f"unrecognized input: {p}")
