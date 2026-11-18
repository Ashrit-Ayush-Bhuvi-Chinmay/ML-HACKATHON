"""Reading PathSnap trips.

A trip folder holds four whitespace-separated, header-less text files named after the trip id:

    <id>.track   lon lat t          GPS observations, t in seconds
    <id>.nodes   lon lat            row index = node id
    <id>.arcs    from_node to_node  row index = arc id (directed)
    <id>.route   arc_id             reference route, chronological (labels / scoring only)

`load_trip` never touches `.route`, so prediction code built on it cannot see the reference
route. `load_route` is the only reader of `.route`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path, dtype) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c").to_numpy()


def find_file(trip_dir: Path, ext: str) -> Path:
    """Locate a trip file. Supports both layouts:
         <folder>/<trip_id>/<trip_id><ext>   (one sub-folder per trip, as supplied)
         <folder>/<trip_id><ext>             (all files flat in one folder; trip_dir is then
                                              the virtual path <folder>/<trip_id>)
    falling back to the only `*<ext>` file inside a trip sub-folder."""
    trip_dir = Path(trip_dir)
    for path in (trip_dir / f"{trip_dir.name}{ext}", trip_dir.parent / f"{trip_dir.name}{ext}"):
        if path.is_file():
            return path
    hits = sorted(trip_dir.glob(f"*{ext}")) if trip_dir.is_dir() else []
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one *{ext} file for trip {trip_dir.name} in {trip_dir}, found {len(hits)}")
    return hits[0]


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: np.ndarray  # (N, 3) float64: lon, lat, t
    nodes: np.ndarray  # (V, 2) float64: lon, lat
    arcs: np.ndarray   # (E, 2) int64: from_node, to_node

    @property
    def n_points(self) -> int:
        return len(self.track)


def list_trips(split_dir: Path) -> list[Path]:
    """Trips under a directory, sorted by id: sub-folders containing a .track file, or (flat
    layout) one virtual path <dir>/<stem> per <stem>.track file. Never hardcode trip ids."""
    split_dir = Path(split_dir)
    folders = sorted(p for p in split_dir.iterdir() if p.is_dir() and any(p.glob("*.track")))
    if folders:
        return folders
    return [split_dir / s for s in sorted({p.stem for p in split_dir.glob("*.track")})]


def load_trip(trip_dir: Path) -> Trip:
    trip_dir = Path(trip_dir)
    track = _read_table(find_file(trip_dir, ".track"), np.float64)
    nodes = _read_table(find_file(trip_dir, ".nodes"), np.float64)
    arcs = _read_table(find_file(trip_dir, ".arcs"), np.int64)
    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(f"{trip_dir.name}: .track must have 3 columns, got {track.shape}")
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"{trip_dir.name}: .nodes must have 2 columns, got {nodes.shape}")
    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError(f"{trip_dir.name}: .arcs must have 2 columns, got {arcs.shape}")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"{trip_dir.name}: non-finite coordinates")
    if len(arcs) and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise ValueError(f"{trip_dir.name}: arc refers to a node outside .nodes")
    if len(track) == 0:
        raise ValueError(f"{trip_dir.name}: empty .track")
    if np.any(np.diff(track[:, 2]) < 0):
        # all supplied trips are sorted; an unseen one might not be. A stable sort keeps the
        # result deterministic, so prediction and scoring see the same point order.
        track = track[np.argsort(track[:, 2], kind="stable")]
    return Trip(trip_dir.name, track, nodes, arcs)


def load_route(trip_dir: Path) -> np.ndarray:
    """Reference route (arc ids in travel order). Only for labels and server-side scoring."""
    return _read_table(find_file(Path(trip_dir), ".route"), np.int64).ravel()
