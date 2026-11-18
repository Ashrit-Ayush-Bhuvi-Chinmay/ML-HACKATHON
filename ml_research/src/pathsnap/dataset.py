"""Readers for the supplied PathSnap no-header, whitespace-delimited trip files."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import ZipFile

import numpy as np

REQUIRED_EXTENSIONS = (".track", ".nodes", ".arcs", ".route")


@dataclass(frozen=True)
class TripRef:
    split: str
    trip_id: str
    files: dict[str, str]


@dataclass(frozen=True)
class TripData:
    split: str
    trip_id: str
    track: np.ndarray  # lon, lat, timestamp_seconds
    nodes: np.ndarray  # lon, lat; row index is node_id
    arcs: np.ndarray  # from_node_id, to_node_id; row index is arc_id
    route: np.ndarray  # chronological arc_id sequence


def list_trips(archive_path: str | Path, split: str | None = None) -> list[TripRef]:
    """Inventory trip files in the supplied ZIP without extracting source data."""
    found: dict[tuple[str, str], dict[str, str]] = {}
    with ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            path = PurePosixPath(entry.filename)
            suffix = path.suffix.lower()
            if suffix not in REQUIRED_EXTENSIONS or len(path.parts) < 4:
                continue
            trip_id, current_split = path.parent.name, path.parent.parent.name
            if split is not None and current_split != split:
                continue
            found.setdefault((current_split, trip_id), {})[suffix] = entry.filename
    refs = []
    for (current_split, trip_id), files in sorted(found.items()):
        missing = set(REQUIRED_EXTENSIONS) - files.keys()
        if missing:
            raise ValueError(f"Trip {current_split}/{trip_id} is missing files: {sorted(missing)}")
        refs.append(TripRef(current_split, trip_id, files))
    return refs


def _numeric_rows(stream: Iterable[str], width: int, cast):
    rows = []
    for line_no, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != width:
            raise ValueError(f"Expected {width} fields on line {line_no}; found {len(fields)}")
        try:
            rows.append(tuple(cast(v) for v in fields))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value on line {line_no}: {line.rstrip()}") from exc
    return rows


def read_trip(archive_path: str | Path, ref: TripRef) -> TripData:
    """Read one trip using the documented column order and implicit row-index IDs."""
    parsed = {}
    with ZipFile(archive_path) as archive:
        for suffix, width, cast in ((".track", 3, float), (".nodes", 2, float),
                                    (".arcs", 2, int), (".route", 1, int)):
            with archive.open(ref.files[suffix]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
                parsed[suffix] = _numeric_rows(text, width, cast)
    track = np.asarray(parsed[".track"], dtype=np.float64).reshape((-1, 3))
    nodes = np.asarray(parsed[".nodes"], dtype=np.float64).reshape((-1, 2))
    arcs = np.asarray(parsed[".arcs"], dtype=np.int64).reshape((-1, 2))
    route = np.asarray([r[0] for r in parsed[".route"]], dtype=np.int64)
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"Trip {ref.trip_id} contains non-finite coordinates or timestamps")
    if len(track) > 1 and np.any(np.diff(track[:, 2]) <= 0):
        raise ValueError(f"Trip {ref.trip_id} timestamps are not strictly increasing")
    if len(arcs) and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise ValueError(f"Trip {ref.trip_id} contains arc endpoint IDs outside its node table")
    if len(route) and (route.min() < 0 or route.max() >= len(arcs)):
        raise ValueError(f"Trip {ref.trip_id} contains route arc IDs outside its arc table")
    return TripData(ref.split, ref.trip_id, track, nodes, arcs, route)
