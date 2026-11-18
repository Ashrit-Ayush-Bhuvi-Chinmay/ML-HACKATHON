"""Projected, corridor-cropped road network of one trip.

The supplied networks reach 307k arcs, far more than one trip ever comes near. We keep only
arcs with an endpoint inside a corridor around the GPS track (grid cells within `margin` metres
of any track point). That makes the spatial index and the shortest-path searches of the HMM
cheap. Original arc ids are kept, so predictions are always reported in the file's own ids.

Topology features (degrees, two-way flag) are computed on the *full* network, so the crop
boundary cannot distort them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from .geo import Projector
from .io import Trip


@dataclass
class Network:
    arc_ids: np.ndarray   # (E',) original arc ids that were kept
    u: np.ndarray         # (E',) original from-node ids
    v: np.ndarray         # (E',) original to-node ids
    ax: np.ndarray        # segment start x (m)
    ay: np.ndarray
    bx: np.ndarray        # segment end x (m)
    by: np.ndarray
    length: np.ndarray    # (m)
    bearing: np.ndarray   # radians, atan2 convention, direction of travel u -> v
    has_twin: np.ndarray  # bool: the reverse arc v -> u exists (two-way road)
    in_deg_u: np.ndarray  # arcs entering the start node (full network)
    out_deg_v: np.ndarray # arcs leaving the end node (full network)
    tree: shapely.STRtree # spatial index over the kept segments (local index order)
    n_arcs_full: int


def _cell_keys(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return ix.astype(np.int64) * (1 << 31) + iy.astype(np.int64)


def corridor_node_mask(px, py, nx, ny, margin: float, cell: float = 250.0) -> np.ndarray:
    tx = np.floor(px / cell).astype(np.int64)
    ty = np.floor(py / cell).astype(np.int64)
    track_cells = np.unique(np.stack([tx, ty], axis=1), axis=0)
    r = int(np.ceil(margin / cell))
    off = np.arange(-r, r + 1)
    ox, oy = np.meshgrid(off, off, indexing="ij")
    keys = np.unique(_cell_keys(track_cells[:, :1] + ox.ravel()[None], track_cells[:, 1:] + oy.ravel()[None]))
    node_keys = _cell_keys(np.floor(nx / cell).astype(np.int64), np.floor(ny / cell).astype(np.int64))
    return np.isin(node_keys, keys)


def build_network(trip: Trip, proj: Projector, px, py, margin: float = 1000.0) -> Network:
    arcs = trip.arcs
    n_nodes = len(trip.nodes)
    nx, ny = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])

    # full-network topology
    out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)
    in_deg = np.bincount(arcs[:, 1], minlength=n_nodes)
    fwd_key = arcs[:, 0] * n_nodes + arcs[:, 1]
    rev_key = arcs[:, 1] * n_nodes + arcs[:, 0]
    has_twin = np.isin(rev_key, fwd_key)

    inside = corridor_node_mask(px, py, nx, ny, margin)
    keep = np.flatnonzero(inside[arcs[:, 0]] | inside[arcs[:, 1]])
    u, v = arcs[keep, 0], arcs[keep, 1]
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]
    seg = np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], axis=1)  # (E', 2, 2)
    return Network(
        arc_ids=keep, u=u, v=v, ax=ax, ay=ay, bx=bx, by=by,
        length=np.hypot(bx - ax, by - ay),
        bearing=np.arctan2(by - ay, bx - ax),
        has_twin=has_twin[keep],
        in_deg_u=in_deg[u], out_deg_v=out_deg[v],
        tree=shapely.STRtree(shapely.linestrings(seg)),
        n_arcs_full=len(arcs),
    )
