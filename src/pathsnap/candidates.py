"""Step 3: candidate road segments for each GPS observation.

A candidate is a directed arc whose segment lies within `radius` metres of the GPS point.
At most `k_max` candidates are kept per point, nearest first. Both directions of a two-way road
are separate candidates at the same distance; later features (heading) tell them apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from .geo import point_segment
from .network import Network


@dataclass
class Candidates:
    obs: np.ndarray    # (P,) GPS point index
    local: np.ndarray  # (P,) index into the Network arrays
    arc_id: np.ndarray # (P,) original arc id
    dist: np.ndarray   # (P,) point-to-segment distance (m)
    t: np.ndarray      # (P,) projection parameter along the arc, 0..1
    rank: np.ndarray   # (P,) 0 = nearest candidate of that point


def generate(net: Network, px, py, radius: float, k_max: int | None) -> Candidates:
    pts = shapely.points(px, py)
    obs, local = net.tree.query(pts, predicate="dwithin", distance=radius)
    d, t = point_segment(px[obs], py[obs], net.ax[local], net.ay[local], net.bx[local], net.by[local])
    # sort by point, then distance (arc id as a deterministic tie-break)
    order = np.lexsort((net.arc_ids[local], d, obs))
    obs, local, d, t = obs[order], local[order], d[order], t[order]
    rank = _rank_within(obs)
    if k_max is not None:
        m = rank < k_max
        obs, local, d, t, rank = obs[m], local[m], d[m], t[m], rank[m]
    return Candidates(obs, local, net.arc_ids[local], d, t, rank)


def _rank_within(sorted_groups: np.ndarray) -> np.ndarray:
    """0, 1, 2, ... within each run of equal values (input already sorted by group)."""
    n = len(sorted_groups)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    return np.arange(n) - first
