"""Step 4: features for every (GPS point, candidate arc) pair.

Everything here is computed from the trip's own .track/.nodes/.arcs. No labels are used and
nothing is fitted, so the same code runs unchanged at training time and inside the API.

Feature groups
  geometry   dist, t_on_arc, beyond_end_m, arc_len
  direction  cos_heading, abs_dheading, heading_valid, speed
             (heading from points >= 10 m apart: a 1 Hz track moving at walking pace has
              more GPS noise than displacement, see results/labels)
  context    distance from neighbouring points (i-5 ... i+5) to the same arc, mean / max
             ("does this arc explain the neighbourhood, not just this one point?")
  topology   has_twin (two-way road), in_deg_u, out_deg_v (junction complexity)
  time/noise log_dt_prev, log_dt_next, jitter
  relative   n_cands, dist_gap, ctx_gap, cos_gap: how this candidate compares with the best
             competitor for the same point (the decision is always "which of these")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import Candidates, generate
from .geo import Projector, point_segment, wrap_angle
from .io import Trip
from .network import Network, build_network

CONTEXT_OFFSETS = (-5, -2, -1, 1, 2, 5)

FEATURES = [
    "dist", "t_on_arc", "beyond_end_m", "arc_len",
    "cos_heading", "abs_dheading", "heading_valid", "speed",
    *[f"dist_ctx{o:+d}" for o in CONTEXT_OFFSETS], "dist_ctx_mean", "dist_ctx_max",
    "has_twin", "in_deg_u", "out_deg_v",
    "log_dt_prev", "log_dt_next", "jitter",
    "n_cands", "dist_gap", "ctx_gap", "cos_gap",
]


@dataclass
class TripContext:
    """Everything derived from one trip that later stages (HMM, API) need again."""
    trip: Trip
    proj: Projector
    px: np.ndarray
    py: np.ndarray
    t: np.ndarray
    heading: np.ndarray
    heading_valid: np.ndarray
    speed: np.ndarray
    net: Network
    cands: Candidates


def robust_heading(x, y, t, min_disp: float = 10.0, max_steps: int = 10, max_dt: float = 30.0):
    """Heading/speed from the nearest points behind and ahead that are >= min_disp metres away
    (searching up to max_steps points and max_dt seconds each way)."""
    n = len(x)
    i = np.arange(n)

    def first_far(direction: int):
        idx = i.copy()
        found = np.zeros(n, dtype=bool)
        for k in range(1, max_steps + 1):
            j = np.clip(i + direction * k, 0, n - 1)
            far = (~found & (j != i) & (np.abs(t[j] - t) <= max_dt)
                   & (np.hypot(x[j] - x, y[j] - y) >= min_disp))
            idx[far] = j[far]
            found |= far
        return idx, found

    ahead, ok_a = first_far(+1)
    behind, ok_b = first_far(-1)
    dx = x[ahead] - x[behind]
    dy = y[ahead] - y[behind]
    disp = np.hypot(dx, dy)
    dt = t[ahead] - t[behind]
    valid = (ok_a | ok_b) & (disp >= min_disp)
    speed = np.where(dt > 0, disp / np.where(dt > 0, dt, 1.0), 0.0)
    return np.arctan2(dy, dx), valid, speed


def _group_reduce(values: np.ndarray, obs: np.ndarray, ufunc) -> np.ndarray:
    """Reduce `values` within each run of equal `obs` (sorted) and broadcast back to rows."""
    if len(obs) == 0:
        return values.copy()
    starts = np.flatnonzero(np.r_[True, obs[1:] != obs[:-1]])
    red = ufunc.reduceat(values, starts)
    counts = np.diff(np.r_[starts, len(obs)])
    return np.repeat(red, counts)


def prepare_trip(trip: Trip, radius: float, k_max: int | None, margin: float = 1000.0) -> TripContext:
    proj = Projector.for_track(trip.track)
    px, py = proj.to_xy(trip.track[:, 0], trip.track[:, 1])
    t = trip.track[:, 2]
    heading, hvalid, speed = robust_heading(px, py, t)
    net = build_network(trip, proj, px, py, margin=margin)
    cands = generate(net, px, py, radius=radius, k_max=k_max)
    return TripContext(trip, proj, px, py, t, heading, hvalid, speed, net, cands)


def pair_features(ctx: TripContext) -> pd.DataFrame:
    net, c = ctx.net, ctx.cands
    px, py, t = ctx.px, ctx.py, ctx.t
    o, a = c.obs, c.local
    n = len(px)
    f: dict[str, np.ndarray] = {}

    # geometry
    ax, ay, bx, by, L = net.ax[a], net.ay[a], net.bx[a], net.by[a], net.length[a]
    t_raw = ((px[o] - ax) * (bx - ax) + (py[o] - ay) * (by - ay)) / np.where(L > 0, L * L, 1.0)
    f["dist"] = c.dist
    f["t_on_arc"] = np.clip(t_raw, 0.0, 1.0)
    f["beyond_end_m"] = np.maximum(0.0, np.maximum(-t_raw, t_raw - 1.0)) * L
    f["arc_len"] = L

    # direction
    dth = wrap_angle(ctx.heading[o] - net.bearing[a])
    hv = ctx.heading_valid[o]
    f["cos_heading"] = np.where(hv, np.cos(dth), 0.0)
    f["abs_dheading"] = np.where(hv, np.abs(dth), np.pi / 2)
    f["heading_valid"] = hv.astype(np.float64)
    f["speed"] = ctx.speed[o]

    # context: the same arc seen from neighbouring points
    ctx_d = []
    for off in CONTEXT_OFFSETS:
        j = np.clip(o + off, 0, n - 1)
        d, _ = point_segment(px[j], py[j], ax, ay, bx, by)
        f[f"dist_ctx{off:+d}"] = d
        ctx_d.append(d)
    ctx_d = np.stack(ctx_d, axis=1)
    f["dist_ctx_mean"] = ctx_d.mean(axis=1)
    f["dist_ctx_max"] = ctx_d.max(axis=1)

    # topology
    f["has_twin"] = net.has_twin[a].astype(np.float64)
    f["in_deg_u"] = net.in_deg_u[a].astype(np.float64)
    f["out_deg_v"] = net.out_deg_v[a].astype(np.float64)

    # time gaps and local GPS noise (point-level, broadcast to its candidates)
    dt = np.diff(t)
    dt_prev = np.r_[np.nan, dt]
    dt_next = np.r_[dt, np.nan]
    f["log_dt_prev"] = np.log1p(np.nan_to_num(dt_prev, nan=1.0))[o]
    f["log_dt_next"] = np.log1p(np.nan_to_num(dt_next, nan=1.0))[o]
    lo, hi = np.clip(np.arange(n) - 1, 0, n - 1), np.clip(np.arange(n) + 1, 0, n - 1)
    jitter = np.hypot(px - 0.5 * (px[lo] + px[hi]), py - 0.5 * (py[lo] + py[hi]))
    f["jitter"] = jitter[o]

    # relative to the other candidates of the same point
    f["n_cands"] = _group_reduce(np.ones(len(o)), o, np.add)
    f["dist_gap"] = c.dist - _group_reduce(c.dist, o, np.minimum)
    f["ctx_gap"] = f["dist_ctx_mean"] - _group_reduce(f["dist_ctx_mean"], o, np.minimum)
    f["cos_gap"] = _group_reduce(f["cos_heading"], o, np.maximum) - f["cos_heading"]

    df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})[FEATURES]
    df.insert(0, "arc_id", c.arc_id.astype(np.int64))
    df.insert(0, "obs", o.astype(np.int64))
    return df
