"""Step 2: a reference arc for every GPS observation.

`.route` is only an ordered list of directed arc ids; the data has no per-observation label.
We derive one by aligning the track to the route *in order*:

    choose route positions j(0) <= j(1) <= ... <= j(N-1) minimising
        sum_i  min(dist(p_i, arc_j(i)), cap)  +  w_heading * h(i, j(i)) * moving_i

  dist      point-to-segment distance in metres
  h         (1 - cos(heading_i - bearing_j)) / 2: 0 when the arc points the way the vehicle
            moves, 1 when it points the opposite way
  moving_i  1 if speed >= min_speed (heading is meaningless when the vehicle is stopped)

Why each piece:
  * monotonic order follows loops and repeated arcs (10 routes revisit arcs) correctly,
    which "nearest route arc" cannot;
  * the heading term separates the two directions of a two-way road (92 % of arcs have a
    reverse twin with identical geometry);
  * the cap stops one GPS spike from dragging the whole alignment forward.

Solved by dynamic programming, C[i, j] = D[i, j] + min_{k <= j} C[i-1, k], row by row with a
running prefix minimum, then backtracked (ties go to the earlier route position).
O(N * M) time and float32 memory: about 175 MB for the largest trip (19k points x 2.3k arcs).
"""
from __future__ import annotations

import numpy as np

from .geo import Projector, point_segment, track_kinematics
from .io import Trip


def align_monotonic(px, py, heading, moving, ax, ay, bx, by,
                    w_heading: float = 25.0, cap: float = 150.0, chunk: int = 1024) -> np.ndarray:
    """Route position (index into the segment arrays) for every point. See module docstring."""
    n, m = len(px), len(ax)
    seg_bearing = np.arctan2(by - ay, bx - ax)
    cost = np.empty((n, m), dtype=np.float32)
    prev = None
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d, _ = point_segment(px[s:e, None], py[s:e, None], ax[None], ay[None], bx[None], by[None])
        d = np.minimum(d, cap)
        if w_heading > 0:
            h = 0.5 * (1.0 - np.cos(heading[s:e, None] - seg_bearing[None]))
            d = d + w_heading * h * moving[s:e, None]
        d = d.astype(np.float32)
        for r in range(e - s):
            cost[s + r] = d[r] if prev is None else d[r] + np.minimum.accumulate(prev)
            prev = cost[s + r]

    pos = np.empty(n, dtype=np.int64)
    j = int(np.argmin(cost[n - 1]))
    pos[n - 1] = j
    for i in range(n - 1, 0, -1):
        j = int(np.argmin(cost[i - 1, : j + 1]))
        pos[i - 1] = j
    return pos


def label_trip(trip: Trip, route: np.ndarray, w_heading: float = 25.0, cap: float = 150.0,
               min_speed: float = 2.0, k: int = 2) -> dict[str, np.ndarray]:
    """Per-point reference labels for one trip.

    Returns arrays of length N: route_pos (index into `route`), arc_id (reference arc),
    dist (metres from the point to that arc), t_on_arc (projection parameter), cos_dir
    (cosine between vehicle heading and arc direction) and speed (m/s).
    """
    proj = Projector.for_track(trip.track)
    px, py = proj.to_xy(trip.track[:, 0], trip.track[:, 1])
    heading, speed = track_kinematics(px, py, trip.track[:, 2], k)
    moving = (speed >= min_speed).astype(np.float64)

    a, b = trip.arcs[route, 0], trip.arcs[route, 1]
    ax, ay = proj.to_xy(trip.nodes[a, 0], trip.nodes[a, 1])
    bx, by = proj.to_xy(trip.nodes[b, 0], trip.nodes[b, 1])

    pos = align_monotonic(px, py, heading, moving, ax, ay, bx, by, w_heading, cap)
    dist, t_on_arc = point_segment(px, py, ax[pos], ay[pos], bx[pos], by[pos])
    cos_dir = np.cos(heading - np.arctan2(by - ay, bx - ax)[pos])
    return {
        "route_pos": pos,
        "arc_id": route[pos],
        "dist": dist,
        "t_on_arc": t_on_arc,
        "cos_dir": cos_dir,
        "speed": speed,
    }
