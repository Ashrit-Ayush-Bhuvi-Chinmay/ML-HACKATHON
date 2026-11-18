"""Per-trip metric projection and vectorised planar geometry.

The raw coordinates are WGS84 lon/lat in degrees and the trips span several continents, so
distances are never computed in degrees. Each trip is projected to the UTM zone of its median
GPS position (conformal, < 0.1 % scale error inside the zone), and all geometry is done in metres.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    """lon/lat (degrees) <-> local UTM x/y (metres) for one trip."""

    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    @classmethod
    def for_track(cls, track: np.ndarray) -> "Projector":
        return cls(float(np.median(track[:, 0])), float(np.median(track[:, 1])))

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (numpy broadcasting), plus the projection
    parameter t clipped to [0, 1] (0 = at A, 1 = at B)."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def wrap_angle(a):
    """Wrap radians to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def track_kinematics(x, y, t, k: int = 2):
    """Heading (radians, atan2 convention) and speed (m/s) at each point from a centred
    difference over the points k steps before and after (clamped at the ends)."""
    n = len(x)
    i = np.arange(n)
    lo = np.clip(i - k, 0, n - 1)
    hi = np.clip(i + k, 0, n - 1)
    dx = x[hi] - x[lo]
    dy = y[hi] - y[lo]
    dt = t[hi] - t[lo]
    heading = np.arctan2(dy, dx)
    speed = np.where(dt > 0, np.hypot(dx, dy) / np.where(dt > 0, dt, 1.0), 0.0)
    return heading, speed
