"""Per-trip UTM projection and KD-tree-assisted arc candidate features."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, degrees, radians, sin, sqrt, tan
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
UTM_K0 = 0.9996


def utm_zone(longitude: float) -> int:
    return max(1, min(60, int((longitude + 180.0) // 6.0) + 1))


def project_utm(longitude: float, latitude: float, zone: int | None = None) -> tuple[float, float, int, str]:
    """Project WGS84 lon/lat to WGS84 / UTM metres (EPSG:326xx or EPSG:327xx).

    The series is the standard Transverse Mercator UTM forward transform. Each trip uses
    the UTM zone containing its median longitude, so worldwide trips use local metres.
    """
    zone = utm_zone(longitude) if zone is None else int(zone)
    if not -80 <= latitude <= 84:
        raise ValueError(f"UTM is defined only from 80S to 84N, got latitude {latitude}")
    e2 = WGS84_F * (2 - WGS84_F)
    ep2 = e2 / (1 - e2)
    phi = radians(latitude)
    lam = radians(longitude)
    lam0 = radians(zone * 6 - 183)
    sin_phi, cos_phi = sin(phi), cos(phi)
    tan_phi = tan(phi)
    n = WGS84_A / sqrt(1 - e2 * sin_phi * sin_phi)
    t = tan_phi * tan_phi
    c = ep2 * cos_phi * cos_phi
    a = cos_phi * (lam - lam0)
    e4, e6 = e2 * e2, e2 * e2 * e2
    m = WGS84_A * ((1 - e2/4 - 3*e4/64 - 5*e6/256) * phi
                   - (3*e2/8 + 3*e4/32 + 45*e6/1024) * sin(2*phi)
                   + (15*e4/256 + 45*e6/1024) * sin(4*phi)
                   - (35*e6/3072) * sin(6*phi))
    easting = UTM_K0 * n * (a + (1-t+c)*a**3/6
                             + (5-18*t+t*t+72*c-58*ep2)*a**5/120) + 500_000
    northing = UTM_K0 * (m + n*tan_phi*(a*a/2
                             + (5-t+9*c+4*c*c)*a**4/24
                             + (61-58*t+t*t+600*c-330*ep2)*a**6/720))
    if latitude < 0:
        northing += 10_000_000
    epsg = f"EPSG:{(32700 if latitude < 0 else 32600) + zone}"
    return easting, northing, zone, epsg


@dataclass
class RoadIndex:
    nodes_xy: np.ndarray
    arcs: np.ndarray
    start_xy: np.ndarray
    end_xy: np.ndarray
    lengths_m: np.ndarray
    bearings_deg: np.ndarray
    node_degree: np.ndarray
    tree: cKDTree
    sample_edge_ids: np.ndarray
    sample_spacing_m: float

    @classmethod
    def build(cls, nodes_lonlat: np.ndarray, arcs: np.ndarray, zone: int,
              sample_spacing_m: float = 40.0) -> "RoadIndex":
        nodes_xy = np.asarray([project_utm(lon, lat, zone)[:2] for lon, lat in nodes_lonlat], dtype=np.float64)
        start_xy, end_xy = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]
        vectors = end_xy - start_xy
        lengths = np.linalg.norm(vectors, axis=1)
        bearings = (np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360.0) % 360.0
        degree = np.zeros(len(nodes_xy), dtype=np.int32)
        np.add.at(degree, arcs[:, 0], 1)
        np.add.at(degree, arcs[:, 1], 1)
        sample_xy_parts, sample_id_parts = [], []
        for edge_id, length in enumerate(lengths):
            n_segments = max(1, int(ceil(length / sample_spacing_m)))
            fractions = np.linspace(0.0, 1.0, n_segments + 1)
            sample_xy_parts.append(start_xy[edge_id] + fractions[:, None] * vectors[edge_id])
            sample_id_parts.append(np.full(n_segments + 1, edge_id, dtype=np.int32))
        sample_xy = np.concatenate(sample_xy_parts) if sample_xy_parts else np.empty((0, 2))
        sample_ids = np.concatenate(sample_id_parts) if sample_id_parts else np.empty((0,), dtype=np.int32)
        return cls(nodes_xy, arcs, start_xy, end_xy, lengths, bearings, degree,
                   cKDTree(sample_xy), sample_ids, sample_spacing_m)

    def query(self, point_xy: np.ndarray, radius_m: float, max_candidates: int | None = None) -> list[dict[str, Any]]:
        """Find exact point-to-segment candidates via a densified-segment KD-tree."""
        hit_samples = self.tree.query_ball_point(point_xy, radius_m + self.sample_spacing_m / 2 + 1e-8)
        if not hit_samples:
            return []
        edge_ids = np.unique(self.sample_edge_ids[np.asarray(hit_samples, dtype=np.int64)])
        starts, ends = self.start_xy[edge_ids], self.end_xy[edge_ids]
        vectors = ends - starts
        denom = np.einsum("ij,ij->i", vectors, vectors)
        t = np.divide(np.einsum("ij,ij->i", point_xy - starts, vectors), denom,
                      out=np.zeros_like(denom), where=denom > 0)
        t = np.clip(t, 0.0, 1.0)
        snapped = starts + t[:, None] * vectors
        distances = np.linalg.norm(snapped - point_xy, axis=1)
        keep = distances <= radius_m
        edge_ids, t, distances = edge_ids[keep], t[keep], distances[keep]
        order = np.lexsort((edge_ids, distances))
        if max_candidates is not None:
            order = order[:max_candidates]
        return [{"edge_id": int(edge_ids[i]), "distance_m": float(distances[i]),
                 "rank": rank, "along_fraction": float(t[i]),
                 "arc_length_m": float(self.lengths_m[edge_ids[i]]),
                 "bearing_deg": float(self.bearings_deg[edge_ids[i]]),
                 "from_degree": int(self.node_degree[self.arcs[edge_ids[i], 0]]),
                 "to_degree": int(self.node_degree[self.arcs[edge_ids[i], 1]])}
                for rank, i in enumerate(order)]


def point_heading_and_speed(previous_xy: np.ndarray, current_xy: np.ndarray,
                            elapsed_s: float) -> tuple[float | None, float | None, float]:
    if elapsed_s <= 0:
        return None, None, 0.0
    dx, dy = current_xy - previous_xy
    distance = sqrt(float(dx*dx + dy*dy))
    if distance == 0:
        return None, 0.0, distance
    return (degrees(np.arctan2(dx, dy)) + 360.0) % 360.0, distance / elapsed_s, distance


def heading_delta(a: float | None, b: float) -> float:
    if a is None:
        return 0.0
    return abs((a - b + 180.0) % 360.0 - 180.0)
