"""Map chronological route arc IDs to observations with a monotone nearest-arc heuristic."""

from __future__ import annotations

import numpy as np

from .dataset import TripData
from .geo import RoadIndex, project_utm, utm_zone


def project_track(trip: TripData) -> tuple[np.ndarray, int, str]:
    median_lon = float(np.median(trip.track[:, 0]))
    median_lat = float(np.median(trip.track[:, 1]))
    zone = utm_zone(median_lon)
    xy = np.asarray([project_utm(lon, lat, zone)[:2] for lon, lat in trip.track[:, :2]], dtype=np.float64)
    epsg = f"EPSG:{(32700 if median_lat < 0 else 32600) + zone}"
    return xy, zone, epsg


def assign_route_labels(trip: TripData, road: RoadIndex, track_xy: np.ndarray,
                        gap_reset_seconds: float = 30.0,
                        normal_lookahead_arcs: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign one supplied route arc to each GPS point without moving backward in route order.

    For a normal sampling interval, search a bounded forward window. After a long GPS gap,
    search the remaining route because multiple arcs may have been traversed unseen. This is
    an explicit heuristic: reference routes are sequences, not point-aligned ground truth.
    """
    if len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no reference route arcs")
    labels = np.empty(len(track_xy), dtype=np.int64)
    route_positions = np.empty(len(track_xy), dtype=np.int64)
    distances = np.empty(len(track_xy), dtype=np.float64)
    cursor = 0
    median_route_length = max(5.0, float(np.median(road.lengths_m[trip.route])))
    for i, point in enumerate(track_xy):
        elapsed = float(trip.track[i, 2] - trip.track[i-1, 2]) if i else 0.0
        step_m = float(np.linalg.norm(point - track_xy[i-1])) if i else 0.0
        if i and elapsed > gap_reset_seconds:
            stop = len(trip.route)
            expand_to_suffix = True
        else:
            # Limit route progress on dense, looping paths: unrestricted nearest-future
            # matching can jump to a later pass at an intersection. If the local route
            # window is clearly far from the observation, expand to the remaining suffix.
            expected_edges = int(np.ceil(step_m / median_route_length * 1.5)) + 2
            forward = max(3, min(normal_lookahead_arcs, expected_edges))
            stop = min(len(trip.route), cursor + forward + 1)
        route_positions_window = np.arange(cursor, max(cursor + 1, stop), dtype=np.int64)
        route_positions_window = np.minimum(route_positions_window, len(trip.route)-1)
        edge_ids = trip.route[route_positions_window]
        starts, ends = road.start_xy[edge_ids], road.end_xy[edge_ids]
        vectors = ends - starts
        denom = np.einsum("ij,ij->i", vectors, vectors)
        t = np.divide(np.einsum("ij,ij->i", point - starts, vectors), denom,
                      out=np.zeros_like(denom), where=denom > 0)
        t = np.clip(t, 0.0, 1.0)
        snapped = starts + t[:, None] * vectors
        d = np.linalg.norm(snapped - point, axis=1)
        expand_to_suffix = expand_to_suffix if i and elapsed > gap_reset_seconds else bool(float(d.min()) > 50.0)
        if expand_to_suffix and stop < len(trip.route):
            route_positions_window = np.arange(cursor, len(trip.route), dtype=np.int64)
            edge_ids = trip.route[route_positions_window]
            starts, ends = road.start_xy[edge_ids], road.end_xy[edge_ids]
            vectors = ends - starts
            denom = np.einsum("ij,ij->i", vectors, vectors)
            t = np.divide(np.einsum("ij,ij->i", point - starts, vectors), denom,
                          out=np.zeros_like(denom), where=denom > 0)
            t = np.clip(t, 0.0, 1.0)
            snapped = starts + t[:, None] * vectors
            d = np.linalg.norm(snapped - point, axis=1)
            # A modest progress cost discourages jumping to a later loop at a crossing
            # while allowing large corrections when the current route arc is far away.
            route_advance = route_positions_window - cursor
            best = int(np.argmin(d + route_advance * 15.0))
        else:
            best = int(np.argmin(d))
        cursor = int(route_positions_window[best])
        labels[i], route_positions[i], distances[i] = int(edge_ids[best]), cursor, float(d[best])
    return labels, route_positions, distances
