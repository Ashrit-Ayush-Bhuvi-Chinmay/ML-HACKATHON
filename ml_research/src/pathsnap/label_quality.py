"""Compare the existing route-label heuristic with a route-state Viterbi alignment.

The alternative labels are written to a separate experiment directory. This module
never edits feature caches or replaces labels in the production feature pipeline.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import TripData, list_trips, read_trip
from .feature_pipeline import TripFeatureSet
from .feature_store import load_trip_feature_set
from .geo import RoadIndex
from .route_labels import project_track

DEFAULT_SKIP_PENALTY_M = 15.0
DEFAULT_BACKWARD_PENALTY_M = 150.0
MAP_TRIPS = {"00000033", "00000054", "00000066"}


def _route_emissions(track_xy: np.ndarray, trip: TripData, road: RoadIndex,
                     chunk_size: int = 512) -> np.ndarray:
    """Return point-to-route-arc distances in metres, one column per route position."""
    route_edges = trip.route
    starts = road.start_xy[route_edges]
    vectors = road.end_xy[route_edges] - starts
    denom = np.einsum("ij,ij->i", vectors, vectors)
    distances = np.empty((len(track_xy), len(route_edges)), dtype=np.float32)
    for offset in range(0, len(track_xy), chunk_size):
        stop = min(len(track_xy), offset + chunk_size)
        delta = track_xy[offset:stop, None, :] - starts[None, :, :]
        numer = np.einsum("bld,ld->bl", delta, vectors)
        fraction = np.divide(numer, denom[None, :], out=np.zeros_like(numer),
                             where=denom[None, :] > 0)
        np.clip(fraction, 0.0, 1.0, out=fraction)
        residual = delta - fraction[:, :, None] * vectors[None, :, :]
        distances[offset:stop] = np.sqrt(np.einsum("bld,bld->bl", residual, residual))
    return distances


def _prefix_argmin(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.int32)
    best = 0
    for index in range(len(values)):
        if values[index] < values[best]:
            best = index
        result[index] = best
    return result


def _suffix_argmin(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.int32)
    best = len(values) - 1
    for index in range(len(values) - 1, -1, -1):
        if values[index] < values[best]:
            best = index
        result[index] = best
    return result


def assign_route_labels_viterbi(
    track_xy: np.ndarray,
    route_edges: np.ndarray,
    emissions_m: np.ndarray,
    *,
    skip_penalty_m: float = DEFAULT_SKIP_PENALTY_M,
    backward_penalty_m: float = DEFAULT_BACKWARD_PENALTY_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align observations to ordered route positions using a Viterbi dynamic program.

    Staying on the same route position or advancing to its next position has no
    transition cost. Skipping intermediate positions costs ``skip_penalty_m`` per
    skipped arc. Moving backward costs ``backward_penalty_m`` per route position.
    The first point pays the skip cost for any route positions before its state.
    """
    if len(route_edges) == 0:
        raise ValueError("Cannot align points to an empty route")
    if emissions_m.shape != (len(track_xy), len(route_edges)):
        raise ValueError("Emission matrix shape must be (point count, route position count)")
    if skip_penalty_m < 0 or backward_penalty_m < 0:
        raise ValueError("Transition penalties must be nonnegative")
    if len(track_xy) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)

    point_count, state_count = emissions_m.shape
    route_index = np.arange(state_count, dtype=np.float64)
    skip_adjusted = float(skip_penalty_m) * route_index
    back_adjusted = float(backward_penalty_m) * route_index
    backpointers = np.empty((point_count, state_count), dtype=np.int32)
    backpointers[0] = -1
    previous = emissions_m[0].astype(np.float64) + skip_adjusted

    for point_index in range(1, point_count):
        prefix = _prefix_argmin(previous - skip_adjusted)
        suffix = _suffix_argmin(previous + back_adjusted)
        current = np.empty(state_count, dtype=np.float64)
        for state in range(state_count):
            best_parent = state
            best_cost = previous[state]  # stay on this route arc

            if state > 0 and previous[state - 1] < best_cost:
                best_parent = state - 1  # advance to the next route arc
                best_cost = previous[best_parent]

            if state > 1:
                parent = int(prefix[state - 2])
                cost = previous[parent] + skip_penalty_m * (state - parent - 1)
                if cost < best_cost:
                    best_parent, best_cost = parent, cost

            if state + 1 < state_count:
                parent = int(suffix[state + 1])
                cost = previous[parent] + backward_penalty_m * (parent - state)
                if cost < best_cost:
                    best_parent, best_cost = parent, cost

            current[state] = float(emissions_m[point_index, state]) + best_cost
            backpointers[point_index, state] = best_parent
        previous = current

    states = np.empty(point_count, dtype=np.int64)
    states[-1] = int(np.argmin(previous))
    for point_index in range(point_count - 1, 0, -1):
        states[point_index - 1] = backpointers[point_index, states[point_index]]
    labels = np.asarray(route_edges, dtype=np.int64)[states]
    distances = emissions_m[np.arange(point_count), states].astype(np.float64)
    return labels, states, distances


def _distance_to_edges(track_xy: np.ndarray, edge_ids: np.ndarray,
                       road: RoadIndex) -> np.ndarray:
    starts = road.start_xy[edge_ids]
    vectors = road.end_xy[edge_ids] - starts
    denom = np.einsum("ij,ij->i", vectors, vectors)
    numer = np.einsum("ij,ij->i", track_xy - starts, vectors)
    fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    np.clip(fraction, 0.0, 1.0, out=fraction)
    residual = track_xy - (starts + fraction[:, None] * vectors)
    return np.linalg.norm(residual, axis=1)


def _trip_summary(trip: TripData, cache: TripFeatureSet, road: RoadIndex,
                  track_xy: np.ndarray, skip_penalty_m: float,
                  backward_penalty_m: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    current_edges = cache.point_true_edges.astype(np.int64)
    current_distances = _distance_to_edges(track_xy, current_edges, road)
    emissions = _route_emissions(track_xy, trip, road)
    alternative_edges, route_positions, alternative_distances = assign_route_labels_viterbi(
        track_xy, trip.route, emissions,
        skip_penalty_m=skip_penalty_m,
        backward_penalty_m=backward_penalty_m,
    )
    offsets = cache.point_offsets
    nearest_edges = np.full(cache.point_count, -1, dtype=np.int64)
    for point_index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if stop > start:
            nearest_edges[point_index] = cache.candidate_edge_ids[start]

    row = {
        "split": trip.split,
        "trip_id": trip.trip_id,
        "point_count": len(track_xy),
        "route_positions": len(trip.route),
        "current_median_distance_m": float(np.median(current_distances)),
        "current_p95_distance_m": float(np.percentile(current_distances, 95)),
        "viterbi_median_distance_m": float(np.median(alternative_distances)),
        "viterbi_p95_distance_m": float(np.percentile(alternative_distances, 95)),
        "median_improved": bool(np.median(alternative_distances) < np.median(current_distances)),
        "p95_improved": bool(np.percentile(alternative_distances, 95) < np.percentile(current_distances, 95)),
        "both_percentiles_improved": bool(
            np.median(alternative_distances) < np.median(current_distances)
            and np.percentile(alternative_distances, 95) < np.percentile(current_distances, 95)
        ),
        "current_candidate_miss_rate": float(np.mean(cache.point_candidate_misses)),
        "viterbi_backward_steps": int(np.sum(np.diff(route_positions) < 0)),
        "viterbi_skipped_route_positions": int(np.sum(np.maximum(0, np.diff(route_positions) - 1))),
        "skip_penalty_m": float(skip_penalty_m),
        "backward_penalty_m": float(backward_penalty_m),
        "epsg": cache.epsg,
    }
    return row, alternative_edges, route_positions, nearest_edges


def _svg_map(trip: TripData, cache: TripFeatureSet, road: RoadIndex,
             track_xy: np.ndarray, nearest_edges: np.ndarray, output: Path) -> None:
    """Write a dependency-free, metric-coordinate two-panel map for visual review."""
    current_ids = np.unique(cache.point_true_edges)
    current_ids = current_ids[current_ids >= 0]
    nearest_ids = np.unique(nearest_edges)
    nearest_ids = nearest_ids[nearest_ids >= 0]
    selected = np.unique(np.concatenate((current_ids, nearest_ids)))
    endpoints = np.concatenate((road.start_xy[selected], road.end_xy[selected]), axis=0)
    bounds = np.vstack((track_xy, endpoints)) if len(endpoints) else track_xy
    x_min, y_min = np.min(bounds, axis=0)
    x_max, y_max = np.max(bounds, axis=0)
    pad_x = max((x_max - x_min) * 0.04, 40.0)
    pad_y = max((y_max - y_min) * 0.04, 40.0)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    width, height = 1280, 760
    panel_width, panel_height = 610, 650
    top, left = 82, 26

    def transform(points: np.ndarray, x0: float) -> str:
        sx = (panel_width - 34) / max(x_max - x_min, 1e-9)
        sy = (panel_height - 34) / max(y_max - y_min, 1e-9)
        scale = min(sx, sy)
        used_w, used_h = (x_max - x_min) * scale, (y_max - y_min) * scale
        ox = x0 + 17 + (panel_width - 34 - used_w) / 2
        oy = top + 17 + (panel_height - 34 - used_h) / 2
        xs = ox + (points[:, 0] - x_min) * scale
        ys = oy + (y_max - points[:, 1]) * scale
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    def arc_lines(edge_ids: np.ndarray, x0: float, color: str) -> list[str]:
        lines = []
        for edge_id in edge_ids:
            pts = np.vstack((road.start_xy[edge_id], road.end_xy[edge_id]))
            coords = transform(pts, x0).split()
            (x1, y1), (x2, y2) = (coords[0].split(","), coords[1].split(","))
            lines.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{color}" stroke-width="2.2" stroke-opacity="0.76" '
                'stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
            )
        return lines

    gps_left = transform(track_xy, left)
    gps_right = transform(track_xy, left + panel_width + 14)
    current_panel = arc_lines(current_ids, left, "#dc2626")
    nearest_panel = arc_lines(nearest_ids, left + panel_width + 14, "#16a34a")
    safe_id = html.escape(trip.trip_id)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Trip {safe_id} GPS and road-label comparison">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-family="Segoe UI,Arial" font-size="23" font-weight="700" fill="#111827">Trip {safe_id}: GPS and road-label comparison</text>',
        '<text x="24" y="58" font-family="Segoe UI,Arial" font-size="13" fill="#4b5563">Projected UTM metres; no basemap. The two panels use the same extent.</text>',
        f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="#f9fafb" stroke="#d1d5db"/>',
        f'<rect x="{left+panel_width+14}" y="{top}" width="{panel_width}" height="{panel_height}" fill="#f9fafb" stroke="#d1d5db"/>',
        f'<text x="{left+12}" y="{top-12}" font-family="Segoe UI,Arial" font-size="16" font-weight="700" fill="#111827">Assigned labels ({len(current_ids)} unique arcs)</text>',
        f'<text x="{left+panel_width+26}" y="{top-12}" font-family="Segoe UI,Arial" font-size="16" font-weight="700" fill="#111827">Nearest-arc predictions ({len(nearest_ids)} unique arcs)</text>',
        *current_panel,
        *nearest_panel,
        f'<polyline points="{gps_left}" fill="none" stroke="#2563eb" stroke-width="1.7" stroke-opacity="0.9" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
        f'<polyline points="{gps_right}" fill="none" stroke="#2563eb" stroke-width="1.7" stroke-opacity="0.9" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>',
        '<line x1="30" y1="746" x2="58" y2="746" stroke="#2563eb" stroke-width="3"/><text x="65" y="751" font-family="Segoe UI,Arial" font-size="13" fill="#111827">GPS trace</text>',
        '<line x1="160" y1="746" x2="188" y2="746" stroke="#dc2626" stroke-width="3"/><text x="195" y="751" font-family="Segoe UI,Arial" font-size="13" fill="#111827">Current route-assigned arcs</text>',
        '<line x1="420" y1="746" x2="448" y2="746" stroke="#16a34a" stroke-width="3"/><text x="455" y="751" font-family="Segoe UI,Arial" font-size="13" fill="#111827">Nearest candidate arcs</text>',
        '</svg>',
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content), encoding="utf-8")


def _save_map(trip: TripData, cache: TripFeatureSet, road: RoadIndex,
              track_xy: np.ndarray, nearest_edges: np.ndarray, map_dir: Path) -> dict[str, str]:
    """Prefer Folium when installed; retain an SVG view when the package is unavailable."""
    map_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trip-{int(trip.trip_id):02d}-label-vs-nearest"
    try:
        import folium
    except ImportError:
        svg_path = map_dir / f"{stem}.svg"
        _svg_map(trip, cache, road, track_xy, nearest_edges, svg_path)
        return {"status": "svg_fallback_folium_unavailable", "path": str(svg_path)}

    center = [float(np.median(trip.track[:, 1])), float(np.median(trip.track[:, 0]))]
    fmap = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    gps_layer = folium.FeatureGroup(name="GPS trace", show=True)
    label_layer = folium.FeatureGroup(name="Current assigned label arcs", show=True)
    nearest_layer = folium.FeatureGroup(name="Nearest-arc predictions", show=True)
    stride = max(1, int(math.ceil(len(trip.track) / 5000)))
    points = trip.track[::stride]
    if len(points) == 0 or not np.array_equal(points[-1], trip.track[-1]):
        points = np.vstack((points, trip.track[-1]))
    folium.PolyLine([(float(lat), float(lon)) for lon, lat in points[:, :2]],
                    color="#2563eb", weight=4, opacity=0.9, tooltip="GPS trace").add_to(gps_layer)

    for edge_id in np.unique(cache.point_true_edges):
        if edge_id < 0:
            continue
        nodes = trip.nodes[trip.arcs[int(edge_id)]]
        folium.PolyLine([(float(lat), float(lon)) for lon, lat in nodes],
                        color="#dc2626", weight=3, opacity=0.78,
                        tooltip=f"Assigned arc {int(edge_id)}").add_to(label_layer)
    for edge_id in np.unique(nearest_edges):
        if edge_id < 0:
            continue
        nodes = trip.nodes[trip.arcs[int(edge_id)]]
        folium.PolyLine([(float(lat), float(lon)) for lon, lat in nodes],
                        color="#16a34a", weight=2, opacity=0.72,
                        tooltip=f"Nearest arc {int(edge_id)}").add_to(nearest_layer)

    for layer in (gps_layer, label_layer, nearest_layer):
        layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    html_path = map_dir / f"{stem}.html"
    fmap.save(str(html_path))
    return {"status": "folium", "path": str(html_path)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def run_label_quality_comparison(
    archive_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    map_dir: str | Path,
    *,
    skip_penalty_m: float = DEFAULT_SKIP_PENALTY_M,
    backward_penalty_m: float = DEFAULT_BACKWARD_PENALTY_M,
) -> dict[str, Any]:
    """Compare all available trips, saving alternative labels separately per trip."""
    archive_path, cache_dir = Path(archive_path), Path(cache_dir)
    output_dir, map_dir = Path(output_dir), Path(map_dir)
    label_dir = output_dir / "alternative_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "viterbi_label_comparison.csv"
    json_path = output_dir / "viterbi_label_comparison.json"
    fields = [
        "split", "trip_id", "point_count", "route_positions",
        "current_median_distance_m", "current_p95_distance_m",
        "viterbi_median_distance_m", "viterbi_p95_distance_m",
        "median_improved", "p95_improved", "both_percentiles_improved",
        "current_candidate_miss_rate", "viterbi_backward_steps",
        "viterbi_skipped_route_positions", "skip_penalty_m", "backward_penalty_m", "epsg",
    ]
    refs = list_trips(archive_path)
    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite label comparison: {csv_path}")
    rows: list[dict[str, Any]] = []
    maps: dict[str, dict[str, str]] = {}
    errors: list[dict[str, str]] = []

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for ref in refs:
            try:
                trip = read_trip(archive_path, ref)
                cache = load_trip_feature_set(cache_dir / ref.split / f"{ref.trip_id}.npz")
                track_xy, zone, epsg = project_track(trip)
                road = RoadIndex.build(trip.nodes, trip.arcs, zone)
                row, alt_edges, states, nearest_edges = _trip_summary(
                    trip, cache, road, track_xy, skip_penalty_m, backward_penalty_m)
                row["epsg"] = epsg
                rows.append(row)
                trip_label_dir = label_dir / ref.split
                trip_label_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    trip_label_dir / f"{ref.trip_id}.npz",
                    alternative_point_true_edges=alt_edges,
                    alternative_route_positions=states,
                )
                writer.writerow(row)
                stream.flush()
                if ref.trip_id in MAP_TRIPS:
                    maps[ref.trip_id] = _save_map(trip, cache, road, track_xy, nearest_edges, map_dir)
                train_rows = [item for item in rows if item["split"] == "train"]
                student_rows = [item for item in rows if item["split"] == "student_test"]
                summary = {
                    "method": "route-position Viterbi; states preserve route sequence positions",
                    "transition_cost": {
                        "stay_or_advance_one": 0.0,
                        "skip_penalty_m_per_skipped_route_arc": skip_penalty_m,
                        "backward_penalty_m_per_route_position": backward_penalty_m,
                        "initial_route_position_skip_penalty_m": skip_penalty_m,
                    },
                    "seed": 42,
                    "cache_policy": "diagnostic alternatives only; current labels and caches were not replaced",
                    "completed_trip_count": len(rows),
                    "training_trip_count": len(train_rows),
                    "training_trips_improved_median": sum(r["median_improved"] for r in train_rows),
                    "training_trips_improved_p95": sum(r["p95_improved"] for r in train_rows),
                    "training_trips_improved_both": sum(r["both_percentiles_improved"] for r in train_rows),
                    "student_test_trip_count_diagnostic_only": len(student_rows),
                    "student_test_trips_improved_both_diagnostic_only": sum(r["both_percentiles_improved"] for r in student_rows),
                    "maps": maps,
                    "errors": errors,
                    "per_trip": rows,
                }
                _atomic_json(json_path, summary)
                print(f"saved {ref.split}/{ref.trip_id} ({len(rows)}/{len(refs)})", flush=True)
            except Exception as exc:  # Preserve progress and make failures visible in the log.
                error = {"split": ref.split, "trip_id": ref.trip_id, "error": f"{type(exc).__name__}: {exc}"}
                errors.append(error)
                _atomic_json(json_path, {"completed_trip_count": len(rows), "per_trip": rows,
                                         "maps": maps, "errors": errors})
                print(f"ERROR {ref.split}/{ref.trip_id}: {error['error']}", flush=True)

    return json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {
        "completed_trip_count": 0, "per_trip": [], "maps": maps, "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/raw/Student folder.zip")
    parser.add_argument("--cache-dir", default="data/interim/features")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--map-dir", default="reports/trip_diagnostics/maps")
    parser.add_argument("--skip-penalty-m", type=float, default=DEFAULT_SKIP_PENALTY_M)
    parser.add_argument("--backward-penalty-m", type=float, default=DEFAULT_BACKWARD_PENALTY_M)
    args = parser.parse_args()
    result = run_label_quality_comparison(
        args.archive, args.cache_dir, args.output_dir, args.map_dir,
        skip_penalty_m=args.skip_penalty_m,
        backward_penalty_m=args.backward_penalty_m,
    )
    print(json.dumps({
        "completed_trip_count": result.get("completed_trip_count"),
        "training_trips_improved_both": result.get("training_trips_improved_both"),
        "student_test_trips_improved_both_diagnostic_only": result.get(
            "student_test_trips_improved_both_diagnostic_only"),
        "errors": result.get("errors", []),
    }, indent=2))


if __name__ == "__main__":
    main()
