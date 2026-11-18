"""Folium map exports for observed tracks, candidate roads, and predictions."""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Iterable, Mapping


def save_route_map(points: Iterable[Mapping], output: str | Path, title: str = "PathSnap route") -> Path:
    try:
        import folium
    except ImportError as exc:
        raise RuntimeError("Install folium to create route maps.") from exc
    points = list(points)
    if not points:
        raise ValueError("Cannot map an empty point sequence.")
    center = [sum(float(p["latitude"]) for p in points)/len(points),
              sum(float(p["longitude"]) for p in points)/len(points)]
    m = folium.Map(location=center, zoom_start=14, control_scale=True)
    folium.PolyLine([(float(p["latitude"]), float(p["longitude"])) for p in points],
                    color="#2563eb", weight=4, tooltip=title).add_to(m)
    folium.LayerControl().add_to(m)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    return output


def save_trip_diagnostic_map(trip, feature_set, output: str | Path) -> Path:
    """Create a Folium overlay of GPS, assigned route labels, and nearest-arc predictions."""
    try:
        import folium
    except ImportError as exc:
        raise RuntimeError("Install folium in the active Python environment before making trip maps.") from exc
    import numpy as np

    track = [(float(lat), float(lon)) for lon, lat in trip.track[:, :2]]
    if not track:
        raise ValueError(f"Trip {trip.trip_id} has no GPS points.")
    label_counts = Counter(map(int, feature_set.point_true_edges))
    predicted_edges = []
    predicted_counts = Counter()
    errors = []
    misses = []
    for point_idx, true_edge in enumerate(feature_set.point_true_edges):
        start = int(feature_set.point_offsets[point_idx])
        stop = int(feature_set.point_offsets[point_idx + 1])
        if start == stop:
            predicted_edges.append(-1)
            continue
        nearest_row = start + int(np.argmin(feature_set.candidate_distances_m[start:stop]))
        predicted_edge = int(feature_set.candidate_edge_ids[nearest_row])
        predicted_edges.append(predicted_edge)
        predicted_counts[predicted_edge] += 1
        present = bool(np.any(feature_set.candidate_edge_ids[start:stop] == int(true_edge)))
        if predicted_edge != int(true_edge):
            errors.append((point_idx, present))
            if not present:
                misses.append(point_idx)

    center = [float(np.median(trip.track[:, 1])), float(np.median(trip.track[:, 0]))]
    map_view = folium.Map(location=center, zoom_start=14, control_scale=True,
                          tiles="OpenStreetMap")
    gps_group = folium.FeatureGroup(name="GPS observations")
    folium.PolyLine(track, color="#2563eb", weight=3.2, opacity=0.8,
                    tooltip="Observed GPS trace").add_to(gps_group)
    gps_group.add_to(map_view)

    def edge_geojson(counts: Counter, label: str) -> dict:
        features = []
        for edge_id, point_count in counts.items():
            u, v = map(int, trip.arcs[edge_id])
            lon1, lat1 = map(float, trip.nodes[u])
            lon2, lat2 = map(float, trip.nodes[v])
            features.append({"type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
                "properties": {"edge_id": edge_id, "from_node": u, "to_node": v,
                               "point_count": int(point_count), "layer": label}})
        return {"type": "FeatureCollection", "features": features}

    label_group = folium.FeatureGroup(name="Assigned reference-route arcs", show=True)
    folium.GeoJson(edge_geojson(label_counts, "reference label"),
        style_function=lambda _: {"color": "#16a34a", "weight": 3, "opacity": 0.78},
        tooltip=folium.GeoJsonTooltip(fields=["edge_id", "from_node", "to_node", "point_count"],
            aliases=["Arc ID", "From node", "To node", "GPS points assigned"])).add_to(label_group)
    label_group.add_to(map_view)

    prediction_group = folium.FeatureGroup(name="Nearest-arc predictions", show=True)
    folium.GeoJson(edge_geojson(predicted_counts, "nearest prediction"),
        style_function=lambda _: {"color": "#dc2626", "weight": 2, "opacity": 0.52},
        tooltip=folium.GeoJsonTooltip(fields=["edge_id", "from_node", "to_node", "point_count"],
            aliases=["Arc ID", "From node", "To node", "GPS points predicted"])).add_to(prediction_group)
    prediction_group.add_to(map_view)

    error_group = folium.FeatureGroup(name="Baseline errors (sampled)", show=False)
    marker_stride = max(1, len(track) // 2500)
    for point_idx, true_arc_present in errors:
        if point_idx % marker_stride:
            continue
        folium.CircleMarker(track[point_idx], radius=2.2,
            color="#f59e0b" if true_arc_present else "#111827",
            fill=True, fill_opacity=0.75,
            tooltip=f"Point {point_idx}: baseline wrong; true arc in candidates={true_arc_present}").add_to(error_group)
    error_group.add_to(map_view)
    if misses:
        miss_group = folium.FeatureGroup(name="Candidate misses", show=False)
        for point_idx in misses:
            folium.CircleMarker(track[point_idx], radius=3, color="#111827",
                fill=True, fill_color="#111827", fill_opacity=0.8,
                tooltip=f"Point {point_idx}: assigned route arc absent from candidates").add_to(miss_group)
        miss_group.add_to(map_view)

    folium.Marker(track[0], tooltip="Trip start",
                  icon=folium.Icon(color="green", icon="play")).add_to(map_view)
    folium.Marker(track[-1], tooltip="Trip end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(map_view)
    map_view.fit_bounds([[min(p[0] for p in track), min(p[1] for p in track)],
                         [max(p[0] for p in track), max(p[1] for p in track)]])
    folium.LayerControl(collapsed=False).add_to(map_view)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(str(output))
    return output
