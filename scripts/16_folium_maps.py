"""Interactive Folium maps for the report (brief section 6), from the real pipeline output.

For a student-test trip: original GPS points, nearby road-network geometry, the reference route
(student-test labels are supplied; never done for evaluator trips), the final predicted route
(classifier + HMM) and the classifier-only route, with a legend and a layer control.
Usage: python scripts/16_folium_maps.py [--trips 00000009 00000077]
Output: results/maps/trip_<id>.html
"""
from __future__ import annotations

import argparse

import folium
import numpy as np

from pathsnap import config
from pathsnap.geo import Projector, point_segment
from pathsnap.io import load_route, load_trip
from pathsnap.pipeline import predict_trip, score_trip

COL = {"road": "#9a9993", "gps": "#2a78d6", "reference": "#1baf7a", "final": "#eb6834", "clf": "#e34948"}


def arc_lines(trip, arc_ids):
    """[[lat, lon], [lat, lon]] per arc (folium uses lat, lon)."""
    a, b = trip.arcs[arc_ids, 0], trip.arcs[arc_ids, 1]
    n = trip.nodes
    return [[[n[u, 1], n[u, 0]], [n[v, 1], n[v, 0]]] for u, v in zip(a, b)]


def route_lines(pred):
    feat = [f for f in pred["geojson"]["features"] if f["properties"]["layer"] == "predicted_route"][0]
    return [[[lat, lon] for lon, lat in line] for line in feat["geometry"]["coordinates"]]


def make_map(trip_id: str) -> None:
    td = config.DATA_ROOT / "student_test" / trip_id
    trip, ref = load_trip(td), load_route(td)
    final, clf = predict_trip(td), predict_trip(td, use_hmm=False)
    m_final, m_clf = score_trip(td, final), score_trip(td, clf)

    lat, lon = trip.track[:, 1], trip.track[:, 0]
    fmap = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14,
                      tiles="OpenStreetMap", control_scale=True)

    # road network near the track (arcs within 150 m of some GPS point, sampled for size)
    proj = Projector.for_track(trip.track)
    px, py = proj.to_xy(lon, lat)
    nx, ny = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    ax, ay, bx, by = nx[trip.arcs[:, 0]], ny[trip.arcs[:, 0]], nx[trip.arcs[:, 1]], ny[trip.arcs[:, 1]]
    keep = np.zeros(len(trip.arcs), dtype=bool)
    for i in range(0, len(px), 5):
        d, _ = point_segment(px[i], py[i], ax, ay, bx, by)
        keep |= d < 150
    road = folium.FeatureGroup(name="Road network (near the track)")
    folium.PolyLine(arc_lines(trip, np.flatnonzero(keep)), color=COL["road"], weight=1, opacity=0.7).add_to(road)
    road.add_to(fmap)

    gps = folium.FeatureGroup(name="GPS observations")
    for la, lo in zip(lat[::2], lon[::2]):
        folium.CircleMarker([la, lo], radius=2, color=COL["gps"], fill=True, fill_opacity=0.8, weight=0).add_to(gps)
    gps.add_to(fmap)

    refg = folium.FeatureGroup(name="Reference route (student-test label)")
    folium.PolyLine(arc_lines(trip, ref), color=COL["reference"], weight=7, opacity=0.55).add_to(refg)
    refg.add_to(fmap)

    clfg = folium.FeatureGroup(name="Classifier only (no HMM)", show=False)
    for line in route_lines(clf):
        folium.PolyLine(line, color=COL["clf"], weight=2.5, opacity=0.9, dash_array="6 6").add_to(clfg)
    clfg.add_to(fmap)

    fin = folium.FeatureGroup(name="Predicted route (final: classifier + HMM)")
    for line in route_lines(final):
        folium.PolyLine(line, color=COL["final"], weight=3, opacity=0.95).add_to(fin)
    fin.add_to(fmap)

    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    legend = f"""
    <div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999; background: white; padding: 8px 10px;
                border: 1px solid #ccc; border-radius: 6px; font: 12px Arial; line-height: 1.5">
      <b>Student-test trip {trip_id}</b> ({trip.n_points} GPS points)<br>
      <span style="color:{COL['gps']}">&#9679;</span> GPS observations<br>
      <span style="color:{COL['reference']}; font-weight:bold">&#9644;</span> reference route<br>
      <span style="color:{COL['final']}; font-weight:bold">&#9644;</span> predicted route (classifier + HMM):
        Top-1 {pct(m_final['top1_accuracy'])}, sequence F1 {pct(m_final['sequence_f1'])}<br>
      <span style="color:{COL['clf']}">- - -</span> classifier only (layer off by default):
        Top-1 {pct(m_clf['top1_accuracy'])}, sequence F1 {pct(m_clf['sequence_f1'])}<br>
      <span style="color:{COL['road']}">&#9644;</span> road network near the track
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend))
    out = config.RESULTS_DIR / "maps"
    out.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out / f"trip_{trip_id}.html"))
    print(f"wrote maps/trip_{trip_id}.html  final top1 {pct(m_final['top1_accuracy'])}  clf {pct(m_clf['top1_accuracy'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+", default=["00000009", "00000052"])
    for t in ap.parse_args().trips:
        make_map(t)
