"""End-to-end inference for one trip. This is what the API calls (see docs/API_CONTRACT.md).

    predict_trip(trip_dir)          -> prediction dict   (reads .track/.nodes/.arcs only)
    score_trip(trip_dir, prediction) -> metrics dict     (server-side; the only reader of .route)

The model bundle (artifacts/pathsnap_model.joblib, written by scripts/06_export.py) holds the
trained classifier and every preprocessing / decoding setting, so training and serving cannot
drift apart.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from . import config
from .features import FEATURES, pair_features, prepare_trip
from .hmm import HMMParams, build_route, plausible_continuity, viterbi
from .io import load_route, load_trip
from .labels import label_trip

ARTIFACT_PATH = config.REPO_ROOT / "artifacts" / "pathsnap_model.joblib"
_BUNDLE: dict | None = None


def load_bundle(path: Path | None = None) -> dict:
    global _BUNDLE
    if _BUNDLE is None or path is not None:
        _BUNDLE = joblib.load(path or ARTIFACT_PATH)
    return _BUNDLE


def _score_candidates(bundle: dict, X: np.ndarray, obs: np.ndarray) -> np.ndarray:
    model = bundle["model"]
    if hasattr(model, "predict_proba_grouped"):
        return model.predict_proba_grouped(X, obs.astype(np.int64))
    return model.predict_proba(X)[:, 1]


def predict_trip(trip_dir, bundle: dict | None = None) -> dict:
    b = bundle or load_bundle()
    trip_dir = Path(trip_dir)
    tm: dict[str, float] = {}
    t = time.perf_counter()

    trip = load_trip(trip_dir)
    tm["load"] = time.perf_counter() - t
    t = time.perf_counter()
    ctx = prepare_trip(trip, b["candidate_radius"], b["candidate_k"])
    tm["candidates"] = time.perf_counter() - t
    t = time.perf_counter()
    feats = pair_features(ctx)
    tm["features"] = time.perf_counter() - t
    t = time.perf_counter()
    prob = _score_candidates(b, feats[b["features"]].to_numpy(), feats["obs"].to_numpy())
    tm["classifier"] = time.perf_counter() - t
    t = time.perf_counter()
    rows, graph = viterbi(ctx, prob, HMMParams(**b["hmm_params"]))
    pieces = build_route(ctx, graph, rows)
    tm["hmm"] = time.perf_counter() - t
    continuity = plausible_continuity(ctx, graph, rows)

    arc_of_row = ctx.cands.arc_id
    predicted = [int(arc_of_row[r]) if r >= 0 else None for r in rows]
    nodes = trip.nodes
    lines = []
    for p in pieces:
        if not p:
            continue
        seq = [int(trip.arcs[p[0], 0])] + [int(trip.arcs[a, 1]) for a in p]
        lines.append([[float(nodes[n, 0]), float(nodes[n, 1])] for n in seq])
    tm["total"] = sum(tm.values())
    return {
        "trip_id": trip.trip_id,
        "n_points": trip.n_points,
        "predicted_arcs": predicted,
        "route_arcs": [a for p in pieces for a in p],
        "geojson": {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"layer": "gps"},
             "geometry": {"type": "MultiPoint", "coordinates": trip.track[:, :2].tolist()}},
            {"type": "Feature", "properties": {"layer": "predicted_route"},
             "geometry": {"type": "MultiLineString", "coordinates": lines}},
        ]},
        "route_continuity": continuity,
        "n_unmatched": int(sum(a is None for a in predicted)),
        "timings_ms": {k: round(v * 1000, 1) for k, v in tm.items()},
        "model": b["model_name"],
    }


def score_trip(trip_dir, prediction: dict) -> dict:
    """Server-side evaluation against the trip's .route. Never call this before predicting."""
    trip_dir = Path(trip_dir)
    trip = load_trip(trip_dir)
    ref_route = load_route(trip_dir)
    lab = label_trip(trip, ref_route)
    rel = lab["dist"] <= config.LABEL_MAX_DIST
    pred = np.array([-1 if a is None else a for a in prediction["predicted_arcs"]], dtype=np.int64)
    top1 = float(np.mean(pred[rel] == lab["arc_id"][rel])) if rel.any() else float("nan")

    pred_set = set(prediction["route_arcs"])
    ref_set = set(int(a) for a in ref_route)
    ids = np.array(sorted(pred_set | ref_set), dtype=np.int64)
    from .geo import Projector
    proj = Projector.for_track(trip.track)
    x1, y1 = proj.to_xy(trip.nodes[trip.arcs[ids, 0], 0], trip.nodes[trip.arcs[ids, 0], 1])
    x2, y2 = proj.to_xy(trip.nodes[trip.arcs[ids, 1], 0], trip.nodes[trip.arcs[ids, 1], 1])
    length = dict(zip(ids.tolist(), np.hypot(x2 - x1, y2 - y1).tolist()))
    L = lambda s: sum(length[a] for a in s)
    inter = L(pred_set & ref_set)
    prec = inter / max(L(pred_set), 1e-9)
    rec = inter / max(L(ref_set), 1e-9)
    return {
        "top1_accuracy": round(top1, 4),
        "route_precision": round(prec, 4),
        "route_recall": round(rec, 4),
        "route_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "route_mismatch": round((L(pred_set - ref_set) + L(ref_set - pred_set)) / max(L(ref_set), 1e-9), 4),
        "route_continuity": round(float(prediction.get("route_continuity", float("nan"))), 4),
        "n_points_scored": int(rel.sum()),
        "reference_median_dist_m": round(float(np.median(lab["dist"])), 1),
    }
