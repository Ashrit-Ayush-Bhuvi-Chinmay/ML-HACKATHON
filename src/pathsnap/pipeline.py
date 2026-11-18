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
from .evaluate import best_per_point
from .features import pair_features, prepare_trip
from .hmm import HMMParams, RoadGraph, build_route, plausible_continuity, viterbi
from .io import load_route, load_trip
from .labels import label_trip

ARTIFACT_PATH = config.REPO_ROOT / "artifacts" / "pathsnap_model.joblib"
MODELS_DIR = config.REPO_ROOT / "artifacts" / "models"
_BUNDLE: dict | None = None
_MODELS: dict[str, dict] = {}


def load_bundle(path: Path | None = None) -> dict:
    """The exported final model (scripts/06_export.py)."""
    global _BUNDLE
    if _BUNDLE is None or path is not None:
        _BUNDLE = joblib.load(path or ARTIFACT_PATH)
    return _BUNDLE


def load_model(name: str | None = None) -> dict:
    """A model bundle by name: None / "default" = the final model, otherwise any model saved by
    the comparison (artifacts/models/<name>.joblib), including the weaker ones."""
    if name in (None, "default"):
        return load_bundle()
    if name not in _MODELS:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise KeyError(f"unknown model {name!r}")
        _MODELS[name] = joblib.load(path)
    return _MODELS[name]


def list_models() -> list[dict]:
    """Leaderboard of every saved model with its recorded CV and student-test metrics."""
    import json

    out = []
    if ARTIFACT_PATH.exists():
        b = load_bundle()
        out.append({"name": "default", "model": b["model_name"], "description": "final model + HMM",
                    "hmm_params": b["hmm_params"], **b.get("metrics", {})})
    reg_path = MODELS_DIR / "registry.json"
    if reg_path.exists():
        for name, info in json.loads(reg_path.read_text()).items():
            out.append({"name": name, **info})
    return out


def experiment_history() -> dict:
    """Past runs for the UI: model comparisons (current and earlier feature set), HMM ablations
    and Bayesian-optimisation trials. Everything comes from files in results/."""
    import pandas as pd

    def table(path):
        return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records") if path.exists() else []

    res = config.RESULTS_DIR
    hist = {
        "model_comparison": table(res / "models" / "model_comparison.csv"),
        "model_comparison_featureset_v1": table(res / "models" / "model_comparison_featureset_v1.csv"),
        "hmm_ablation": {p.stem.removeprefix("summary_"): table(p) for p in sorted((res / "hmm").glob("summary_*.csv"))},
        "tuning_trials": {p.stem.removeprefix("trials_"): table(p) for p in sorted((res / "tuning").glob("trials_*.csv"))},
    }
    return hist


def _score_candidates(bundle: dict, feats, obs: np.ndarray) -> np.ndarray:
    if len(feats) == 0:  # no point has a road within the candidate radius
        return np.zeros(0)
    model = bundle["model"]
    if bundle.get("kind") == "distance_gaussian":
        return model.predict_proba_grouped(feats["dist"].to_numpy(), obs.astype(np.int64))
    X = feats[bundle["features"]].to_numpy()
    if hasattr(model, "predict_proba_grouped"):
        return model.predict_proba_grouped(X, obs.astype(np.int64))
    return model.predict_proba(X)[:, 1]


def predict_trip(trip_dir, bundle: dict | None = None, model: str | None = None, use_hmm: bool = True) -> dict:
    """Run the full pipeline on one trip. `model` picks any saved model (default: the final one);
    `use_hmm=False` returns the classifier-only route, for side-by-side comparison."""
    b = bundle or load_model(model)
    hmm_params = b.get("hmm_params") or (load_bundle()["hmm_params"] if ARTIFACT_PATH.exists()
                                         else HMMParams().as_dict())
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
    prob = _score_candidates(b, feats, feats["obs"].to_numpy())
    tm["classifier"] = time.perf_counter() - t
    t = time.perf_counter()
    params = HMMParams(**hmm_params)
    if use_hmm:
        rows, graph = viterbi(ctx, prob, params)
    else:  # classifier only: best candidate of each point independently
        graph = RoadGraph(ctx, params.sp_limit)
        rows = np.full(trip.n_points, -1, dtype=np.int64)
        if len(ctx.cands.obs):
            k, idx = best_per_point(ctx.cands.obs, prob)
            rows[k] = idx
    pieces = build_route(ctx, graph, rows)
    tm["hmm" if use_hmm else "route"] = time.perf_counter() - t
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
        "hmm": use_hmm,
    }


def score_trip(trip_dir, prediction: dict) -> dict:
    """Server-side evaluation against the trip's .route. Never call this before predicting."""
    trip_dir = Path(trip_dir)
    trip = load_trip(trip_dir)
    try:
        ref_route = load_route(trip_dir)
    except FileNotFoundError:
        return {"top1_accuracy": None, "route_precision": None, "route_recall": None, "route_f1": None,
                "route_mismatch": None, "route_continuity": prediction.get("route_continuity"),
                "n_points_scored": 0, "note": "no reference route (.route) for this trip; prediction only"}
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
    out = {
        "top1_accuracy": top1,
        "route_precision": prec,
        "route_recall": rec,
        "route_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "route_mismatch": (L(pred_set - ref_set) + L(ref_set - pred_set)) / max(L(ref_set), 1e-9),
        "route_continuity": prediction.get("route_continuity"),
        "n_points_scored": int(rel.sum()),
        "reference_median_dist_m": float(np.median(lab["dist"])),
    }
    # JSON has no NaN/inf: report "not computable" as null instead of crashing the API response
    for k, v in out.items():
        if isinstance(v, float):
            out[k] = round(v, 4) if np.isfinite(v) else None
    if out["reference_median_dist_m"] is not None and out["reference_median_dist_m"] > config.TRIP_MAX_MEDIAN_DIST:
        out["note"] = "reference route lies far from the GPS track; scores for this trip are unreliable"
    return out
