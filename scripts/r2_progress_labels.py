"""Research: labels from the vehicle's SMOOTHED progress along the reference route.

Current labels (pathsnap.labels) assign each point to its nearest route arc in route order; GPS
jitter near a node makes the assignment flip between the two arcs meeting there. Here:
  1. along-route position s_i = start of the aligned arc + t_on_arc * arc length (old alignment)
  2. smooth s over time inside each evidence segment: rolling median (5 points) then isotonic
     regression (s must not decrease)
  3. new label = the route arc containing the smoothed s_i
Label quality (no ground truth exists per point): A->B->A flips, direction agreement at speed.

Referee = route-level metrics (sequence / coverage F1, mismatch), which compare against .route
directly and do not depend on per-point labels. Compared with identical models / decoder:
  old labels: the saved lightgbm_tuned predictions; new labels: same params retrained.
Train evidence: 3-fold trip-grouped OOF on 20 train trips' decoding; then student-test once.
Evaluator set NOT used.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.evaluate import point_keys, top1_accuracy
from pathsnap.features import FEATURES
from pathsnap.geo import Projector, point_segment, track_kinematics
from pathsnap.hmm import HMMParams
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.models import SPECS

spec = importlib.util.spec_from_file_location("r1", Path(__file__).with_name("r1_decoder_research.py"))
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)

OUT = config.RESULTS_DIR / "research"
NEW_LABELS = OUT / "labels_progress"
DECODER = HMMParams(beta=10.0, emission_weight=0.5, unreachable_logp=-25.0, backward_tol=30.0, gap_dt=300.0)
TUNED = json.loads((Path(__file__).resolve().parents[1] / "results" / "tuning" / "best_lightgbm.json").read_text())


def progress_labels(split: str, trip_id: str) -> dict:
    d = config.DATA_ROOT / split / trip_id
    trip, route = load_trip(d), load_route(d)
    old = np.load(config.CACHE_DIR / "labels" / split / f"{trip_id}.npz")
    proj = Projector.for_track(trip.track)
    px, py = proj.to_xy(trip.track[:, 0], trip.track[:, 1])
    a, b = trip.arcs[route, 0], trip.arcs[route, 1]
    ax, ay = proj.to_xy(trip.nodes[a, 0], trip.nodes[a, 1])
    bx, by = proj.to_xy(trip.nodes[b, 0], trip.nodes[b, 1])
    L = np.hypot(bx - ax, by - ay)
    start = np.r_[0.0, np.cumsum(L)[:-1]]
    s = start[old["route_pos"]] + old["t_on_arc"] * L[old["route_pos"]]
    t = trip.track[:, 2]
    seg = np.cumsum(np.r_[False, np.diff(t) > 60])
    s_smooth = np.empty_like(s)
    for g in np.unique(seg):
        m = seg == g
        sm = pd.Series(s[m]).rolling(5, center=True, min_periods=1).median().to_numpy()
        s_smooth[m] = IsotonicRegression(increasing=True).fit_transform(t[m], sm) if m.sum() > 1 else sm
    pos = np.clip(np.searchsorted(start, s_smooth, side="right") - 1, 0, len(route) - 1)
    pos = np.maximum.accumulate(pos)  # progress along the route never goes backwards
    dist, _ = point_segment(px, py, ax[pos], ay[pos], bx[pos], by[pos])
    heading, speed = track_kinematics(px, py, t, k=3)
    bearing = np.arctan2(by - ay, bx - ax)
    return {"route_pos": pos, "arc_id": route[pos], "dist": dist, "old_arc": old["arc_id"], "old_dist": old["dist"],
            "speed": speed, "cos_old": np.cos(heading - bearing[old["route_pos"]]),
            "cos_new": np.cos(heading - bearing[pos])}


def flips(arcs: np.ndarray) -> int:
    """A -> B -> A patterns in the per-point label sequence (boundary jitter)."""
    s = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
    return int(np.sum(s[2:] == s[:-2])) if len(s) > 2 else 0


def main() -> None:
    t0 = time.perf_counter()
    NEW_LABELS.mkdir(parents=True, exist_ok=True)
    bad = config.bad_reference_trips()
    stats = []
    for split in config.SPLITS:
        (NEW_LABELS / split).mkdir(exist_ok=True)
        for td in list_trips(config.DATA_ROOT / split):
            if (split, td.name) in bad:
                continue
            lab = progress_labels(split, td.name)
            np.savez_compressed(NEW_LABELS / split / f"{td.name}.npz", **lab)
            rel = (lab["dist"] <= config.LABEL_MAX_DIST) & (lab["old_dist"] <= config.LABEL_MAX_DIST)
            fast = rel & (lab["speed"] >= 5.0)
            stats.append({"split": split, "trip_id": td.name, "n": len(lab["arc_id"]),
                          "changed": float(np.mean(lab["arc_id"][rel] != lab["old_arc"][rel])),
                          "flips_old": flips(lab["old_arc"]), "flips_new": flips(lab["arc_id"]),
                          "n_fast": int(fast.sum()),
                          "dir_agree_old": float(np.mean(lab["cos_old"][fast] > 0)) if fast.any() else np.nan,
                          "dir_agree_new": float(np.mean(lab["cos_new"][fast] > 0)) if fast.any() else np.nan})
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "r2_label_stats.csv", index=False)
    w = st["n"] / st["n"].sum()
    wf = st["n_fast"] / st["n_fast"].sum()
    print(f"labels changed {np.sum(w * st['changed']):.2%} of points; A-B-A flips {st['flips_old'].sum()} -> "
          f"{st['flips_new'].sum()}; direction agreement at >=5 m/s {np.nansum(wf * st['dir_agree_old']):.4f} -> "
          f"{np.nansum(wf * st['dir_agree_new']):.4f}", flush=True)

    # new targets for the cached candidate tables
    def with_new_y(split):
        P = pd.read_parquet(config.CACHE_DIR / "dataset" / f"{split}_pairs.parquet")
        ys, rels = [], []
        for tid, g in P.groupby("trip_id", sort=False):
            lab = np.load(NEW_LABELS / split / f"{tid}.npz")
            o = g["obs"].to_numpy()
            ys.append(g["arc_id"].to_numpy() == lab["arc_id"][o])
            rels.append(lab["dist"][o] <= config.LABEL_MAX_DIST)
        P["y_new"] = np.concatenate(ys).astype(np.int8)
        P["rel_new"] = np.concatenate(rels)
        return P

    P_tr, P_te = with_new_y("train"), with_new_y("student_test")
    params = {**SPECS["lightgbm"].params, **TUNED}

    # train evidence: OOF (3 folds) for both label sets on all train trips, then decode 20 trips
    codes = {t: i for i, t in enumerate(sorted(P_tr["trip_id"].unique()))}
    keys = point_keys(P_tr["trip_id"].map(codes).to_numpy(), P_tr["obs"].to_numpy())
    folds = list(GroupKFold(n_splits=3, shuffle=True, random_state=config.SEED).split(P_tr, groups=P_tr["trip_id"]))
    X = P_tr[FEATURES].to_numpy()
    oof = {}
    for name, ycol, relcol in (("old", "y", "reliable"), ("new", "y_new", "rel_new")):
        prob = np.zeros(len(P_tr))
        for tr, va in folds:
            tr = tr[P_tr[relcol].to_numpy()[tr]]
            m = SPECS["lightgbm"].build(params, config.SEED).fit(X[tr], P_tr[ycol].to_numpy()[tr])
            prob[va] = m.predict_proba(X[va])[:, 1]
        oof[name] = prob
        print(f"OOF done for {name} labels", flush=True)

    rng = np.random.default_rng(config.SEED)
    trips = sorted(rng.choice(P_tr["trip_id"].unique(), size=20, replace=False).tolist())
    bundles = {t: r1.trip_bundle("train", t) for t in trips}
    res = {}
    for name in ("old", "new"):
        df = P_tr[["trip_id", "obs", "arc_id"]].assign(prob=oof[name]).groupby("trip_id", sort=False)
        probs = {t: r1.probs_for(b, df, t) for t, b in bundles.items()}
        res[f"train_{name}_labels_route"] = r1.run(bundles, probs, DECODER)  # route metrics + top1(old labels)
    # top-1 of each model under each label definition (pair level, OOF, classifier only)
    rel_both = P_tr["reliable"].to_numpy() & P_tr["rel_new"].to_numpy()
    Q = P_tr[rel_both]
    n_pts = Q.groupby(["trip_id", "obs"]).ngroups
    for name in ("old", "new"):
        for lab_def, ycol in (("old_def", "y"), ("new_def", "y_new")):
            res[f"train_oof_classifier_top1_model_{name}_{lab_def}"] = top1_accuracy(
                keys[rel_both], Q[ycol].to_numpy(), oof[name][rel_both], n_pts)

    # student-test once: models trained on all train trips
    test_bundles = {t: r1.trip_bundle("student_test", t) for t in sorted(P_te["trip_id"].unique())}
    for name, ycol, relcol in (("old", "y", "reliable"), ("new", "y_new", "rel_new")):
        tr = P_tr[relcol].to_numpy()
        m = SPECS["lightgbm"].build(params, config.SEED).fit(X[tr], P_tr[ycol].to_numpy()[tr])
        prob = m.predict_proba(P_te[FEATURES].to_numpy())[:, 1]
        df = P_te[["trip_id", "obs", "arc_id"]].assign(prob=prob).groupby("trip_id", sort=False)
        probs = {t: r1.probs_for(b, df, t) for t, b in test_bundles.items()}
        res[f"test_{name}_labels_route"] = r1.run(test_bundles, probs, DECODER)
        # top-1 of the decoded route under the NEW label definition as well
        hits = n = 0
        for t, b in test_bundles.items():
            rows = r1.decode_viterbi(r1.lattice(b["ctx"], b["graph"], probs[t], DECODER), len(b["ctx"].px))
            newlab = np.load(NEW_LABELS / "student_test" / f"{t}.npz")
            rel = newlab["dist"] <= config.LABEL_MAX_DIST
            pred = np.where(rows >= 0, b["ctx"].cands.arc_id[np.maximum(rows, 0)], -1)
            hits += int((pred[rel] == newlab["arc_id"][rel]).sum())
            n += int(rel.sum())
        res[f"test_{name}_labels_top1_new_def"] = hits / n
        print(f"test done for {name} labels", flush=True)

    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    (OUT / "r2_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
