"""Report figures, all generated from recorded results (no numbers typed in).

Writes PNGs to results/figures/ and small tables to results/report/.
Needs the cached dataset/labels/predictions (scripts 01-05); set PATHSNAP_CACHE_DIR if the cache
lives in another checkout.
Usage: python scripts/15_report_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.evaluate import best_per_point
from pathsnap.features import FEATURES
from pathsnap.geo import Projector
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.network import build_network
from pathsnap.pipeline import MODELS_DIR, _score_candidates, load_model

FIG = config.RESULTS_DIR / "figures"
TAB = config.RESULTS_DIR / "report"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# validated categorical palette (light): fixed order, never cycled
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False,
})
NAMES = {"distance_gaussian": "Distance only", "gaussian_nb": "Gaussian NB", "logreg": "Logistic Reg.",
         "knn": "k-NN", "random_forest": "Random Forest", "mlp_torch": "MLP", "ranker_torch": "Listwise ranker",
         "catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM", "lightgbm_tuned": "LightGBM (tuned)",
         "lightgbm_v1feats": "LightGBM (no long heading)"}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name, flush=True)


# ------------------------------------------------------------------ EDA
def eda():
    rows, dts = [], []
    for split in ("train", "student_test"):
        for td in list_trips(config.DATA_ROOT / split):
            trip = load_trip(td)
            t = trip.track[:, 2]
            dt = np.diff(t)
            dts.append(pd.DataFrame({"split": split, "dt": dt}))
            proj = Projector.for_track(trip.track)
            route = load_route(td)
            rows.append({"split": split, "trip_id": td.name, "points": trip.n_points, "nodes": len(trip.nodes),
                         "arcs": len(trip.arcs), "route_arcs": len(route), "duration_h": (t[-1] - t[0]) / 3600,
                         "nonfinite": int((~np.isfinite(trip.track)).sum()), "unsorted_steps": int((dt < 0).sum()),
                         "duplicate_times": int((dt == 0).sum()), "gaps_over_30s": int((dt > 30).sum()),
                         "max_gap_s": float(dt.max()) if len(dt) else 0.0, "lon": float(np.median(trip.track[:, 0])),
                         "lat": float(np.median(trip.track[:, 1])), "utm_epsg": proj.epsg})
    inv = pd.DataFrame(rows)
    inv.to_csv(TAB / "eda_trip_inventory.csv", index=False)
    lab = pd.read_csv(config.RESULTS_DIR / "labels" / "labels_summary.csv", dtype={"trip_id": str})
    summary = inv.groupby("split").agg(trips=("trip_id", "count"), points=("points", "sum"),
                                       median_points=("points", "median"), max_arcs=("arcs", "max"),
                                       nonfinite=("nonfinite", "sum"), unsorted=("unsorted_steps", "sum"),
                                       duplicate_times=("duplicate_times", "sum"), gaps_over_30s=("gaps_over_30s", "sum"),
                                       max_gap_h=("max_gap_s", lambda s: s.max() / 3600))
    summary.to_csv(TAB / "eda_split_summary.csv")
    d = pd.concat(dts)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for i, split in enumerate(("train", "student_test")):
        ax[0].hist(inv.loc[inv.split == split, "points"], bins=np.logspace(2.3, 4.4, 20), color=C[i], alpha=0.85,
                   label=split.replace("_", "-"), edgecolor="white")
    ax[0].set_xscale("log"); ax[0].set_title("GPS points per trip"); ax[0].set_xlabel("points"); ax[0].set_ylabel("trips")
    ax[0].legend()
    ax[1].hist(inv["arcs"], bins=np.logspace(2.5, 5.6, 20), color=C[0], edgecolor="white")
    ax[1].set_xscale("log"); ax[1].set_title("Road-network size per trip"); ax[1].set_xlabel("arcs (directed segments)")
    vals = d["dt"].clip(upper=60)
    ax[2].hist(vals, bins=np.arange(0.5, 61.5, 1), color=C[0], edgecolor="white")
    ax[2].set_yscale("log"); ax[2].set_title("Time between consecutive GPS points")
    ax[2].set_xlabel("seconds (60 = 60 s or more)"); ax[2].set_ylabel("steps (log)")
    save(fig, "eda_trip_sizes_sampling.png")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, split in enumerate(("train", "student_test")):
        s = inv[inv.split == split]
        ax.scatter(s["lon"], s["lat"], s=18, color=C[i], label=split.replace("_", "-"), edgecolor="white", linewidth=0.5)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 75); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Where the trips are (median GPS position of each trip)"); ax.legend(loc="lower left")
    save(fig, "eda_trip_locations.png")

    # label distance distribution (from the per-point labels)
    dist = {s: np.concatenate([np.load(p)["dist"] for p in sorted((config.CACHE_DIR / "labels" / s).glob("*.npz"))])
            for s in ("train", "student_test")}
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, (s, v) in enumerate(dist.items()):
        ax.hist(np.clip(v, 0, 200), bins=np.arange(0, 201, 4), color=C[i], alpha=0.8, density=True,
                label=f"{s.replace('_', '-')} (median {np.median(v):.1f} m)", edgecolor="white")
    ax.axvline(config.LABEL_MAX_DIST, color=INK, linewidth=1, linestyle="--")
    ax.text(config.LABEL_MAX_DIST + 3, ax.get_ylim()[1] * 0.85, "50 m: beyond = unlabelled", color=INK2, fontsize=8)
    ax.set_title("Distance from each GPS point to its derived reference segment")
    ax.set_xlabel("metres (200 = 200 m or more)"); ax.set_ylabel("density"); ax.legend()
    save(fig, "eda_label_distance.png")
    lab[["split", "trip_id", "n_points", "dist_median_m", "dist_p90_m", "frac_gt_50m"]].to_csv(
        TAB / "eda_label_quality.csv", index=False)

    # one trip: road network, GPS, reference route (projected, metres)
    split, tid = "student_test", "00000009"
    td = config.DATA_ROOT / split / tid
    trip, route = load_trip(td), load_route(td)
    proj = Projector.for_track(trip.track)
    px, py = proj.to_xy(trip.track[:, 0], trip.track[:, 1])
    net = build_network(trip, proj, px, py, margin=300)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(np.c_[net.ax, net.bx, np.full_like(net.ax, np.nan)].ravel(),
            np.c_[net.ay, net.by, np.full_like(net.ay, np.nan)].ravel(), color="#b9b8b2", linewidth=0.5, label="road network")
    nx, ny = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    a, b = trip.arcs[route, 0], trip.arcs[route, 1]
    ax.plot(np.c_[nx[a], nx[b], np.full(len(a), np.nan)].ravel(), np.c_[ny[a], ny[b], np.full(len(a), np.nan)].ravel(),
            color=C[2], linewidth=2.5, label="reference route")
    ax.scatter(px, py, s=3, color=C[0], label="GPS points", zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"Student-test trip {tid}: GPS track, road network and reference route")
    ax.legend(loc="upper right", markerscale=3)
    save(fig, "eda_trip_network.png")


# ------------------------------------------------------------------ candidates
def candidates():
    sw = pd.read_csv(config.RESULTS_DIR / "candidates" / "sweep.csv")
    s = sw[sw.split == "train"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, r in enumerate([25, 50, 100, 200]):
        q = s[s.radius_m == r].copy()
        q = q[q.k_max != "all"].sort_values("mean_set_size")
        ax.plot(q["mean_set_size"], 100 * q["miss_rate"], marker="o", markersize=4, linewidth=2, color=C[i], label=f"radius {r} m")
    ax.annotate("chosen: 50 m, K=16", xy=(7.8, 0.11), xytext=(12, 1.5), arrowprops=dict(arrowstyle="->", color=INK2), color=INK)
    ax.set_xlabel("mean candidates per GPS point"); ax.set_ylabel("reference segment missed (%)")
    ax.set_title("Candidate generation: cost vs miss rate (train split)"); ax.legend()
    save(fig, "candidates_sweep.png")


# ------------------------------------------------------------------ models
def models():
    comp = pd.read_csv(config.RESULTS_DIR / "models" / "model_comparison.csv")
    comp = comp[comp.model.isin(NAMES)].copy()
    comp["name"] = comp.model.map(NAMES)
    order = comp.sort_values("test_top1")
    final = order.model == "lightgbm_tuned"

    def colors(base=C[0]):
        return [C[1] if f else base for f in final]

    # fold std from run files
    stds = []
    for m in order.model:
        runs = sorted((config.RESULTS_DIR / "models" / "runs").glob(f"{m}_2*.json"))
        f = json.loads(runs[-1].read_text())["folds"]
        stds.append(np.std([x["macro_f1"] for x in f], ddof=1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(order.name, 100 * order.cv_macro_f1_mean, xerr=100 * np.array(stds), color=colors(), height=0.6,
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_xlim(60, 100); ax.set_xlabel("CV Macro F1 (%) — mean ± std over 5 trip-grouped folds")
    ax.set_title("Cross-validation Macro F1 (train split); final model in orange")
    save(fig, "models_cv_macro_f1.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax[0].barh(order.name, 100 * order.test_top1, color=colors(), height=0.6)
    ax[0].set_xlim(50, 100); ax[0].set_title("Student-test Top-1 (%)")
    ax[1].barh(order.name, 100 * order.test_macro_f1, color=colors(C[2]), height=0.6)
    ax[1].set_xlim(60, 100); ax[1].set_title("Student-test Macro F1 (%)")
    for a, col in zip(ax, ("test_top1", "test_macro_f1")):
        for y, v in enumerate(order[col]):
            a.text(100 * v + 0.3, y, f"{100 * v:.1f}", va="center", fontsize=7, color=INK2)
    save(fig, "models_test_top1_macro_f1.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    o2 = comp.sort_values("test_log_loss", ascending=False)
    ax.barh(o2.name, o2.test_log_loss, color=[C[1] if m == "lightgbm_tuned" else C[0] for m in o2.model], height=0.6)
    ax.set_xscale("log"); ax.set_xlabel("student-test log loss (log scale, lower is better)")
    ax.set_title("Probability quality (log loss)")
    save(fig, "models_log_loss.png")

    # epoch-wise training histories (only the models that have them)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for a, m in zip(ax, ("mlp_torch", "ranker_torch")):
        runs = sorted((config.RESULTS_DIR / "models" / "runs").glob(f"{m}_2*.json"))
        run = json.loads(runs[-1].read_text())
        h = run["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        a.plot(ep, h["train_loss"], color=C[0], linewidth=2, label="training loss")
        a.plot(ep, h["val_loss"], color=C[1], linewidth=2, label="validation loss (held-out trips)")
        a.axvline(run.get("best_epoch", 0), color=INK2, linestyle="--", linewidth=0.8)
        a.set_title(f"{NAMES[m]}: loss per epoch (final fit)"); a.set_xlabel("epoch"); a.legend()
    ax[0].set_ylabel("binary cross-entropy"); ax[1].set_ylabel("softmax cross-entropy")
    save(fig, "models_loss_curves.png")


def fit_gap(n_trips=10):
    """In-sample (training) Top-1 of each saved final model on a sample of train trips, to set
    against CV and test Top-1 (over/underfitting evidence)."""
    P = pd.read_parquet(config.CACHE_DIR / "dataset" / "train_pairs.parquet")
    rng = np.random.default_rng(config.SEED)
    trips = rng.choice(P.trip_id.unique(), size=n_trips, replace=False)
    P = P[P.trip_id.isin(trips) & P.reliable].reset_index(drop=True)
    code = {t: i for i, t in enumerate(trips)}
    keys = P.trip_id.map(code).to_numpy().astype(np.int64) * 1_000_000 + P.obs.to_numpy()
    Q = pd.read_parquet(config.CACHE_DIR / "dataset" / "train_points.parquet")
    n_rel = int(Q[Q.trip_id.isin(trips) & Q.reliable].shape[0])
    comp = pd.read_csv(config.RESULTS_DIR / "models" / "model_comparison.csv").set_index("model")
    rows = []
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        b = load_model(path.stem)
        prob = _score_candidates(b, P, keys)
        _, idx = best_per_point(keys, prob)
        rows.append({"model": path.stem, "train_fit_top1": P.y.to_numpy()[idx].sum() / n_rel,
                     "cv_top1": comp.loc[path.stem, "cv_top1"], "test_top1": comp.loc[path.stem, "test_top1"]})
        print("fit gap", path.stem, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "fit_gap.csv", index=False)
    df = df[df.model.isin(NAMES)].sort_values("test_top1")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(df))
    for i, (col, lab) in enumerate((("train_fit_top1", "training (in-sample)"), ("cv_top1", "cross-validation"),
                                    ("test_top1", "student-test"))):
        ax.scatter(100 * df[col], y, color=C[i], s=28, label=lab, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, df.model.map(NAMES)); ax.set_xlabel("Top-1 (%)"); ax.set_xlim(50, 101)
    ax.set_title("Over/underfitting check: training vs CV vs student-test Top-1"); ax.legend(loc="center left")
    save(fig, "models_fit_gap.png")


def confusion():
    t = pd.read_parquet(config.CACHE_DIR / "preds" / "lightgbm_tuned_test.parquet")
    P = pd.read_parquet(config.CACHE_DIR / "dataset" / "student_test_pairs.parquet", columns=["y", "reliable"])
    m = P.reliable.to_numpy()
    y, p = P.y.to_numpy()[m], (t.prob.to_numpy()[m] >= 0.5).astype(int)
    cm = np.array([[((y == a) & (p == b)).sum() for b in (0, 1)] for a in (0, 1)])
    pd.DataFrame(cm, index=["true: not reference", "true: reference"],
                 columns=["pred: not reference", "pred: reference"]).to_csv(TAB / "confusion_lightgbm_tuned.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({100 * cm[i, j] / cm[i].sum():.1f}%)", ha="center", va="center",
                    color="white" if (i == j) else INK, fontsize=8)
    ax.set_xticks([0, 1], ["not reference", "reference"]); ax.set_yticks([0, 1], ["not reference", "reference"])
    ax.set_xlabel("predicted (threshold 0.5)"); ax.set_ylabel("true"); ax.grid(False)
    ax.set_title("Tuned LightGBM, student-test candidates")
    save(fig, "models_confusion.png")


# ------------------------------------------------------------------ tuning, HMM, final
def tuning():
    t = pd.read_csv(config.RESULTS_DIR / "tuning" / "trials_lightgbm.csv")
    done = t[t.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.scatter(done.number, 100 * done.value, color=C[0], s=30, zorder=3, label="completed trial")
    pr = t[t.state == "PRUNED"]
    ax.scatter(pr.number, np.full(len(pr), 100 * done.value.min()), marker="x", color=C[7], label="pruned trial")
    ax.plot(done.number, 100 * done.value.cummax(), color=C[1], linewidth=2, label="best so far")
    ax.axvspan(0.5, 7.5, color="#f1f0ec", zorder=0)
    ax.text(4, 100 * done.value.min() + 0.05, "random warm-up", ha="center", color=INK2, fontsize=8)
    ax.set_xlabel("trial"); ax.set_ylabel("3-fold CV Top-1 (%)"); ax.legend(loc="lower right")
    ax.set_title("Bayesian optimisation of LightGBM (Optuna TPE); trial 0 = untuned")
    save(fig, "tuning_trials.png")


def hmm():
    s = pd.read_csv(config.RESULTS_DIR / "hmm" / "summary_lightgbm_tuned.csv", index_col=0)
    order = ["distance_only", "classifier_only", "distance_hmm", "classifier_hmm"]
    labels = ["Nearest road", "Classifier only", "Classic HMM", "Classifier + HMM"]
    metrics = [("top1", "Top-1"), ("sequence_f1", "Sequence F1"), ("coverage_f1", "Coverage F1"),
               ("continuity_pooled", "Continuity")]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.2
    x = np.arange(len(order))
    for i, (k, lab) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, 100 * s.loc[order, k], width=w - 0.02, color=C[i], label=lab)
    ax.set_xticks(x, labels); ax.set_ylim(0, 105); ax.set_ylabel("%")
    ax.set_title("What the HMM adds (tuned LightGBM, student-test)"); ax.legend(ncol=4, loc="upper left", fontsize=8)
    save(fig, "hmm_ablation.png")


def final():
    ev = pd.read_csv(config.RESULTS_DIR / "evaluation" / "evaluator_per_trip.csv", dtype={"trip_id": str})
    ev = ev[ev.variant == "final_system"]
    st = pd.read_csv(config.RESULTS_DIR / "pipeline" / "rehearsal_lightgbm_tuned.csv", dtype={"trip_id": str})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for a, df, title in ((ax[0], st, "Student-test trips"), (ax[1], ev, "Evaluator trips (one recorded pass)")):
        x = np.arange(len(df))
        a.bar(x - 0.2, 100 * df.top1_accuracy, width=0.38, color=C[0], label="Top-1")
        a.bar(x + 0.2, 100 * df.sequence_f1, width=0.38, color=C[2], label="Sequence F1")
        a.set_xticks(x, df.trip_id.str[-2:]); a.set_xlabel("trip (last two digits)"); a.set_title(title)
        a.set_ylim(80, 101)
    ax[0].set_ylabel("%"); ax[0].legend(loc="lower left")
    save(fig, "final_per_trip.png")


if __name__ == "__main__":
    eda()
    candidates()
    models()
    confusion()
    tuning()
    hmm()
    final()
    fit_gap()
