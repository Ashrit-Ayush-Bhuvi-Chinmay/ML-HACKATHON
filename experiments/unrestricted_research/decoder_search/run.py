"""Compare experimental decoders without writing to the frozen model or results.

See the experiment README for setup and reproduction commands. Uses the supplied
train and student-test folders and the saved classifier predictions.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.evaluate import arc_lengths_m, route_scores
from pathsnap.features import prepare_trip
from pathsnap.hmm import HMMParams, RoadGraph, build_route, route_distance, segment_ids, viterbi
from pathsnap.io import load_route, load_trip


MODELS = ("lightgbm_tuned", "lightgbm", "xgboost", "ranker_torch", "catboost")
TRANSITION_FEATURES = [
    "distance_error", "network_distance", "gps_displacement", "elapsed_seconds",
    "same_arc", "reverse_arc", "angle_between_arcs", "angle_from_gps_step_to_source",
    "angle_from_gps_step_to_destination", "gps_distance_to_source_arc",
    "gps_distance_to_destination_arc", "source_speed", "destination_speed",
    "source_arc_progress", "destination_arc_progress", "near_junction",
]


@dataclass(frozen=True)
class Setting:
    name: str
    model: str
    beta: float
    weight: float = 1.0
    adaptive: bool = False
    learned_weight: float = 0.0


FROZEN = Setting("frozen", "lightgbm_tuned", 20)
BASELINES = [
    FROZEN,
    Setting("untuned_reference", "lightgbm", 10, weight=2),
    Setting("xgboost_reference", "xgboost", 5),
    Setting("ranker_reference", "ranker_torch", 5),
]
SEARCH = BASELINES + [
    Setting(f"{model}_beta{beta}", model, beta)
    for model in (*MODELS, "blend")
    for beta in (3, 8, 12, 20, 32)
    if (model, beta) != (FROZEN.model, FROZEN.beta)
] + [
    Setting(f"blend_adaptive{beta}", "blend", beta, adaptive=True)
    for beta in (3, 6, 10)
] + [
    Setting(f"blend_learned_beta{beta}_w{weight}", "blend", beta, learned_weight=weight)
    for beta in (5, 12, 20)
    for weight in (0.25, 0.75, 1.5)
]


def load_predictions(suffix):
    return {
        name: pd.read_parquet(config.CACHE_DIR / "preds" / f"{name}_{suffix}.parquet")
        .groupby("trip_id", sort=False)
        for name in MODELS
    }


def get_trip(split, trip_id, predictions):
    directory = config.DATA_ROOT / split / trip_id
    trip = load_trip(directory)
    ctx = prepare_trip(trip, config.CAND_RADIUS, config.CAND_K)
    ref = load_route(directory).tolist()
    with np.load(config.CACHE_DIR / "labels" / split / f"{trip_id}.npz") as labels:
        reliable = labels["dist"] <= config.LABEL_MAX_DIST
        reference_arcs = labels["arc_id"]
    scores = {}
    for name, grouped in predictions.items():
        rows = grouped.get_group(trip_id)
        if not (np.array_equal(rows["obs"], ctx.cands.obs)
                and np.array_equal(rows["arc_id"], ctx.cands.arc_id)):
            raise ValueError(f"Saved candidates differ for {trip_id}/{name}")
        scores[name] = rows["prob"].to_numpy()

    # Normalize within each candidate set before averaging. Pair classifiers and the listwise
    # ranker have different probability scales.
    _, starts = np.unique(ctx.cands.obs, return_index=True)
    ends = np.r_[starts[1:], len(ctx.cands.obs)]
    for name, p in scores.items():
        p = np.clip(p, 1e-12, None)
        scores[name] = p / np.repeat(np.add.reduceat(p, starts), ends - starts)
    scores["blend"] = np.mean([scores[name] for name in MODELS], axis=0)
    return ctx, reliable, reference_arcs, ref, scores


def transition_lattice(ctx, graph, segments):
    c = ctx.cands
    observations, starts = np.unique(c.obs, return_index=True)
    ends = np.r_[starts[1:], len(c.obs)]
    groups = [slice(start, end) for start, end in zip(starts, ends)]
    for prev_obs, obs, prev, current in zip(observations[:-1], observations[1:], groups[:-1], groups[1:]):
        if segments[prev_obs] != segments[obs]:
            continue
        dx = ctx.px[obs] - ctx.px[prev_obs]
        dy = ctx.py[obs] - ctx.py[prev_obs]
        gps_distance = np.hypot(dx, dy)
        source, destination = c.local[prev], c.local[current]
        network_distance = route_distance(graph, ctx, source, c.t[prev], destination, c.t[current], 15)
        yield prev_obs, obs, prev, current, gps_distance, np.arctan2(dy, dx), network_distance


def transition_features(ctx, step):
    prev_obs, obs, prev, current, gps_distance, gps_bearing, network_distance = step
    net, c = ctx.net, ctx.cands
    source, destination = c.local[prev, None], c.local[None, current]
    source_bearing, destination_bearing = net.bearing[source], net.bearing[destination]
    reverse = (net.u[source] == net.v[destination]) & (net.v[source] == net.u[destination])
    features = [
        np.abs(network_distance - gps_distance).clip(0, 2000),
        network_distance.clip(0, 2000), gps_distance, ctx.t[obs] - ctx.t[prev_obs],
        source == destination, reverse, np.cos(source_bearing - destination_bearing),
        np.cos(source_bearing - gps_bearing), np.cos(destination_bearing - gps_bearing),
        c.dist[prev, None], c.dist[None, current], ctx.speed[prev_obs], ctx.speed[obs],
        c.t[prev, None], c.t[None, current], net.v[source] == net.u[destination],
    ]
    return np.stack(np.broadcast_arrays(*features), axis=-1).astype(np.float32)


def fit_transitions(trip_ids, predictions, rng):
    samples, targets, weights = [], [], []
    for trip_id in trip_ids:
        ctx, rel, ref_arcs, _, _ = get_trip("train", trip_id, predictions)
        c = ctx.cands
        params = HMMParams()
        graph = RoadGraph(ctx, params.sp_limit)
        segments = segment_ids(ctx, params)
        for step in transition_lattice(ctx, graph, segments):
            prev_obs, obs, prev, current, _, _, network_distance = step
            if not (rel[prev_obs] and rel[obs]):
                continue
            positive = np.flatnonzero(((c.arc_id[prev, None] == ref_arcs[prev_obs])
                                       & (c.arc_id[None, current] == ref_arcs[obs])
                                       & np.isfinite(network_distance)).ravel())
            negative = np.flatnonzero(np.isfinite(network_distance).ravel())
            negative = negative[negative != positive[0]] if len(positive) else []
            if not len(negative):
                continue
            chosen = rng.choice(negative, size=min(8, len(negative)), replace=False)
            pairs = np.r_[positive, chosen]
            features = transition_features(ctx, step).reshape(-1, len(TRANSITION_FEATURES))
            samples.append(features[pairs])
            targets.extend([1] + [0] * len(chosen))
            # Each observation pair contributes equally to each class.
            weights.extend([1.0] + [1.0 / len(chosen)] * len(chosen))
        print(f"transition fit: {trip_id}", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=180, num_leaves=31, learning_rate=0.06, min_child_samples=150,
        random_state=config.SEED, n_jobs=4, verbosity=-1,
    )
    model.fit(np.concatenate(samples), targets, sample_weight=weights, feature_name=TRANSITION_FEATURES)
    print(f"learned from {sum(targets):,} transitions", flush=True)
    return model


def decode(ctx, graph, segments, probabilities, settings, learned):
    c = ctx.cands
    obs, starts = np.unique(c.obs, return_index=True)
    ends = np.r_[starts[1:], len(c.obs)]
    output = np.full((len(settings), len(ctx.px)), -1, dtype=np.int64)
    if not len(obs):
        return output
    emissions = np.stack([np.log(probabilities[s.model]) * s.weight for s in settings])
    beta = np.array([s.beta for s in settings], dtype=float)
    adaptive = np.array([s.adaptive for s in settings])
    learned_weight = np.array([s.learned_weight for s in settings])
    steps = list(transition_lattice(ctx, graph, segments))

    # Batch the tree predictions to avoid one Python/model call per GPS observation.
    learned_scores = {}
    if np.any(learned_weight):
        counts = [step[-1].size for step in steps]
        if counts:
            matrix = np.concatenate(
                [transition_features(ctx, step).reshape(-1, len(TRANSITION_FEATURES)) for step in steps]
            )
            logits = learned.booster_.predict(matrix, raw_score=True, num_threads=4)
            bounds = np.r_[0, np.cumsum(counts)]
            learned_scores = {
                step[1]: logits[a:b].reshape(step[-1].shape).clip(-6, 6)
                for step, a, b in zip(steps, bounds[:-1], bounds[1:])
            }
    step_by_obs = {step[1]: step for step in steps}
    variants = np.arange(len(settings))
    chain = []
    score = None

    def finish():
        if not chain:
            return
        state = score.argmax(axis=1)
        for point, start, backpointer in reversed(chain):
            output[:, point] = start + state
            if backpointer is not None:
                state = backpointer[variants, state]

    for point, start, end in zip(obs, starts, ends):
        emission = emissions[:, start:end]
        if point not in step_by_obs:
            finish()
            score = emission
            chain = [(point, start, None)]
            continue
        _, _, _, _, gps_distance, _, network_distance = step_by_obs[point]
        scale = beta + adaptive * (0.25 * gps_distance)
        transition = -np.abs(network_distance - gps_distance) / scale[:, None, None]
        if point in learned_scores:
            transition += learned_weight[:, None, None] * learned_scores[point]
        transition = np.where(np.isfinite(network_distance), transition, -40.0)
        total = score[:, :, None] + transition
        backpointer = total.argmax(axis=1)
        score = total.max(axis=1) + emission
        chain.append((point, start, backpointer))
    finish()
    return output


def evaluate(split, trip_ids, predictions, settings, learned, cohort):
    records = []
    for trip_id in trip_ids:
        ctx, rel, ref_arcs, reference, probabilities = get_trip(split, trip_id, predictions)
        params = HMMParams()
        graph = RoadGraph(ctx, params.sp_limit)
        segments = segment_ids(ctx, params)
        rows = decode(ctx, graph, segments, probabilities, settings, learned)
        if settings[0] == FROZEN:
            expected, _ = viterbi(ctx, probabilities[FROZEN.model], HMMParams(beta=20), graph)
            np.testing.assert_array_equal(rows[0], expected)
        for setting, prediction in zip(settings, rows):
            arcs = np.full(len(prediction), -1, dtype=np.int64)
            matched = prediction >= 0
            arcs[matched] = ctx.cands.arc_id[prediction[matched]]
            pieces = build_route(ctx, graph, prediction, segments)
            route = [arc for piece in pieces for arc in piece]
            lengths = arc_lengths_m(ctx.trip, ctx.proj, route + reference)
            metrics = route_scores(route, reference, lengths)
            records.append({
                "cohort": cohort, "trip_id": trip_id, "setting": setting.name,
                "hits": int(np.sum(rel & (arcs == ref_arcs))), "scored": int(np.sum(rel)), **metrics,
            })
        print(f"{cohort}: {trip_id}", flush=True)
    return pd.DataFrame(records)


def summarize(frame):
    scores = frame.groupby(["cohort", "setting"], sort=False).agg(
        trips=("trip_id", "size"), hits=("hits", "sum"), scored=("scored", "sum"),
        sequence_f1=("sequence_f1", "mean"), coverage_f1=("coverage_f1", "mean"),
    )
    scores["top1"] = scores["hits"] / scores["scored"]
    return scores.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (args.output / "configs.csv").open("w", newline="") as handle:
        pd.DataFrame([vars(s) for s in SEARCH]).to_csv(handle, index=False)
    rng = np.random.default_rng(config.SEED)
    train = load_predictions("oof")
    ids = rng.permutation(sorted(train[FROZEN.model].groups))
    fit, select, check = ids[:30], ids[30:54], ids[54:]
    pd.DataFrame([{"trip_id": trip, "role": role} for role, group in
                  (("fit_transitions", fit), ("select", select), ("check", check)) for trip in group]
                 ).to_csv(args.output / "train_split.csv", index=False)
    learned = fit_transitions(fit, train, rng)
    learned.booster_.save_model(str(args.output / "transition_model.txt"))

    selection = evaluate("train", select, train, SEARCH, learned, "train_select")
    selection.to_csv(args.output / "selection_per_trip.csv", index=False)
    leaderboard = summarize(selection)
    winners = {leaderboard.nlargest(1, metric).iloc[0]["setting"] for metric in ("top1", "sequence_f1")}
    learned_names = {s.name for s in SEARCH if s.learned_weight}
    learned_winner = leaderboard[leaderboard.setting.isin(learned_names)].nlargest(1, "sequence_f1").iloc[0]
    shortlist = BASELINES + [
        s for s in SEARCH if s not in BASELINES and s.name in winners | {learned_winner.setting}
    ]
    print(f"Shortlist from train selection: {[s.name for s in shortlist]}", flush=True)
    verification = evaluate("train", check, train, shortlist, learned, "train_check")
    test = load_predictions("test")
    heldout = evaluate("student_test", sorted(test[FROZEN.model].groups), test, shortlist, learned, "student_test")
    pd.concat([verification, heldout]).to_csv(args.output / "comparison_per_trip.csv", index=False)
    summary = summarize(pd.concat([selection, verification, heldout]))
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary[summary.cohort != "train_select"].to_string(index=False), flush=True)
    print(f"Finished in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
