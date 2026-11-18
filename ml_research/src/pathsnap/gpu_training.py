"""CUDA MLP training and group-aware HMM comparison."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .decoding import decode_trip_hmm
from .feature_pipeline import FEATURE_COLUMNS
from .feature_store import load_trip_feature_set
from .hmm_training import DECODER_GRID, _metric_for_classifier, _metric_for_edges
from .training import _observation_metric_summary, _save_json, _training_matrix


def _require_cuda():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install a CUDA-enabled PyTorch build in the selected environment.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this Python environment; GPU training was not started.")
    return torch, nn, torch.device("cuda:0")


def _network(nn, input_width: int):
    return nn.Sequential(
        nn.Linear(input_width, 64), nn.ReLU(), nn.Dropout(0.05),
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def _fit_gpu_mlp(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int,
                 epochs: int, batch_size: int, learning_rate: float = 0.001):
    torch, nn, device = _require_cuda()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    w_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    mean = x_tensor.mean(dim=0)
    scale = x_tensor.std(dim=0, unbiased=False).clamp_min(1e-4)
    x_tensor = (x_tensor - mean) / scale
    model = _network(nn, x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    sample_count = len(y)
    epoch_losses = []
    for epoch in range(epochs):
        permutation = torch.randperm(sample_count, device=device)
        loss_total = weight_total = 0.0
        for offset in range(0, sample_count, batch_size):
            rows = permutation[offset:offset + batch_size]
            batch_x, batch_y, batch_w = x_tensor[rows], y_tensor[rows], w_tensor[rows]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x).squeeze(1)
            loss_vector = criterion(logits, batch_y) * batch_w
            loss = loss_vector.sum() / batch_w.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
            loss_total += float(loss_vector.detach().sum().item())
            weight_total += float(batch_w.detach().sum().item())
        epoch_losses.append(loss_total / max(weight_total, 1e-8))
        print(f"    GPU epoch {epoch+1}/{epochs}: weighted_bce={epoch_losses[-1]:.5f}", flush=True)
    model.eval()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fitted = {"model": model, "mean": mean, "scale": scale, "device": device,
              "epoch_losses": epoch_losses, "fit_seconds": elapsed}
    del x_tensor, y_tensor, w_tensor
    return fitted


def _predict_gpu_mlp(fitted, features: np.ndarray, chunk_rows: int = 131_072) -> np.ndarray:
    torch, _, device = _require_cuda()
    outputs = np.empty(len(features), dtype=np.float32)
    model, mean, scale = fitted["model"], fitted["mean"], fitted["scale"]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), chunk_rows):
            stop = min(len(features), start + chunk_rows)
            batch = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
            batch = (batch - mean) / scale
            outputs[start:stop] = torch.sigmoid(model(batch).squeeze(1)).detach().cpu().numpy()
    return np.clip(outputs, 1e-7, 1 - 1e-7)


def _summarize_folds(trip_rows: dict[str, dict], fold_by_trip: dict[str, int],
                     excluded: set[str] | None = None) -> dict:
    excluded = excluded or set()
    rows = [row for trip_id, row in trip_rows.items() if trip_id not in excluded]
    folds = []
    for fold in sorted(set(fold_by_trip.values())):
        fold_rows = [trip_rows[trip_id] for trip_id, assigned in fold_by_trip.items()
                     if assigned == fold and trip_id not in excluded and trip_id in trip_rows]
        if fold_rows:
            folds.append({"fold": fold, **_observation_metric_summary(fold_rows)})
    fold_scores = [row["observation_top1_accuracy"] for row in folds]
    return {**_observation_metric_summary(rows),
            "fold_top1_mean": float(np.mean(fold_scores)) if fold_scores else 0.0,
            "fold_top1_std": float(np.std(fold_scores)) if fold_scores else 0.0,
            "folds": folds}


def run_gpu_mlp_hmm_experiment(train_paths, student_paths, output_dir: str | Path,
                               n_splits: int = 5, seed: int = 42,
                               epochs: int = 6, batch_size: int = 32_768) -> dict:
    """Fit and evaluate a CUDA MLP; select HMM settings without using student labels."""
    torch, _, device = _require_cuda()
    train_paths, student_paths = list(map(Path, train_paths)), list(map(Path, student_paths))
    if not train_paths or not student_paths:
        raise ValueError("Both training and student feature caches are required.")
    root = Path(output_dir)
    probability_dir = root / "oof_probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    ids = [path.stem for path in train_paths]
    folds = list(GroupKFold(n_splits=n_splits).split(np.arange(len(ids)), groups=ids))
    fold_by_trip: dict[str, int] = {}
    classifier_rows: dict[str, dict] = {}
    fit_log = []

    for fold_no, (fit_idx, valid_idx) in enumerate(folds, start=1):
        print(f"[GPU MLP fold {fold_no}/{n_splits}] preparing training data", flush=True)
        matrix_start = time.perf_counter()
        x, y, weights, _ = _training_matrix([train_paths[i] for i in fit_idx], seed + fold_no)
        matrix_seconds = time.perf_counter() - matrix_start
        fitted = _fit_gpu_mlp(x, y, weights, seed + fold_no, epochs, batch_size)
        del x, y, weights
        inference_start = time.perf_counter()
        for idx in valid_idx:
            path = train_paths[idx]
            data = load_trip_feature_set(path)
            probabilities = _predict_gpu_mlp(fitted, data.features)
            np.save(probability_dir / f"{path.stem}.npy", probabilities)
            classifier_rows[path.stem] = _metric_for_classifier(data, probabilities)
            fold_by_trip[path.stem] = fold_no
            del data, probabilities
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        fit_log.append({"fold": fold_no, "training_matrix_seconds": matrix_seconds,
            "fit_seconds": fitted["fit_seconds"], "inference_seconds": inference_seconds,
            "training_trips": len(fit_idx), "validation_trips": len(valid_idx),
            "sampled_rows": int(len(y)) if "y" in locals() else None,
            "epochs": epochs, "batch_size": batch_size})
        _save_json(root / "gpu_fit_log.json", {"device": torch.cuda.get_device_name(0),
            "folds_completed": fold_no, "folds": fit_log})
        print(f"[GPU MLP fold {fold_no}/{n_splits}] saved held-out predictions", flush=True)
        del fitted

    config_keys = [json.dumps(params, sort_keys=True) for params in DECODER_GRID]
    params_by_key = {json.dumps(params, sort_keys=True): params for params in DECODER_GRID}
    hmm_rows = {key: {} for key in config_keys}
    for path in train_paths:
        data = load_trip_feature_set(path)
        probabilities = np.load(probability_dir / f"{path.stem}.npy", allow_pickle=False)
        for key in config_keys:
            decoded = decode_trip_hmm(data, probabilities, **params_by_key[key])
            hmm_rows[key][path.stem] = _metric_for_edges(data, decoded)
        del data, probabilities
        print(f"[GPU MLP HMM] decoded trip={path.stem}", flush=True)

    nested_rows, nested_choices = {}, []
    for held_out in range(1, n_splits + 1):
        choices = []
        for key in config_keys:
            tune_rows = [rows[trip_id] for trip_id, fold in fold_by_trip.items()
                         if fold != held_out and trip_id in hmm_rows[key]]
            score = _observation_metric_summary(tune_rows)["observation_top1_accuracy"]
            choices.append((score, key))
        score, chosen_key = max(choices, key=lambda item: (item[0], item[1]))
        nested_choices.append({"held_out_fold": held_out, "selected_params": params_by_key[chosen_key],
                               "selection_top1_other_folds": score})
        for trip_id, fold in fold_by_trip.items():
            if fold == held_out:
                nested_rows[trip_id] = hmm_rows[chosen_key][trip_id]
    full_config_scores = [( _observation_metric_summary(list(rows.values()))["observation_top1_accuracy"], key)
                           for key, rows in hmm_rows.items()]
    selected_score, selected_key = max(full_config_scores, key=lambda item: (item[0], item[1]))
    selected_params = params_by_key[selected_key]
    outlier_ids = [trip_id for trip_id, row in sorted(classifier_rows.items(),
        key=lambda item: (-item[1]["candidate_miss_rate"], item[0]))[:3]]

    print("[student comparison] fitting CUDA MLP on train trips only", flush=True)
    x, y, weights, _ = _training_matrix(train_paths, seed)
    final_model = _fit_gpu_mlp(x, y, weights, seed, epochs, batch_size)
    del x, y, weights
    student_classifier_rows, student_hmm_rows = {}, {}
    inference_start = time.perf_counter()
    for path in student_paths:
        data = load_trip_feature_set(path)
        probabilities = _predict_gpu_mlp(final_model, data.features)
        student_classifier_rows[path.stem] = _metric_for_classifier(data, probabilities)
        student_hmm_rows[path.stem] = _metric_for_edges(
            data, decode_trip_hmm(data, probabilities, **selected_params))
        del data, probabilities
    torch.cuda.synchronize()
    student_inference_seconds = time.perf_counter() - inference_start

    checkpoint = root / "gpu_mlp_checkpoint.pt"
    torch.save({"state_dict": final_model["model"].state_dict(),
        "mean": final_model["mean"].detach().cpu(), "scale": final_model["scale"].detach().cpu(),
        "feature_columns": list(FEATURE_COLUMNS), "architecture": "15-64-32-1 ReLU MLP",
        "candidate_radius_m": 250.0, "max_candidates": 128,
        "decoder_params": selected_params, "seed": seed, "epochs": epochs,
        "batch_size": batch_size}, checkpoint)

    result = {"protocol": {"model": "PyTorch MLP on CUDA", "device": torch.cuda.get_device_name(0),
        "cv": "GroupKFold by training trip_id", "n_splits": n_splits, "seed": seed,
        "epochs": epochs, "batch_size": batch_size,
        "decoder_selection": "each held-out fold uses settings selected on the other OOF folds",
        "student_test_role": "comparison only; not used for fitting or decoder selection"},
        "classifier_only": {"all_training_oof": _summarize_folds(classifier_rows, fold_by_trip),
            "excluding_three_highest_miss_trips": _summarize_folds(classifier_rows, fold_by_trip, set(outlier_ids)),
            "excluded_trip_ids": outlier_ids},
        "classifier_plus_hmm": {"nested_training_oof": _summarize_folds(nested_rows, fold_by_trip),
            "excluding_three_highest_miss_trips": _summarize_folds(nested_rows, fold_by_trip, set(outlier_ids)),
            "excluded_trip_ids": outlier_ids},
        "student_test_development_comparison": {
            "classifier_only": _observation_metric_summary(list(student_classifier_rows.values())),
            "classifier_plus_hmm": _observation_metric_summary(list(student_hmm_rows.values())),
            "train_fit_seconds": final_model["fit_seconds"],
            "inference_seconds": student_inference_seconds},
        "decoder_selected_on_training_oof": selected_params,
        "decoder_grid_results": [{"params": params_by_key[key], "training_oof_top1": score}
                                 for score, key in sorted(full_config_scores, reverse=True)],
        "nested_decoder_choices": nested_choices, "gpu_fit_log": fit_log,
        "checkpoint": str(checkpoint)}
    _save_json(root / "gpu_classifier_vs_hmm.json", result)
    with (root / "gpu_hmm_trip_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "trip_id", "classifier_top1", "hmm_top1", "candidate_miss_rate"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trip_id in classifier_rows:
            writer.writerow({"split": "train_oof", "trip_id": trip_id,
                "classifier_top1": classifier_rows[trip_id]["observation_top1_accuracy"],
                "hmm_top1": nested_rows[trip_id]["observation_top1_accuracy"],
                "candidate_miss_rate": classifier_rows[trip_id]["candidate_miss_rate"]})
        for trip_id in student_classifier_rows:
            writer.writerow({"split": "student_test", "trip_id": trip_id,
                "classifier_top1": student_classifier_rows[trip_id]["observation_top1_accuracy"],
                "hmm_top1": student_hmm_rows[trip_id]["observation_top1_accuracy"],
                "candidate_miss_rate": student_classifier_rows[trip_id]["candidate_miss_rate"]})
    return result
