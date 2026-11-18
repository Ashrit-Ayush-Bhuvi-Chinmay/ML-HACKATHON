"""Deterministic group splits with evaluator labels gated from model inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SplitManifest:
    train_groups: tuple[str, ...]
    development_groups: tuple[str, ...]
    evaluator_groups: tuple[str, ...]
    seed: int

    def validate(self) -> None:
        a, b, c = map(set, (self.train_groups, self.development_groups, self.evaluator_groups))
        if a & b or a & c or b & c:
            raise ValueError("Split group sets overlap; evaluator leakage risk.")
        if not a or not b or not c:
            raise ValueError("Train, development, and evaluator splits must all be non-empty.")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {"seed": self.seed, "train_groups": list(self.train_groups),
                "development_groups": list(self.development_groups),
                "evaluator_groups": list(self.evaluator_groups)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")


def make_group_split(groups: Iterable[Any], seed: int = 42,
                     train_fraction: float = .70, development_fraction: float = .15) -> SplitManifest:
    """Stable split by whole group IDs; never splits points from one trajectory across sets."""
    if train_fraction <= 0 or development_fraction <= 0 or train_fraction + development_fraction >= 1:
        raise ValueError("Fractions must be positive and leave a positive evaluator fraction.")
    unique = sorted({str(g) for g in groups})
    if len(unique) < 3:
        raise ValueError("At least three distinct groups are needed for three-way splitting.")
    # Hash ordering makes the assignment stable across Python versions and input order.
    ordered = sorted(unique, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    n_train = max(1, min(len(unique) - 2, round(len(unique) * train_fraction)))
    n_dev = max(1, min(len(unique) - n_train - 1, round(len(unique) * development_fraction)))
    manifest = SplitManifest(tuple(ordered[:n_train]), tuple(ordered[n_train:n_train+n_dev]),
                             tuple(ordered[n_train+n_dev:]), seed)
    manifest.validate()
    return manifest


@dataclass(frozen=True)
class ModelPartition:
    """Model-visible inputs and labels for training or development only."""
    role: str
    inputs: Any
    labels: Any

    def __post_init__(self) -> None:
        if self.role not in {"train", "development"}:
            raise ValueError("ModelPartition may contain labels only for train/development roles.")


@dataclass(frozen=True)
class EvaluatorPartition:
    """Evaluator features only. Ground truth is retained separately for final scoring."""
    inputs: Any
    group_ids: tuple[str, ...]
    _labels_for_final_scoring: Any

    def model_inputs(self) -> Any:
        return self.inputs

    def final_labels(self, *, final_evaluation: bool = False) -> Any:
        if not final_evaluation:
            raise PermissionError("Evaluator labels are available only during final evaluation.")
        return self._labels_for_final_scoring


@dataclass(frozen=True)
class PartitionBundle:
    train: ModelPartition
    development: ModelPartition
    evaluator: EvaluatorPartition


def prepare_partitions(observations: Iterable[Mapping[str, Any]],
                       route_labels: Iterable[Mapping[str, Any]],
                       manifest: SplitManifest,
                       group_column: str = "trajectory_id",
                       key_columns: tuple[str, ...] = ("trajectory_id", "point_index")) -> PartitionBundle:
    """Separate evaluator rows before exposing any route labels to model-facing partitions.

    The returned evaluator object exposes its coordinates through ``model_inputs`` and only
    releases its ground-truth mapping when ``final_labels(final_evaluation=True)`` is called.
    """
    manifest.validate()
    observations, route_labels = list(observations), list(route_labels)
    label_index = {}
    for label in route_labels:
        key = tuple(str(label[column]) for column in key_columns)
        if key in label_index:
            raise ValueError(f"Duplicate ground-truth key: {key}")
        label_index[key] = str(label["edge_id"])
    groups = set(map(str, manifest.train_groups + manifest.development_groups + manifest.evaluator_groups))
    rows_by_role: dict[str, list[dict[str, Any]]] = {"train": [], "development": [], "evaluator": []}
    evaluator_truth = {}
    role_by_group = {g: role for role, vals in (("train", manifest.train_groups),
                                                  ("development", manifest.development_groups),
                                                  ("evaluator", manifest.evaluator_groups)) for g in vals}
    for observation in observations:
        group = str(observation[group_column])
        if group not in groups:
            raise ValueError(f"Observation group missing from split manifest: {group}")
        key = tuple(str(observation[column]) for column in key_columns)
        role = role_by_group[group]
        clean_input = {k: v for k, v in observation.items() if str(k).lower() not in
                       {"route", "label", "target", "ground_truth", "matched_edge", "is_match", "edge_id"}}
        if role == "evaluator":
            rows_by_role[role].append(clean_input)
            if key in label_index:
                evaluator_truth[key] = label_index[key]
        else:
            if key not in label_index:
                raise ValueError(f"Missing route label for {role} observation {key}")
            rows_by_role[role].append({"inputs": clean_input, "target_edge_id": label_index[key]})
    evaluator_groups = tuple(sorted({str(row[group_column]) for row in rows_by_role["evaluator"]}))
    evaluator = EvaluatorPartition(inputs=tuple(rows_by_role["evaluator"]), group_ids=evaluator_groups,
                                   _labels_for_final_scoring=evaluator_truth)
    train_inputs = tuple(row["inputs"] for row in rows_by_role["train"])
    train_labels = tuple(row["target_edge_id"] for row in rows_by_role["train"])
    dev_inputs = tuple(row["inputs"] for row in rows_by_role["development"])
    dev_labels = tuple(row["target_edge_id"] for row in rows_by_role["development"])
    return PartitionBundle(ModelPartition("train", train_inputs, train_labels),
                           ModelPartition("development", dev_inputs, dev_labels), evaluator)


def evaluator_partition_is_disjoint(manifest: SplitManifest, evaluator_groups: Iterable[Any]) -> None:
    manifest.validate()
    unexpected = set(map(str, evaluator_groups)) - set(manifest.evaluator_groups)
    if unexpected:
        raise ValueError(f"Evaluator rows include groups outside the manifest: {sorted(unexpected)[:5]}")
