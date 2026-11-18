"""Feature safety checks for the PathSnap candidate classifier."""

from __future__ import annotations

from typing import Iterable

FORBIDDEN_FEATURE_NAMES = {"route", "label", "target", "ground_truth", "matched_edge", "is_match",
                          "true_edge_id", "route_position", "route_distance_m"}


def validate_feature_columns(columns: Iterable[str]) -> None:
    """Reject target/reference columns from the model feature matrix."""
    bad = [str(c) for c in columns if str(c).strip().lower() in FORBIDDEN_FEATURE_NAMES]
    if bad:
        raise ValueError(f"Target or ground-truth columns cannot be model features: {bad}")
