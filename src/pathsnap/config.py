"""Paths and global settings. Locations can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The raw data lives outside the repo (course material is not redistributed).
DATA_ROOT = Path(os.environ.get(
    "PATHSNAP_DATA_ROOT", REPO_ROOT.parent / "Provided Material" / "Student folder"))
CACHE_DIR = Path(os.environ.get("PATHSNAP_CACHE_DIR", REPO_ROOT / "cache"))
RESULTS_DIR = Path(os.environ.get("PATHSNAP_RESULTS_DIR", REPO_ROOT / "results"))

SPLITS = ("train", "student_test")
SEED = 352  # fixed for every experiment; logged with every result

# Label reliability (chosen from the train label-distance distribution, see results/labels/).
LABEL_MAX_DIST = 50.0       # a point further than this from its reference arc is unlabelled
TRIP_MAX_MEDIAN_DIST = 50.0 # a trip whose median label distance exceeds this has an unusable
                            # reference (train 33: route ~120 m from on-road GPS; train 54:
                            # whole track shifted ~117 m off every road) and is left out


def bad_reference_trips() -> set[tuple[str, str]]:
    """(split, trip_id) pairs whose reference route is unusable, from labels_summary.csv."""
    import pandas as pd

    df = pd.read_csv(RESULTS_DIR / "labels" / "labels_summary.csv", dtype={"trip_id": str})
    bad = df[df["dist_median_m"] > TRIP_MAX_MEDIAN_DIST]
    return set(zip(bad["split"], bad["trip_id"]))
