"""Paths and global settings. Locations can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def _find_data_root() -> Path:
    """The raw data lives outside the repo (course material is not redistributed): the nearest
    '<ancestor>/Provided Material/Student folder', so any checkout or worktree under the hackathon
    folder finds it. Override with PATHSNAP_DATA_ROOT."""
    for parent in REPO_ROOT.parents:
        cand = parent / "Provided Material" / "Student folder"
        if cand.is_dir():
            return cand
    return REPO_ROOT.parent / "Provided Material" / "Student folder"


DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT") or _find_data_root())
CACHE_DIR = Path(os.environ.get("PATHSNAP_CACHE_DIR", REPO_ROOT / "cache"))
RESULTS_DIR = Path(os.environ.get("PATHSNAP_RESULTS_DIR", REPO_ROOT / "results"))

SPLITS = ("train", "student_test")
SEED = 352  # fixed for every experiment; logged with every result

# Label reliability (chosen from the train label-distance distribution, see results/labels/).
LABEL_MAX_DIST = 50.0       # a point further than this from its reference arc is unlabelled
CAND_RADIUS = 50.0          # metres; from results/candidates/sweep.csv (train miss rate 0.11 %
CAND_K = 16                 # at ~7.8 candidates per point)
TRIP_MAX_MEDIAN_DIST = 50.0 # a trip whose median label distance exceeds this has an unusable
                            # reference (train 33: route ~120 m from on-road GPS; train 54:
                            # whole track shifted ~117 m off every road) and is left out


def bad_reference_trips() -> set[tuple[str, str]]:
    """(split, trip_id) pairs whose reference route is unusable, from labels_summary.csv."""
    import pandas as pd

    df = pd.read_csv(RESULTS_DIR / "labels" / "labels_summary.csv", dtype={"trip_id": str})
    bad = df[df["dist_median_m"] > TRIP_MAX_MEDIAN_DIST]
    return set(zip(bad["split"], bad["trip_id"]))
