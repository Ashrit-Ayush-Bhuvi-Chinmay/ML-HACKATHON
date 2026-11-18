"""Freeze record: what exactly will be demonstrated, fixed BEFORE the evaluator data arrives.

Writes docs/FREEZE.md with the time, the git commit, SHA-256 of the final bundle and of every
saved model, the final configuration, and the student-test rehearsal summary. Commit this file
right after generating it; the commit timestamp and hashes show that nothing was tuned on the
evaluator trips (anyone can re-hash the files to check they are unchanged).
Usage: python scripts/13_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime

import joblib
import pandas as pd

from pathsnap import config
from pathsnap.pipeline import ARTIFACT_PATH, MODELS_DIR


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "artifacts"],
                           cwd=config.REPO_ROOT, capture_output=True, text=True).stdout.strip()
    b = joblib.load(ARTIFACT_PATH)
    reh_path = config.RESULTS_DIR / "pipeline" / f"rehearsal_{b['model_name']}.csv"
    reh = pd.read_csv(reh_path, dtype={"trip_id": str}) if reh_path.exists() else None

    lines = [
        "# Freeze record",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} on the `core-ml` branch, before the evaluator",
        "test folder was received. Nothing below may change after the evaluator data arrives; the API only",
        "points `PATHSNAP_EVAL_DIR` at that folder.",
        "",
        f"- Code commit at freeze time: `{head}`" + (" (with uncommitted changes in src/scripts/artifacts!)" if dirty else ""),
        f"- Final bundle: `artifacts/{ARTIFACT_PATH.name}`  SHA-256 `{sha256(ARTIFACT_PATH)}`",
        f"- Model: `{b['model_name']}`; trained on {b['trained_on']}",
        f"- Model parameters: `{json.dumps(b['model_params'], default=str)}`",
        f"- Candidates: radius {b['candidate_radius']} m, at most {b['candidate_k']} per point; label cutoff {b['label_max_dist']} m",
        f"- HMM decoder: `{json.dumps(b['hmm_params'])}`",
        f"- Features ({len(b['features'])}): `{', '.join(b['features'])}`",
        f"- Seed: {b['seed']}",
        "- Data used for any fitting, tuning or selection: `train` (80 trips, 78 with usable references) and",
        "  `student_test` (10 trips, comparison only). The evaluator partition was never available.",
        "",
        "## Student-test rehearsal (full pipeline, exactly as the API runs it)",
        "",
    ]
    if reh is not None:
        w = reh["n_points_scored"] / reh["n_points_scored"].sum()
        lines += [
            f"- Point-weighted Top-1: {float((w * reh['top1_accuracy']).sum()):.4f}",
            f"- Mean sequence F1: {reh['sequence_f1'].mean():.4f}; mean coverage F1: {reh['coverage_f1'].mean():.4f}",
            f"- Mean route continuity: {reh['route_continuity'].mean():.4f}",
            f"- Prediction wall time per trip: max {reh['wall_ms_predict'].max():.0f} ms; scoring max {reh['wall_ms_score'].max():.0f} ms",
            "",
            "| trip | points | Top-1 | sequence F1 | coverage F1 | continuity | predict ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in reh.iterrows():
            lines.append(f"| {r.trip_id} | {r.n_points} | {r.top1_accuracy:.4f} | {r.sequence_f1:.4f} | "
                         f"{r.coverage_f1:.4f} | {r.route_continuity:.4f} | {r.wall_ms_predict:.0f} |")
    lines += ["", "## Every saved model (for the UI history view)", "", "| file | SHA-256 |", "|---|---|"]
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        lines.append(f"| `artifacts/models/{p.name}` | `{sha256(p)}` |")
    out = config.REPO_ROOT / "docs" / "FREEZE.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
