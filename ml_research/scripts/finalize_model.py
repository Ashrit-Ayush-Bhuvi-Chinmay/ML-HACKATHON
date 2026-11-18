"""Freeze the selected training-only model, then score the student partition once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pathsnap.final_model import fit_final_model, score_final_student_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/interim/features")
    parser.add_argument("--archive", default="data/raw/Student folder.zip")
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--output-dir", default="experiments/final_model_15features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-only", action="store_true",
                        help="Fit and save the model without opening or scoring student-test trips")
    args = parser.parse_args()
    manifest_path = Path(args.output_dir) / "final_model_manifest.json"
    artifact_path = Path(args.output_dir) / "pathsnap_model.joblib"
    if manifest_path.is_file() and artifact_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("Using the already-frozen final model.", flush=True)
    else:
        manifest = fit_final_model(args.cache_dir, args.cv_dir, args.output_dir, args.seed)
    print(json.dumps({"stage": "final_model_fitted", "model": manifest["model"],
                      "artifact": manifest["artifact"]}, indent=2), flush=True)
    if not args.fit_only:
        result = score_final_student_test(
            args.archive, args.cache_dir,
            artifact_path,
            args.output_dir, seed=args.seed,
        )
        print(json.dumps({"stage": "student_test_scored_once",
                          "classifier_top1": result["classifier_only"]["pooled_top1"],
                          "hmm_top1": result["classifier_plus_hmm"]["pooled_top1"]}, indent=2))


if __name__ == "__main__":
    main()
