"""Canonical-table EDA summary and charts; values are computed from the supplied data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_table(frame) -> dict[str, Any]:
    """Summarize missingness, duplicates, numeric ranges, and common PathSnap fields."""
    numeric = frame.select_dtypes(include="number")
    summary: dict[str, Any] = {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "columns_detail": {str(c): {"dtype": str(frame[c].dtype),
                                    "missing_count": int(frame[c].isna().sum()),
                                    "missing_fraction": float(frame[c].isna().mean()),
                                    "unique_count": int(frame[c].nunique(dropna=True))}
                           for c in frame.columns},
        "numeric_describe": numeric.describe().replace({float("inf"): None, float("-inf"): None}).to_dict(),
    }
    if {"latitude", "longitude"}.issubset(frame.columns):
        import pandas as pd
        valid = frame[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
        summary["coordinate_coverage"] = {
            "valid_points": int(len(valid)),
            "latitude_min": float(valid.latitude.min()) if len(valid) else None,
            "latitude_max": float(valid.latitude.max()) if len(valid) else None,
            "longitude_min": float(valid.longitude.min()) if len(valid) else None,
            "longitude_max": float(valid.longitude.max()) if len(valid) else None,
            "out_of_bounds_count": int(((valid.latitude.abs() > 90) | (valid.longitude.abs() > 180)).sum()),
        }
    if "trajectory_id" in frame.columns:
        sizes = frame.groupby("trajectory_id", dropna=False).size()
        summary["trajectory_count"] = int(len(sizes))
        summary["points_per_trajectory"] = {"min": int(sizes.min()), "median": float(sizes.median()),
                                             "max": int(sizes.max())} if len(sizes) else {}
    if "timestamp" in frame.columns:
        parsed = __import__("pandas").to_datetime(frame["timestamp"], errors="coerce", utc=True)
        summary["timestamp_coverage"] = {"valid_count": int(parsed.notna().sum()),
                                          "invalid_count": int(parsed.isna().sum()),
                                          "min_utc": None if parsed.notna().sum() == 0 else parsed.min().isoformat(),
                                          "max_utc": None if parsed.notna().sum() == 0 else parsed.max().isoformat()}
    return summary


def write_eda(frame, output_json: str | Path, figures_dir: str | Path | None = None) -> dict[str, Any]:
    """Write actual data summaries and missingness/coordinate plots when matplotlib is installed."""
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_table(frame)
    output_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if figures_dir is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Install matplotlib to create EDA figures.") from exc
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        missing = frame.isna().mean().sort_values(ascending=False)
        if len(missing):
            ax = missing.plot(kind="bar", figsize=(max(7, len(missing) * .45), 4), color="#2563eb")
            ax.set(title="Missing values by column", ylabel="Fraction missing", xlabel="")
            ax.figure.tight_layout()
            ax.figure.savefig(figures_dir / "missingness.png", dpi=160)
            plt.close(ax.figure)
        if {"latitude", "longitude"}.issubset(frame.columns):
            import pandas as pd
            coordinates = frame[["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce").dropna()
            ax = coordinates.plot.scatter(x="longitude", y="latitude", s=3, alpha=.4, figsize=(7, 6))
            ax.set(title="Observed GPS point coverage")
            ax.figure.tight_layout()
            ax.figure.savefig(figures_dir / "coordinate_coverage.png", dpi=160)
            plt.close(ax.figure)
    return summary
