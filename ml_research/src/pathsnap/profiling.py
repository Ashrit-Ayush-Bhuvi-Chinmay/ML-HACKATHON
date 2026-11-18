"""Non-destructive discovery and profiling for PathSnap source files."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_SUFFIXES = {".track", ".nodes", ".arcs", ".route"}


def _tokens(line: str) -> list[str]:
    line = line.strip()
    if not line or line.startswith(("#", "//", "%")):
        return []
    if "," in line or "\t" in line or ";" in line:
        try:
            return next(csv.reader([line], delimiter="\t" if "\t" in line else ";" if ";" in line and "," not in line else ","))
        except csv.Error:
            pass
    return re.split(r"\s+", line)


def profile_file(path: Path, sample_limit: int = 5) -> dict[str, Any]:
    """Summarize a text file without assuming its domain schema."""
    result: dict[str, Any] = {"path": str(path), "suffix": path.suffix.lower(), "bytes": path.stat().st_size}
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            rows = [tokens for line in f if (tokens := _tokens(line))]
        widths = Counter(map(len, rows))
        result.update({
            "parsed_nonempty_rows": len(rows),
            "field_count_frequency": {str(k): v for k, v in sorted(widths.items())},
            "sample_rows": rows[:sample_limit],
            "consistent_width": len(widths) <= 1,
        })
        if widths and len(widths) > 1:
            result["warning"] = "Rows have inconsistent field counts; inspect sample_rows before adapting."
    except (OSError, UnicodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def profile_dataset(root: str | Path) -> dict[str, Any]:
    """Inventory supported files and summarize directory coverage."""
    root = Path(root)
    if not root.exists():
        return {"root": str(root), "exists": False, "files": [], "counts_by_suffix": {}}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXPECTED_SUFFIXES)
    details = [profile_file(p) for p in files]
    counts = Counter(p.suffix.lower() for p in files)
    return {
        "root": str(root), "exists": True, "file_count": len(files),
        "counts_by_suffix": {suffix: counts.get(suffix, 0) for suffix in sorted(EXPECTED_SUFFIXES)},
        "files": details,
        "notice": "Profiling records structure only; it does not assign semantics to columns.",
    }


def write_profile(root: str | Path, output: str | Path) -> dict[str, Any]:
    profile = profile_dataset(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile
