"""Explicit adapters from text files to the project's canonical column contract."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def _split(line: str, delimiter: str | None) -> list[str]:
    if delimiter:
        return next(csv.reader([line], delimiter=delimiter))
    if "\t" in line:
        return next(csv.reader([line], delimiter="\t"))
    if "," in line:
        return next(csv.reader([line], delimiter=","))
    return re.split(r"\s+", line.strip())


def read_text_records(path: str | Path, *, delimiter: str | None = None,
                      has_header: bool = False) -> list[dict[str, Any]]:
    """Read delimited or whitespace text while preserving source field names if present.

    This does not infer geographic semantics. Headerless files receive field_0, field_1, ...
    and must be mapped explicitly in the config after profiling.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        rows = [_split(line.strip(), delimiter) for line in handle
                if line.strip() and not line.lstrip().startswith(("#", "//", "%"))]
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Inconsistent field counts in {path}; inspect the dataset profile first.")
    if has_header:
        headers = [value.strip() for value in rows.pop(0)]
        if len(set(headers)) != len(headers):
            raise ValueError(f"Duplicate header names in {path}.")
    else:
        headers = [f"field_{i}" for i in range(width)]
    return [dict(zip(headers, row)) for row in rows]


def canonicalize(records: Sequence[Mapping[str, Any]], source_columns: Mapping[str, str | int],
                 required: Sequence[str]) -> list[dict[str, Any]]:
    """Map configured source names/indices to canonical names and validate required fields."""
    result = []
    for row_index, record in enumerate(records):
        normalized = {}
        for canonical, source in source_columns.items():
            source_key = f"field_{source}" if isinstance(source, int) else source
            if source_key in record:
                normalized[canonical] = record[source_key]
        missing = [column for column in required if column not in normalized or normalized[column] == ""]
        if missing:
            raise ValueError(f"Record {row_index} lacks required mapped fields: {missing}")
        result.append(normalized)
    return result
