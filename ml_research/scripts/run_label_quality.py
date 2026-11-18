"""Run the trip-by-trip route-label comparison from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pathsnap.label_quality import main


if __name__ == "__main__":
    main()
