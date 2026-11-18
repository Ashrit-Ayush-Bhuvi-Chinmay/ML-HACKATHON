"""Test setup: point the API at the supplied student_test trips before importing the app."""
import os
from pathlib import Path

import pytest

# tests/conftest.py -> tests -> secure-backend -> branches -> ML_HACKATHON
REPO_ROOT = Path(__file__).resolve().parents[3]
STUDENT_TEST = REPO_ROOT / "Provided Material" / "Student folder" / "student_test"

# Settings are read when `config` is imported, so set the environment first.
os.environ["PATHSNAP_EVAL_DIR"] = str(STUDENT_TEST)
os.environ["PATHSNAP_REQUIRE_AUTH"] = "false"
os.environ["PATHSNAP_DISCOVERY"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # The context manager runs the lifespan: trip scan, model load and warm-up prediction.
    with TestClient(app) as test_client:
        yield test_client
