"""Contract tests for the ML endpoints (see BACKEND_TASK.md section 7)."""
import json
from pathlib import Path

from pathsnap.io import load_route

from config import settings

TRIP = "00000052"
STUDENT_TEST = (
    Path(__file__).resolve().parents[3] / "Provided Material" / "Student folder" / "student_test"
)
TRIP_DIR = STUDENT_TEST / TRIP


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model"], str) and body["model"]
    assert body["n_trips"] == 10


def test_trips(client):
    response = client.get("/trips")
    assert response.status_code == 200

    trips = response.json()["trips"]
    assert len(trips) == 10
    assert [t["trip_id"] for t in trips] == sorted(t["trip_id"] for t in trips)
    assert all(t["n_points"] > 0 for t in trips)
    assert {t["trip_id"] for t in trips} >= {TRIP}


def test_predict(client):
    response = client.post("/predict", json={"trip_id": TRIP})
    assert response.status_code == 200

    body = response.json()
    assert body["trip_id"] == TRIP
    assert body["n_points"] == 262
    assert len(body["predicted_arcs"]) == body["n_points"]
    assert body["metrics"]["top1_accuracy"] > 0.9


def test_predict_does_not_leak_reference_route(client):
    # Loading the label here is fine: this is the evaluator side of the test, not the API.
    reference = load_route(TRIP_DIR).tolist()
    assert reference  # the fixture trip really has a reference route

    body = client.post("/predict", json={"trip_id": TRIP}).json()

    keys = {key.lower() for key in _all_keys(body)}
    assert not [key for key in keys if key.startswith("ref")], keys
    assert "reference_route" not in keys
    assert "route_ref" not in keys

    text = json.dumps(body).lower()
    assert "reference_route" not in text
    assert '"ref' not in text

    metrics = body["metrics"]
    assert isinstance(metrics, dict) and metrics
    for key, value in metrics.items():
        assert isinstance(value, (int, float, str, bool, type(None))), (key, type(value))
        if isinstance(value, str):
            assert len(value) <= 200, (key, value)


def test_unknown_trip_returns_404(client):
    response = client.post("/predict", json={"trip_id": "99999999"})
    assert response.status_code == 404
    assert response.json()["detail"] == "unknown trip_id"


def test_trip_id_with_path_returns_422(client):
    response = client.post("/predict", json={"trip_id": "../etc"})
    assert response.status_code == 422


def test_unknown_model_returns_404(client):
    response = client.post("/predict", json={"trip_id": TRIP, "model": "no-such-model"})
    assert response.status_code == 404
    assert "unknown model" in response.json()["detail"]


def test_predict_without_hmm(client):
    response = client.post("/predict", json={"trip_id": TRIP, "hmm": False})
    assert response.status_code == 200

    body = response.json()
    assert body["hmm"] is False
    assert len(body["predicted_arcs"]) == body["n_points"]


def test_models_include_default(client):
    response = client.get("/models")
    assert response.status_code == 200

    names = [model["name"] for model in response.json()["models"]]
    assert "default" in names


def test_auth_required_when_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "PATHSNAP_REQUIRE_AUTH", True)

    response = client.get("/trips")
    assert response.status_code == 401
