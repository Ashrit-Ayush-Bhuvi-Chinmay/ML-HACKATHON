"""FastAPI entry point for a two-laptop evaluator setup."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .inference import predict_route

app = FastAPI(title="PathSnap evaluator", version="0.1.0")
UI_FILE = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


class Point(BaseModel):
    trajectory_id: str
    point_index: int
    timestamp: str | None = None
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class MatchRequest(BaseModel):
    points: list[Point] = Field(min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "PathSnap"}


@app.get("/", include_in_schema=False)
def evaluator_ui():
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="Evaluator UI file was not found.")
    return FileResponse(UI_FILE)


@app.post("/api/match")
def match_route(payload: MatchRequest) -> dict[str, Any]:
    """Run both classifier-only and decoded route inference with frozen artifacts."""
    artifact_path = Path("models/selected_model.joblib")
    if not artifact_path.exists():
        raise HTTPException(status_code=503, detail="No trained model artifact is available yet.")
    try:
        import joblib
        artifact = joblib.load(artifact_path)
        return predict_route([p.model_dump() for p in payload.points], artifact["edges"],
                             artifact["model"], artifact["feature_columns"],
                             artifact.get("candidate_radius_m", 100), artifact.get("max_candidates", 8))
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
