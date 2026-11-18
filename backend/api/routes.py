"""Small system routes that are not part of the ML contract.

`/health` lives in `api/ml_routes.py`: the evaluator UI expects `{status, model, n_trips}`,
not the older application metadata shape.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["System"])
async def root():
    return {
        "message": "Welcome to PathSnap Backend",
        "status": "running"
    }


@router.get("/ready", tags=["System"])
async def readiness_check():
    return {
        "ready": True,
        "message": "PathSnap backend is ready"
    }
