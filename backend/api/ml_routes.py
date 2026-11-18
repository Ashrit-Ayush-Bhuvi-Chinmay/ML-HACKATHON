"""The five ML endpoints consumed by the evaluator UI (see docs/API_CONTRACT.md)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator

from config import settings
from integrations.ml_adapter import ml_adapter
from security.authentication import bearer_scheme, get_current_device
from utils.logger import logger


async def require_device_when_enabled(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    """No-op by default; a bearer-token check when PATHSNAP_REQUIRE_AUTH=true."""
    if not settings.PATHSNAP_REQUIRE_AUTH:
        return None
    return await get_current_device(credentials)


router = APIRouter(tags=["ML"], dependencies=[Depends(require_device_when_enabled)])


class PredictRequest(BaseModel):
    # Same rule as the task: ids are folder names, never paths.
    trip_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    model: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    hmm: bool = True

    @field_validator("model")
    @classmethod
    def _reject_traversal(cls, value: str | None) -> str | None:
        # Model names become artifact file names inside pathsnap; keep them plain.
        if value and ".." in value:
            raise ValueError("model name must not contain '..'")
        return value


@router.get("/health")
async def health():
    return ml_adapter.health()


@router.get("/trips")
async def trips():
    return {"trips": ml_adapter.trips()}


@router.post("/predict")
async def predict(payload: PredictRequest, request: Request):
    if not ml_adapter.has_trip(payload.trip_id):
        raise HTTPException(status_code=404, detail="unknown trip_id")

    client = request.client.host if request.client else "-"
    try:
        return await ml_adapter.predict(payload.trip_id, payload.model, payload.hmm, client)
    except KeyError as exc:
        detail = str(exc.args[0]) if exc.args else "unknown model"
        if "unknown model" in detail:
            logger.warning("Rejected prediction: %s", detail)
            raise HTTPException(status_code=404, detail=detail) from exc
        logger.exception("Prediction failed for trip=%s", payload.trip_id)
        raise HTTPException(status_code=500, detail="prediction failed on the server") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - the client gets a short message, the console a traceback
        logger.exception("Prediction failed for trip=%s", payload.trip_id)
        raise HTTPException(status_code=500, detail="prediction failed on the server") from exc


@router.get("/models")
async def models():
    return {"models": await ml_adapter.models()}


@router.get("/experiments")
async def experiments():
    return await ml_adapter.experiments()
