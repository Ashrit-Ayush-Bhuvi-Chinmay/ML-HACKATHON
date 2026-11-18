
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field

from security.pairing import pairing_manager
from security.device_store import register_device
from security.authentication import get_current_device
from security.rate_limiter import limiter


router = APIRouter(prefix="/auth", tags=["Authentication"])


class PairRequest(BaseModel):
    device_name: str = Field(min_length=2, max_length=50)
    pairing_code: str = Field(min_length=12, max_length=12)


@router.post("/pair")
@limiter.limit("5/minute")
async def pair_device(request: Request, payload: PairRequest):

    valid = pairing_manager.verify_code(payload.pairing_code)

    if not valid:
        raise HTTPException(
            status_code=403,
            detail="Invalid, expired or already-used pairing code"
        )

    device = register_device(payload.device_name.strip())

    return {
        "status": "paired",
        "message": "Device successfully authorized",
        "device_id": device["device_id"],
        "access_token": device["access_token"],
        "token_type": "bearer",
        "expires_at": device["expires_at"]
    }


@router.get("/me")
async def current_device(
    device=Depends(get_current_device)
):
    return {
        "authenticated": True,
        "device": device
    }
