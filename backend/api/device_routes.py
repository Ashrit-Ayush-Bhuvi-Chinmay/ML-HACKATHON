
from fastapi import APIRouter, Depends, HTTPException

from security.authentication import get_current_device
from security.authorization import get_admin_device

from security.device_store import (
    list_devices,
    revoke_device
)


router = APIRouter(
    prefix="/devices",
    tags=["Device Management"]
)


# Each authenticated device can view itself
@router.get("/me")
async def my_device(
    device=Depends(get_current_device)
):
    return {
        "device": device
    }


# Only administrators can list all devices
@router.get("/")
async def get_all_devices(
    admin=Depends(get_admin_device)
):
    return {
        "devices": list_devices()
    }


# Only administrators can revoke devices
@router.delete("/{device_id}")
async def remove_device(
    device_id: str,
    admin=Depends(get_admin_device)
):
    devices = list_devices()

    target = next(
        (
            item for item in devices
            if item["device_id"] == device_id
        ),
        None
    )

    if not target:
        raise HTTPException(
            status_code=404,
            detail="Device not found"
        )

    if target["revoked"]:
        raise HTTPException(
            status_code=409,
            detail="Device is already revoked"
        )

    if target["role"] == "admin":
        active_admins = [
            item for item in devices
            if item["role"] == "admin"
            and not item["revoked"]
        ]

        if len(active_admins) <= 1:
            raise HTTPException(
                status_code=409,
                detail="Cannot revoke the last active administrator"
            )

    success = revoke_device(device_id)

    if not success:
        raise HTTPException(
            status_code=409,
            detail="Device could not be revoked"
        )

    return {
        "status": "revoked",
        "device_id": device_id,
        "message": "Device access has been revoked"
    }
