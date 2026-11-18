
from fastapi import Depends, HTTPException, status

from security.authentication import get_current_device


async def get_admin_device(
    device=Depends(get_current_device)
):
    if device["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required"
        )

    return device
