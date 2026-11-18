
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from security.device_store import get_authenticated_device


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_device(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    )
):
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    device = get_authenticated_device(credentials.credentials)

    if not device:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired or revoked device session"
        )

    return device
