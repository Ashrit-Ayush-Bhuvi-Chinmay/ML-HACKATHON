
from fastapi import APIRouter
from datetime import datetime, timezone

from config import settings

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/", tags=["System"])
async def root():
    return {
        "message": "Welcome to PathSnap Backend",
        "status": "running"
    }
