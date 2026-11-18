
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
import time

from config import settings
from api.routes import router
from api.auth_routes import router as auth_router 
from api.device_routes import router as device_router
from security.rate_limiter import limiter
from security.pairing import pairing_manager
from security.device_store import initialize_database
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):

    initialize_database()

    pairing_code = pairing_manager.generate_code()

    logger.info("Starting PathSnap Backend")
    logger.info("Environment: %s", settings.APP_ENV)

    # Display pairing code only in the server console
    print("\n" + "=" * 55)
    print("PATHSNAP PRIVATE DEVICE PAIRING")
    print("Pairing code: " + pairing_code)
    print("Expires in: 120 seconds")
    print("=" * 55 + "\n")

    yield

    logger.info("Shutting down PathSnap Backend")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Secure distributed backend for PathSnap",
    lifespan=lifespan
)


# Rate limiting
app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600
)


# Request logging and security headers
@app.middleware("http")
async def security_middleware(request: Request, call_next):

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"

        duration = (time.perf_counter() - start_time) * 1000

        logger.info(
            "%s %s | %s | %.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            duration
        )

        return response

    except Exception:
        logger.exception("Unhandled request error")

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


# Register routes
app.include_router(router)
app.include_router(auth_router) 
app.include_router(device_router)


@app.get("/server-info", tags=["System"])
async def server_info():
    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV
    }


@app.get("/ready", tags=["System"])
async def readiness_check():
    return {
        "ready": True,
        "message": "PathSnap backend is ready"
    }
