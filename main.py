"""PathSnap FastAPI backend.

Demo flow: uvicorn main:app on laptop 1, the React UI on laptop 2 calls /health, /trips and
/predict. The model is loaded once at startup and every request is logged to this console.
"""
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.auth_routes import router as auth_router
from api.device_routes import router as device_router
from api.ml_routes import router as ml_router
from api.routes import router as system_router
from config import settings
from integrations.ml_adapter import ml_adapter
from networking.discovery import start_broadcaster, stop_broadcaster
from networking.network_utils import get_local_ipv4
from security.device_store import initialize_database
from security.pairing import pairing_manager
from security.rate_limiter import limiter
from utils.logger import logger


def _api_url() -> str:
    ip = get_local_ipv4()
    host = ip or "127.0.0.1"
    return f"http://{host}:{settings.PORT}"


def _print_startup_banner(discovery_active: bool) -> None:
    lines = [
        "=" * 64,
        " PathSnap API",
        "-" * 64,
        f" Model        : {ml_adapter.model_name}",
        f" Trips folder : {ml_adapter.eval_dir}",
        f" Trips found  : {ml_adapter.n_trips}",
        f" API address  : {_api_url()}",
        "                (type this into the UI's API address box)",
        f" Auth         : {'required' if settings.PATHSNAP_REQUIRE_AUTH else 'disabled'}",
        f" Discovery    : {'on' if discovery_active else 'off'}",
        "=" * 64,
    ]
    print("\n".join(lines) + "\n", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PathSnap Backend")
    logger.info("Environment: %s", settings.APP_ENV)

    initialize_database()

    if settings.PATHSNAP_REQUIRE_AUTH:
        pairing_code = pairing_manager.generate_code()
        print("\n" + "=" * 55)
        print("PATHSNAP PRIVATE DEVICE PAIRING")
        print("Pairing code: " + pairing_code)
        print("Expires in: 120 seconds")
        print("=" * 55 + "\n", flush=True)
    else:
        logger.info("Device pairing is disabled (PATHSNAP_REQUIRE_AUTH=false)")

    # Load the trip registry, the model bundle and run one prediction before serving.
    await ml_adapter.startup()

    discovery_active = False
    if settings.PATHSNAP_DISCOVERY:
        if not settings.DISCOVERY_SECRET.strip():
            logger.warning(
                "UDP discovery requested, but DISCOVERY_SECRET is empty; continuing without it"
            )
        else:
            try:
                start_broadcaster(settings.PORT)
                discovery_active = True
            except Exception:  # noqa: BLE001 - discovery must never block the demo
                logger.warning("UDP discovery could not start; continuing without it", exc_info=True)

    _print_startup_banner(discovery_active)

    try:
        yield
    finally:
        if discovery_active:
            stop_broadcaster()
        logger.info("Shutting down PathSnap Backend")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Secure distributed backend for PathSnap",
    lifespan=lifespan,
)


# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# CORS: laptop 2's origin is unknown in advance, so the default is "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)


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
            duration,
        )

        return response

    except Exception:
        logger.exception("Unhandled request error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# The ML routes must come first: /health is the contract's shape, not the old system one.
app.include_router(ml_router)
app.include_router(system_router)
app.include_router(auth_router)
app.include_router(device_router)


@app.get("/server-info", tags=["System"])
async def server_info():
    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
    }
