"""
Wine Label Recognition – FastAPI application entry point.

Start the server with:
    uvicorn app.main:app --reload --port 8000
"""

import logging
import sys

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.config import Settings, get_settings
from app.schemas import ErrorDetail
from app.services.rapidapi_client import RapidAPIClient, RapidAPIError

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title="Wine Label Recognition API",
    description=(
        "Upload a wine-bottle photo or supply an image URL and receive "
        "AI-powered label recognition results powered by api4ai via RapidAPI."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(router)


def _get_rapidapi_client(settings: Settings = Depends(get_settings)) -> RapidAPIClient:
    """Return RapidAPIClient when credentials are configured."""
    if not settings.rapidapi_configured:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RapidAPI credentials are not configured on this server.",
        )
    return RapidAPIClient(settings)


@app.get("/api/v1/version", tags=["api"], summary="Upstream API version (GET).")
async def api_version(client: RapidAPIClient = Depends(_get_rapidapi_client)):
    """GET the wine-recognition2 API version from RapidAPI. Uses GET (no body)."""
    try:
        return await client.get_version()
    except RapidAPIError as exc:
        logger.warning("Version endpoint upstream error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "upstream_error", "detail": str(exc)},
        )
    except Exception as exc:
        logger.exception("Version endpoint unexpected error")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": "upstream_error",
                "detail": f"{type(exc).__name__}: {exc!s}",
            },
        )


# ── Global exception handlers ─────────────────────────────────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so internal errors never leak stack traces to clients."""
    logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorDetail(
            error="internal_server_error",
            detail="An unexpected error occurred. Please try again later.",
        ).model_dump(),
    )


# ── Health / readiness probes ─────────────────────────────────────────────────


@app.get("/health", tags=["ops"], summary="Liveness probe.")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"], summary="Readiness probe – checks RapidAPI config.")
async def ready() -> JSONResponse:
    if not settings.rapidapi_configured:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "RapidAPI credentials missing."},
        )
    return JSONResponse(content={"status": "ready"})


# ── Startup / shutdown events ─────────────────────────────────────────────────


@app.on_event("startup")
async def on_startup() -> None:
    logger.info(
        "Wine Recognition API starting | rapidapi_configured=%s cors_origins=%s",
        settings.rapidapi_configured,
        settings.allowed_origins,
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Wine Recognition API shutting down.")
