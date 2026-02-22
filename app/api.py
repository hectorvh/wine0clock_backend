"""
API router – wine label recognition endpoints.

Endpoints
---------
POST /api/v1/recognize/file  – upload an image file (multipart/form-data)
POST /api/v1/recognize/url   – supply a publicly reachable image URL
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status

from app.config import Settings, get_settings
from app.schemas import ErrorDetail, RecognizeResponse, UrlRecognizeRequest, WineCandidate
from app.services.rapidapi_client import RapidAPIClient, RapidAPIError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/recognize", tags=["recognition"])


# ── Dependency helpers ────────────────────────────────────────────────────────


def _get_client(settings: Settings = Depends(get_settings)) -> RapidAPIClient:
    """Verify that RapidAPI is configured, then return a client instance."""
    if not settings.rapidapi_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RapidAPI credentials are not configured on this server.",
        )
    return RapidAPIClient(settings)


def _validated_top_k(
    top_k: int = Query(default=5, ge=1, le=10, description="Maximum candidates to return."),
    settings: Settings = Depends(get_settings),
) -> int:
    return min(top_k, settings.max_top_k)


# ── Shared validation logic ───────────────────────────────────────────────────


def _validate_upload_file(file: UploadFile, settings: Settings) -> None:
    """
    Raise ``HTTPException(422)`` if the uploaded file fails content-type or
    extension checks.  Size is validated after the bytes are read.
    """
    # Content-type check (provided by the client; not fully trusted but useful
    # for early rejection).
    if file.content_type and file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Allowed: {', '.join(sorted(settings.allowed_content_types))}"
            ),
        )

    # Extension check.
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed: {', '.join(sorted(settings.allowed_extensions))}"
            ),
        )


def _save_result_to_file(response: RecognizeResponse, settings: Settings) -> None:
    """If results_dir is configured, write the response as JSON to {request_id}.json."""
    if not settings.results_dir or not settings.results_dir.strip():
        return
    root = Path(__file__).resolve().parent.parent
    out_dir = root / settings.results_dir.strip()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{response.request_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)
        logger.debug("Saved result to %s", path)
    except OSError as e:
        logger.warning("Could not save result to %s: %s", out_dir, e)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/file",
    response_model=RecognizeResponse,
    responses={
        422: {"model": ErrorDetail, "description": "Invalid file / parameters."},
        503: {"model": ErrorDetail, "description": "RapidAPI not configured."},
        502: {"model": ErrorDetail, "description": "Upstream API error."},
    },
    summary="Recognise wine label from an uploaded image file.",
)
async def recognize_file(
    file: UploadFile,
    top_k: Annotated[int, Query(ge=1, le=10, description="Maximum candidates to return.")] = 5,
    include_raw: Annotated[bool, Query(description="Include full upstream JSON in response.")] = False,
    settings: Settings = Depends(get_settings),
    client: RapidAPIClient = Depends(_get_client),
) -> RecognizeResponse:
    """
    Accept a wine-label image (JPG / PNG / WebP, ≤ 10 MB) as multipart
    form-data and return recognition candidates ordered by confidence.
    """
    request_id = str(uuid.uuid4())

    # ── File validation ───────────────────────────────────────────────────────
    _validate_upload_file(file, settings)

    image_bytes = await file.read()
    file_size = len(image_bytes)

    logger.info(
        "File upload received | request_id=%s filename=%s size_bytes=%d content_type=%s",
        request_id,
        file.filename,
        file_size,
        file.content_type,
    )

    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"File too large: {file_size / 1024 / 1024:.1f} MB. "
                f"Maximum allowed: {settings.max_file_size_bytes // 1024 // 1024} MB."
            ),
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    # ── Call upstream API ─────────────────────────────────────────────────────
    try:
        candidates, raw, elapsed_ms = await client.recognize_file(
            image_bytes=image_bytes,
            filename=file.filename or "image.jpg",
            content_type=file.content_type or "image/jpeg",
            request_id=request_id,
        )
    except RapidAPIError as exc:
        logger.error(
            "RapidAPI error | request_id=%s error=%s status_code=%s",
            request_id,
            str(exc),
            exc.status_code,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream recognition API error: {exc}",
        ) from exc

    top_k = min(top_k, settings.max_top_k)
    top_candidates: list[WineCandidate] = candidates[:top_k]

    logger.info(
        "Recognition complete | request_id=%s candidates=%d elapsed_ms=%.1f",
        request_id,
        len(top_candidates),
        elapsed_ms,
    )

    response = RecognizeResponse(
        request_id=request_id,
        top_candidates=top_candidates,
        candidate_count=len(top_candidates),
        elapsed_ms=round(elapsed_ms, 2),
        raw_response=raw if include_raw else None,
    )
    _save_result_to_file(response, settings)
    return response


@router.post(
    "/url",
    response_model=RecognizeResponse,
    responses={
        422: {"model": ErrorDetail, "description": "Invalid URL / parameters."},
        503: {"model": ErrorDetail, "description": "RapidAPI not configured."},
        502: {"model": ErrorDetail, "description": "Upstream API error."},
    },
    summary="Recognise wine label from a publicly reachable image URL.",
)
async def recognize_url(
    body: UrlRecognizeRequest,
    top_k: Annotated[int, Query(ge=1, le=10, description="Maximum candidates to return.")] = 5,
    include_raw: Annotated[bool, Query(description="Include full upstream JSON in response.")] = False,
    settings: Settings = Depends(get_settings),
    client: RapidAPIClient = Depends(_get_client),
) -> RecognizeResponse:
    """
    Supply a publicly accessible image URL; the upstream API fetches and
    analyses it server-side.
    """
    request_id = str(uuid.uuid4())
    image_url = str(body.url)

    logger.info(
        "URL recognition request | request_id=%s url=%s",
        request_id,
        image_url,
    )

    # ── Call upstream API ─────────────────────────────────────────────────────
    try:
        candidates, raw, elapsed_ms = await client.recognize_url(
            image_url=image_url,
            request_id=request_id,
        )
    except RapidAPIError as exc:
        logger.error(
            "RapidAPI error | request_id=%s error=%s status_code=%s",
            request_id,
            str(exc),
            exc.status_code,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream recognition API error: {exc}",
        ) from exc

    top_k = min(top_k, settings.max_top_k)
    top_candidates: list[WineCandidate] = candidates[:top_k]

    logger.info(
        "Recognition complete | request_id=%s candidates=%d elapsed_ms=%.1f",
        request_id,
        len(top_candidates),
        elapsed_ms,
    )

    response = RecognizeResponse(
        request_id=request_id,
        top_candidates=top_candidates,
        candidate_count=len(top_candidates),
        elapsed_ms=round(elapsed_ms, 2),
        raw_response=raw if include_raw else None,
    )
    _save_result_to_file(response, settings)
    return response
