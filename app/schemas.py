"""
Pydantic schemas for request validation and response serialisation.

Using explicit models (rather than raw dicts) keeps the API contract
clear for both the frontend and any auto-generated OpenAPI docs.
"""

from typing import Any

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator


# ── Request bodies ────────────────────────────────────────────────────────────


class UrlRecognizeRequest(BaseModel):
    """Body accepted by POST /api/v1/recognize/url."""

    url: AnyHttpUrl = Field(..., description="Publicly reachable URL of the wine-label image.")


# ── Response bodies ───────────────────────────────────────────────────────────


class WineCandidate(BaseModel):
    """A single label-recognition candidate returned by the upstream API."""

    label: str = Field(..., description="Human-readable wine name / label text.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score in the [0, 1] range."
    )


class RecognizeResponse(BaseModel):
    """Unified response schema returned by both /file and /url endpoints."""

    request_id: str = Field(..., description="UUID generated for this request (useful for logs).")
    top_candidates: list[WineCandidate] = Field(
        default_factory=list,
        description="Top-K candidates ordered by confidence (descending).",
    )
    candidate_count: int = Field(..., description="Total number of candidates returned.")
    elapsed_ms: float = Field(..., description="Time taken to call the upstream API (ms).")
    raw_response: dict[str, Any] | None = Field(
        default=None,
        description="Full raw JSON from the upstream API, only included when include_raw=true.",
    )

    @field_validator("top_candidates", mode="before")
    @classmethod
    def sort_by_confidence(cls, candidates: list) -> list:
        """Guarantee descending confidence order regardless of upstream ordering."""
        return sorted(candidates, key=lambda c: c.get("confidence", 0) if isinstance(c, dict) else c.confidence, reverse=True)


class ErrorDetail(BaseModel):
    """Structured error payload."""

    error: str
    detail: str | None = None
    request_id: str | None = None
