"""
RapidAPI wine-recognition2 HTTP client.

Responsibilities
----------------
* Build the correct multipart / JSON request for api4ai.
* Enforce per-request timeouts and simple retry logic.
* Parse the api4ai response envelope into our internal schema.
* Never log secrets; never persist image bytes to disk.

api4ai response envelope (simplified):
{
  "results": [
    {
      "status": { "code": "ok" },
      "entities": [
        {
          "kind": "wine",
          "classes": [
            { "class": "Château Margaux 2015", "score": 0.91 },
            ...
          ]
        }
      ]
    }
  ]
}
"""

import logging
import time
from typing import Any

import httpx

from app.config import Settings
from app.schemas import WineCandidate

logger = logging.getLogger(__name__)

# Transient HTTP status codes that warrant a single retry.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class RapidAPIError(Exception):
    """Raised when the upstream API returns a non-successful response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RapidAPIClient:
    """
    Thin async wrapper around the api4ai wine-recognition2 endpoint.

    Parameters
    ----------
    settings:
        Application settings; the client reads API credentials from here.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.rapidapi_base_url
        self._results_path = settings.rapidapi_results_path
        self._timeout = httpx.Timeout(settings.http_timeout_seconds)
        self._max_retries = settings.http_max_retries
        # Build auth headers once; never include them in logs.
        self._auth_headers = {
            "X-RapidAPI-Key": settings.rapidapi_key,
            "X-RapidAPI-Host": settings.rapidapi_host,
        }
        self._version_path = settings.rapidapi_version_path

    # ── Public helpers ────────────────────────────────────────────────────────

    async def get_version(self) -> dict[str, Any]:
        """
        GET the upstream API version (no image/body). Used to verify connectivity.
        Raises RapidAPIError on non-200, timeout, or invalid response.
        """
        url = self._base_url + self._version_path
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._auth_headers,
                    timeout=self._timeout,
                )
                if response.status_code != 200:
                    raise RapidAPIError(
                        f"Version endpoint returned HTTP {response.status_code}: {response.text[:200]!r}",
                        status_code=response.status_code,
                    )
                try:
                    return response.json()
                except ValueError as e:
                    raise RapidAPIError(f"Version endpoint returned non-JSON: {e}") from e
        except httpx.TimeoutException as e:
            raise RapidAPIError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise RapidAPIError(f"Request failed: {e}") from e

    async def recognize_file(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        request_id: str,
    ) -> tuple[list[WineCandidate], dict[str, Any], float]:
        """
        Send raw image bytes to the api4ai endpoint.

        Returns
        -------
        (candidates, raw_json, elapsed_ms)
        """
        url = self._base_url + self._results_path

        async def _do_request(client: httpx.AsyncClient) -> httpx.Response:
            files = {"image": (filename, image_bytes, content_type)}
            return await client.post(
                url,
                headers=self._auth_headers,
                files=files,
                timeout=self._timeout,
            )

        return await self._call_with_retry(_do_request, request_id)

    async def recognize_url(
        self,
        image_url: str,
        request_id: str,
    ) -> tuple[list[WineCandidate], dict[str, Any], float]:
        """
        Ask the api4ai endpoint to fetch and recognise an image from a URL.

        Returns
        -------
        (candidates, raw_json, elapsed_ms)
        """
        url = self._base_url + self._results_path

        async def _do_request(client: httpx.AsyncClient) -> httpx.Response:
            return await client.post(
                url,
                headers=self._auth_headers,
                data={"url": image_url},
                timeout=self._timeout,
            )

        return await self._call_with_retry(_do_request, request_id)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _call_with_retry(
        self,
        request_fn,
        request_id: str,
    ) -> tuple[list[WineCandidate], dict[str, Any], float]:
        """
        Execute *request_fn* with up to ``self._max_retries`` retries on
        transient failures.  Returns ``(candidates, raw_json, elapsed_ms)``.
        """
        attempts = self._max_retries + 1
        last_exc: Exception | None = None

        async with httpx.AsyncClient() as client:
            for attempt in range(1, attempts + 1):
                t0 = time.perf_counter()
                try:
                    response = await request_fn(client)
                    elapsed_ms = (time.perf_counter() - t0) * 1_000

                    logger.info(
                        "RapidAPI response | request_id=%s attempt=%d status=%d elapsed_ms=%.1f",
                        request_id,
                        attempt,
                        response.status_code,
                        elapsed_ms,
                    )

                    if response.status_code == 200:
                        try:
                            raw = response.json()
                        except ValueError as e:
                            raise RapidAPIError(f"Upstream returned invalid JSON: {e}") from e
                        if not isinstance(raw, dict):
                            raise RapidAPIError("Upstream response is not a JSON object")
                        candidates = self._parse_candidates(raw)
                        return candidates, raw, elapsed_ms

                    if response.status_code in _RETRYABLE_STATUS_CODES and attempt < attempts:
                        logger.warning(
                            "Retryable status %d for request_id=%s, retrying (attempt %d/%d)…",
                            response.status_code,
                            request_id,
                            attempt,
                            attempts,
                        )
                        last_exc = RapidAPIError(
                            f"Upstream returned {response.status_code}",
                            status_code=response.status_code,
                        )
                        continue

                    # Non-retryable non-200
                    raise RapidAPIError(
                        f"Upstream API returned HTTP {response.status_code}: {response.text[:200]}",
                        status_code=response.status_code,
                    )

                except httpx.TimeoutException as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1_000
                    logger.warning(
                        "Timeout calling RapidAPI | request_id=%s attempt=%d elapsed_ms=%.1f",
                        request_id,
                        attempt,
                        elapsed_ms,
                    )
                    last_exc = RapidAPIError(f"Request timed out after {elapsed_ms:.0f} ms")
                    if attempt == attempts:
                        break
                    continue
                except httpx.RequestError as exc:
                    logger.warning("Request error calling RapidAPI | request_id=%s error=%s", request_id, exc)
                    last_exc = RapidAPIError(f"Request failed: {exc}")
                    if attempt == attempts:
                        break
                    continue

        raise last_exc or RapidAPIError("All retry attempts exhausted.")

    @staticmethod
    def _parse_candidates(raw: dict[str, Any]) -> list[WineCandidate]:
        """
        Extract ``WineCandidate`` objects from the api4ai response envelope.

        The api4ai response wraps entities inside ``results[*].entities``.
        Each entity with ``kind == "wine"`` contains a ``classes`` list of
        ``{class: str, score: float}`` objects.
        """
        candidates: list[WineCandidate] = []

        results: list[dict] = raw.get("results", [])
        for result in results:
            # Check the per-result status to avoid processing error blocks.
            status = result.get("status", {})
            if status.get("code", "").lower() not in ("ok", "success", ""):
                logger.debug("Skipping result with status: %s", status)
                continue

            entities: list[dict] = result.get("entities", [])
            for entity in entities:
                classes: list[dict] = entity.get("classes", [])
                for cls in classes:
                    if not isinstance(cls, dict):
                        continue
                    label = (cls.get("class") or "").strip()
                    try:
                        confidence = float(cls.get("score") or 0.0)
                    except (TypeError, ValueError):
                        confidence = 0.0
                    if label:
                        candidates.append(
                            WineCandidate(label=label, confidence=confidence)
                        )

        # Sort descending by confidence (schema validator also does this, but
        # we do it here too so callers of this static method get sorted data).
        return sorted(candidates, key=lambda c: c.confidence, reverse=True)
