"""
Pytest test suite for the Wine Label Recognition API.

Test categories
---------------
1. File validation  – wrong extension, wrong content-type, too large, empty.
2. Successful mock  – happy-path file upload and URL recognition.
3. Error handling   – upstream timeout and non-200 status codes.

All RapidAPI calls are intercepted with ``httpx.MockTransport`` / ``respx``
or a manual ``unittest.mock.AsyncMock`` so no real network requests are made.
"""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.rapidapi_client import RapidAPIClient, RapidAPIError

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_settings(**overrides) -> Settings:
    """Return a Settings instance pre-loaded with test credentials."""
    defaults = dict(
        rapidapi_key="test-key-123",
        rapidapi_host="wine-recognition2.p.rapidapi.com",
        frontend_origin="http://localhost:3000",
    )
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[call-arg]


@pytest.fixture()
def settings() -> Settings:
    return _make_settings()


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    """TestClient with the real app but overridden settings."""
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


# Minimal valid 1×1 PNG in bytes (no real image data needed for routing tests)
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Sample api4ai-shaped response
_MOCK_API_RESPONSE = {
    "results": [
        {
            "status": {"code": "ok"},
            "entities": [
                {
                    "kind": "wine",
                    "classes": [
                        {"class": "Château Margaux 2015", "score": 0.92},
                        {"class": "Château Latour 2016", "score": 0.75},
                        {"class": "Penfolds Grange 2018", "score": 0.61},
                    ],
                }
            ],
        }
    ]
}


# ── Helper to mock httpx inside RapidAPIClient ─────────────────────────────────


def _mock_httpx_response(status_code: int = 200, json_body: dict | None = None, exc=None):
    """
    Returns an async context manager mock whose ``__aenter__`` returns a mock
    httpx.AsyncClient that responds with the given status_code / json_body.
    """
    if exc is not None:
        # Raise exception on .post()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=exc)
    else:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.json.return_value = json_body or {}
        mock_response.text = json.dumps(json_body or {})

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FILE VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFileValidation:
    """Ensure invalid uploads are rejected before reaching the upstream API."""

    def test_unsupported_extension_rejected(self, client: TestClient) -> None:
        """GIF images must be rejected with 422."""
        response = client.post(
            "/api/v1/recognize/file",
            files={"file": ("label.gif", _TINY_PNG, "image/gif")},
        )
        assert response.status_code == 422
        assert "gif" in response.text.lower() or "extension" in response.text.lower()

    def test_unsupported_content_type_rejected(self, client: TestClient) -> None:
        """application/pdf content type must be rejected with 422."""
        response = client.post(
            "/api/v1/recognize/file",
            files={"file": ("label.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 422

    def test_file_too_large_rejected(self, client: TestClient) -> None:
        """Files larger than 10 MB must be rejected with 422."""
        big_file = b"x" * (10 * 1024 * 1024 + 1)
        response = client.post(
            "/api/v1/recognize/file",
            files={"file": ("label.jpg", big_file, "image/jpeg")},
        )
        assert response.status_code == 422
        assert "large" in response.text.lower() or "size" in response.text.lower()

    def test_empty_file_rejected(self, client: TestClient) -> None:
        """Empty files must be rejected with 422."""
        response = client.post(
            "/api/v1/recognize/file",
            files={"file": ("label.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 422

    def test_valid_png_passes_validation(self, client: TestClient) -> None:
        """
        A valid PNG should pass validation and reach the upstream call.
        We mock the upstream so the test remains self-contained.
        """
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, _MOCK_API_RESPONSE)):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.png", _TINY_PNG, "image/png")},
            )
        # Either 200 (mock worked) or 502 (mock wiring issue) – NOT 422.
        assert response.status_code != 422, response.text

    def test_valid_jpeg_passes_validation(self, client: TestClient) -> None:
        """JPEGs with .jpeg extension should also be accepted."""
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, _MOCK_API_RESPONSE)):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.jpeg", _TINY_PNG, "image/jpeg")},
            )
        assert response.status_code != 422, response.text


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SUCCESSFUL MOCK RAPIDAPI CALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSuccessfulRecognition:
    """Happy-path tests with a mocked upstream API."""

    def test_file_upload_returns_candidates(self, client: TestClient) -> None:
        """POST /file with a valid image should return top candidates."""
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, _MOCK_API_RESPONSE)):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.png", _TINY_PNG, "image/png")},
                params={"top_k": 3},
            )

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "top_candidates" in data
        assert len(data["top_candidates"]) <= 3
        # First candidate should be highest confidence
        if len(data["top_candidates"]) > 1:
            assert (
                data["top_candidates"][0]["confidence"]
                >= data["top_candidates"][1]["confidence"]
            )

    def test_file_upload_candidate_fields(self, client: TestClient) -> None:
        """Each candidate must have label and confidence fields."""
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, _MOCK_API_RESPONSE)):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.jpg", _TINY_PNG, "image/jpeg")},
            )
        assert response.status_code == 200
        candidates = response.json()["top_candidates"]
        for candidate in candidates:
            assert "label" in candidate
            assert "confidence" in candidate
            assert isinstance(candidate["confidence"], float)
            assert 0.0 <= candidate["confidence"] <= 1.0

    def test_url_recognition_returns_candidates(self, client: TestClient) -> None:
        """POST /url with a valid URL should return recognition results."""
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, _MOCK_API_RESPONSE)):
            response = client.post(
                "/api/v1/recognize/url",
                json={"url": "https://example.com/wine-label.jpg"},
                params={"top_k": 2},
            )
        assert response.status_code == 200
        data = response.json()
        assert len(data["top_candidates"]) <= 2

    def test_include_raw_flag_adds_raw_response(self, client: TestClient) -> None:
        """When include_raw=true the response must contain the raw_response field."""
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, _MOCK_API_RESPONSE)):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.png", _TINY_PNG, "image/png")},
                params={"include_raw": "true"},
            )
        assert response.status_code == 200
        assert response.json()["raw_response"] is not None

    def test_include_raw_false_omits_raw_response(self, client: TestClient) -> None:
        """raw_response must be null when include_raw=false (default)."""
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, _MOCK_API_RESPONSE)):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.png", _TINY_PNG, "image/png")},
            )
        assert response.status_code == 200
        assert response.json()["raw_response"] is None

    def test_top_k_capped_at_max(self, client: TestClient) -> None:
        """top_k > 10 should be rejected at the query-param level (FastAPI validates)."""
        response = client.post(
            "/api/v1/recognize/file",
            files={"file": ("label.png", _TINY_PNG, "image/png")},
            params={"top_k": 99},
        )
        # FastAPI returns 422 for out-of-range Query params.
        assert response.status_code == 422

    def test_elapsed_ms_present_and_positive(self, client: TestClient) -> None:
        """elapsed_ms must be a positive number."""
        with patch("httpx.AsyncClient", return_value=_mock_httpx_response(200, _MOCK_API_RESPONSE)):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.jpg", _TINY_PNG, "image/jpeg")},
            )
        assert response.status_code == 200
        assert response.json()["elapsed_ms"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ERROR HANDLING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Ensure the API handles upstream failures gracefully."""

    def test_upstream_timeout_returns_502(self, client: TestClient) -> None:
        """httpx.TimeoutException from the upstream should yield HTTP 502."""
        with patch(
            "httpx.AsyncClient",
            return_value=_mock_httpx_response(exc=httpx.TimeoutException("timed out")),
        ):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.jpg", _TINY_PNG, "image/jpeg")},
            )
        assert response.status_code == 502

    def test_upstream_500_returns_502(self, client: TestClient) -> None:
        """HTTP 500 from the upstream API should yield HTTP 502 to the client."""
        with patch(
            "httpx.AsyncClient",
            return_value=_mock_httpx_response(500, {"error": "internal"}),
        ):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.jpg", _TINY_PNG, "image/jpeg")},
            )
        assert response.status_code == 502

    def test_upstream_401_returns_502(self, client: TestClient) -> None:
        """A 401 from RapidAPI (bad key) should propagate as 502."""
        with patch(
            "httpx.AsyncClient",
            return_value=_mock_httpx_response(401, {"message": "Invalid API key"}),
        ):
            response = client.post(
                "/api/v1/recognize/file",
                files={"file": ("label.png", _TINY_PNG, "image/png")},
            )
        assert response.status_code == 502

    def test_missing_api_key_returns_503(self) -> None:
        """When RapidAPI credentials are absent the server should return 503."""
        unconfigured_settings = _make_settings(rapidapi_key="", rapidapi_host="")
        app.dependency_overrides[get_settings] = lambda: unconfigured_settings
        try:
            with TestClient(app) as c:
                response = c.post(
                    "/api/v1/recognize/file",
                    files={"file": ("label.png", _TINY_PNG, "image/png")},
                )
            assert response.status_code == 503
        finally:
            app.dependency_overrides.clear()

    def test_url_with_invalid_url_returns_422(self, client: TestClient) -> None:
        """A malformed URL body should be rejected by Pydantic with 422."""
        response = client.post(
            "/api/v1/recognize/url",
            json={"url": "not-a-url"},
        )
        assert response.status_code == 422

    def test_url_missing_body_returns_422(self, client: TestClient) -> None:
        """Empty body to /url should return 422."""
        response = client.post("/api/v1/recognize/url", json={})
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# 4. UNIT TESTS – RapidAPIClient._parse_candidates
# ═══════════════════════════════════════════════════════════════════════════════


class TestParsecandidates:
    """Unit-test the private parsing logic without touching HTTP."""

    def test_parses_standard_response(self) -> None:
        candidates = RapidAPIClient._parse_candidates(_MOCK_API_RESPONSE)
        assert len(candidates) == 3
        assert candidates[0].label == "Château Margaux 2015"
        assert candidates[0].confidence == pytest.approx(0.92)

    def test_sorted_descending(self) -> None:
        candidates = RapidAPIClient._parse_candidates(_MOCK_API_RESPONSE)
        scores = [c.confidence for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_skips_error_status_results(self) -> None:
        bad_response = {
            "results": [
                {
                    "status": {"code": "error", "message": "recognition failed"},
                    "entities": [
                        {"kind": "wine", "classes": [{"class": "Some Wine", "score": 0.9}]}
                    ],
                }
            ]
        }
        candidates = RapidAPIClient._parse_candidates(bad_response)
        assert len(candidates) == 0

    def test_empty_results(self) -> None:
        candidates = RapidAPIClient._parse_candidates({"results": []})
        assert candidates == []

    def test_no_results_key(self) -> None:
        candidates = RapidAPIClient._parse_candidates({})
        assert candidates == []

    def test_ignores_empty_labels(self) -> None:
        response = {
            "results": [
                {
                    "status": {"code": "ok"},
                    "entities": [
                        {
                            "kind": "wine",
                            "classes": [
                                {"class": "", "score": 0.8},      # empty label
                                {"class": "Real Wine", "score": 0.7},
                            ],
                        }
                    ],
                }
            ]
        }
        candidates = RapidAPIClient._parse_candidates(response)
        assert len(candidates) == 1
        assert candidates[0].label == "Real Wine"

    def test_parses_dict_classes_format(self) -> None:
        """When entity['classes'] is a dict (label -> score), parse into candidates."""
        response = {
            "results": [
                {
                    "status": {"code": "ok"},
                    "entities": [
                        {
                            "kind": "classes",
                            "name": "wine-image-classes",
                            "classes": {
                                "aldi primitivo": 0.825,
                                "latentia winery primitivo 2016": 0.77,
                                "torresanta primitivo": 0.704,
                            },
                        }
                    ],
                }
            ]
        }
        candidates = RapidAPIClient._parse_candidates(response)
        assert len(candidates) == 3
        assert candidates[0].label == "aldi primitivo"
        assert candidates[0].confidence == pytest.approx(0.825)
        assert candidates[1].label == "latentia winery primitivo 2016"
        assert candidates[1].confidence == pytest.approx(0.77)
        assert candidates[2].label == "torresanta primitivo"
        assert candidates[2].confidence == pytest.approx(0.704)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HEALTH / READINESS PROBES
# ═══════════════════════════════════════════════════════════════════════════════


class TestProbes:
    def test_health_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_ready_configured(self, client: TestClient) -> None:
        assert client.get("/ready").status_code == 200

    def test_ready_unconfigured_returns_503(self) -> None:
        unconfigured = _make_settings(rapidapi_key="", rapidapi_host="")
        app.dependency_overrides[get_settings] = lambda: unconfigured
        try:
            with TestClient(app) as c:
                assert c.get("/ready").status_code == 503
        finally:
            app.dependency_overrides.clear()
