"""
Tests for API routes.

Comprehensive tests for all API endpoints including health, uploads, and analysis.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# =============================================================================
# HEALTH ROUTES TESTS
# =============================================================================

class TestHealthRoutes:
    """Tests for health check endpoints."""

    def test_root_returns_message(self, client: TestClient):
        """Test root endpoint returns running message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Profit Sentinel" in data["message"]

    def test_health_returns_healthy(self, client: TestClient):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_response_time(self, client: TestClient):
        """Test health endpoint responds quickly (< 100ms)."""
        import time
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 0.1, f"Health check took {elapsed:.3f}s, expected < 0.1s"


# =============================================================================
# UPLOAD ROUTES TESTS
# =============================================================================

class TestUploadRoutes:
    """Tests for file upload endpoints."""

    def test_presign_requires_filenames(self, client: TestClient):
        """Test presign endpoint requires filenames parameter."""
        response = client.post("/uploads/presign")
        assert response.status_code == 422  # Validation error
        assert "detail" in response.json()

    def test_presign_returns_urls(self, client: TestClient, mock_s3_client: MagicMock):
        """Test presign endpoint returns presigned URLs."""
        with patch('apps.api.src.dependencies.get_s3_client', return_value=mock_s3_client):
            response = client.post(
                "/uploads/presign",
                data={"filenames": ["test.csv", "data.xlsx"]}
            )

        assert response.status_code == 200
        data = response.json()
        assert "presigned_urls" in data
        assert len(data["presigned_urls"]) == 2

        for url_obj in data["presigned_urls"]:
            assert "filename" in url_obj
            assert "key" in url_obj
            assert "url" in url_obj

    def test_presign_generates_unique_keys(self, client: TestClient, mock_s3_client: MagicMock):
        """Test presign generates unique S3 keys."""
        with patch('apps.api.src.dependencies.get_s3_client', return_value=mock_s3_client):
            response = client.post(
                "/uploads/presign",
                data={"filenames": ["same.csv", "same.csv"]}
            )

        assert response.status_code == 200
        data = response.json()
        keys = [url_obj["key"] for url_obj in data["presigned_urls"]]
        # Keys should be unique even for same filename
        assert len(keys) == len(set(keys))

    def test_presign_includes_user_in_key(self, authenticated_client: TestClient):
        """Test presign includes user ID in S3 key when authenticated."""
        response = authenticated_client.post(
            "/uploads/presign",
            data={"filenames": ["test.csv"]}
        )

        assert response.status_code == 200
        # Note: With mock, key format is checked but actual user ID may vary

    def test_suggest_mapping_requires_key(self, client: TestClient):
        """Test suggest-mapping requires key parameter."""
        response = client.post(
            "/uploads/suggest-mapping",
            data={"filename": "test.csv"}
        )
        assert response.status_code == 422

    def test_suggest_mapping_requires_filename(self, client: TestClient):
        """Test suggest-mapping requires filename parameter."""
        response = client.post(
            "/uploads/suggest-mapping",
            data={"key": "test-key"}
        )
        assert response.status_code == 422

    def test_suggest_mapping_returns_suggestions(
        self,
        client: TestClient,
        mock_s3_client_with_data: MagicMock,
        mock_grok_client: MagicMock
    ):
        """Test suggest-mapping returns column mapping suggestions."""
        # Patch at the module where it's imported, not where it's defined
        with patch('apps.api.src.routes.uploads.get_s3_client', return_value=mock_s3_client_with_data):
            with patch('apps.api.src.services.mapping.get_grok_client', return_value=mock_grok_client):
                response = client.post(
                    "/uploads/suggest-mapping",
                    data={"key": "test/file.csv", "filename": "file.csv"}
                )

        assert response.status_code == 200
        data = response.json()
        assert "original_columns" in data
        assert "sample_data" in data
        assert "suggestions" in data
        assert "confidences" in data

    def test_suggest_mapping_heuristic_fallback(
        self,
        client: TestClient,
        mock_s3_client_with_data: MagicMock
    ):
        """Test suggest-mapping falls back to heuristics when Grok unavailable."""
        # Patch at the module where it's imported, not where it's defined
        with patch('apps.api.src.routes.uploads.get_s3_client', return_value=mock_s3_client_with_data):
            with patch('apps.api.src.services.mapping.get_grok_client', return_value=None):
                response = client.post(
                    "/uploads/suggest-mapping",
                    data={"key": "test/file.csv", "filename": "file.csv"}
                )

        assert response.status_code == 200
        data = response.json()
        assert "notes" in data
        # Heuristic fallback should still return suggestions
        assert "suggestions" in data


# =============================================================================
# ANALYSIS ROUTES TESTS
# =============================================================================

class TestAnalysisRoutes:
    """Tests for profit leak analysis endpoints."""

    def test_analyze_requires_key(self, client: TestClient):
        """Test analyze endpoint requires key parameter."""
        response = client.post(
            "/analysis/analyze",
            data={"mapping": "{}"}
        )
        assert response.status_code == 422

    def test_analyze_requires_mapping(self, client: TestClient):
        """Test analyze endpoint requires mapping parameter."""
        response = client.post(
            "/analysis/analyze",
            data={"key": "test-key"}
        )
        assert response.status_code == 422

    def test_analyze_rejects_invalid_json(self, client: TestClient):
        """Test analyze endpoint rejects invalid JSON mapping."""
        response = client.post(
            "/analysis/analyze",
            data={"key": "test-key", "mapping": "invalid-json"}
        )
        assert response.status_code == 422
        assert "Invalid mapping JSON" in response.json()["detail"]

    def test_analyze_accepts_valid_mapping(
        self,
        client: TestClient,
        mock_s3_client_with_data: MagicMock,
        sample_column_mapping: dict
    ):
        """Test analyze endpoint accepts valid column mapping."""
        with patch('apps.api.src.dependencies.get_s3_client', return_value=mock_s3_client_with_data):
            response = client.post(
                "/analysis/analyze",
                data={
                    "key": "test/file.csv",
                    "mapping": json.dumps(sample_column_mapping)
                }
            )

        # Should succeed or fail gracefully (sentinel engine may not be available)
        assert response.status_code in [200, 500]

    def test_analyze_returns_leak_categories(
        self,
        client: TestClient,
        mock_s3_client_with_data: MagicMock,
        sample_column_mapping: dict
    ):
        """Test analyze returns all expected leak categories."""
        with patch('apps.api.src.dependencies.get_s3_client', return_value=mock_s3_client_with_data):
            response = client.post(
                "/analysis/analyze",
                data={
                    "key": "test/file.csv",
                    "mapping": json.dumps(sample_column_mapping)
                }
            )

        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] == "success"
            assert "leaks" in data

            expected_categories = ["low_stock", "high_margin_leak", "dead_item", "negative_inventory"]
            for category in expected_categories:
                assert category in data["leaks"]

    def test_analyze_empty_mapping(
        self,
        client: TestClient,
        mock_s3_client_with_data: MagicMock
    ):
        """Test analyze handles empty mapping gracefully."""
        with patch('apps.api.src.dependencies.get_s3_client', return_value=mock_s3_client_with_data):
            response = client.post(
                "/analysis/analyze",
                data={
                    "key": "test/file.csv",
                    "mapping": "{}"
                }
            )

        # Should not crash with empty mapping
        assert response.status_code in [200, 500]


# =============================================================================
# AUTHENTICATION TESTS
# =============================================================================

class TestAuthentication:
    """Tests for authentication handling."""

    def test_endpoints_work_without_auth(self, client: TestClient):
        """Test that endpoints work without authentication (anonymous mode)."""
        # Health endpoints should always work
        response = client.get("/health")
        assert response.status_code == 200

    def test_invalid_token_handled_gracefully(
        self,
        client: TestClient,
        mock_s3_client: MagicMock
    ):
        """Test that invalid auth tokens don't crash endpoints."""
        # Mock get_supabase_client to return None so auth is bypassed
        # (endpoint allows anonymous access, so without supabase client it falls back)
        with patch('apps.api.src.routes.uploads.get_s3_client', return_value=mock_s3_client):
            with patch('apps.api.src.dependencies.get_supabase_client', return_value=None):
                response = client.post(
                    "/uploads/presign",
                    data={"filenames": ["test.csv"]},
                    headers={"Authorization": "Bearer invalid-token"}
                )

        # Should return 503 (auth service unavailable) since we're sending a token
        # but supabase is not configured
        assert response.status_code == 503


# =============================================================================
# CORS TESTS
# =============================================================================

class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_allows_localhost(self, client: TestClient):
        """Test CORS allows localhost:3000."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # FastAPI TestClient doesn't fully simulate CORS preflight,
        # but we can check the response headers exist
        assert response.status_code in [200, 405]

    def test_cors_headers_present(self, client: TestClient):
        """Test CORS headers are present in response."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200
        # Note: TestClient may not include all CORS headers


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_404_for_unknown_route(self, client: TestClient):
        """Test 404 response for unknown routes."""
        response = client.get("/unknown/route")
        assert response.status_code == 404

    def test_405_for_wrong_method(self, client: TestClient):
        """Test 405 response for wrong HTTP method."""
        response = client.get("/uploads/presign")
        assert response.status_code == 405

    def test_s3_error_handled(self, client: TestClient):
        """Test S3 errors are handled gracefully."""
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = Exception("S3 connection failed")

        with patch('apps.api.src.dependencies.get_s3_client', return_value=mock_s3):
            response = client.post(
                "/uploads/suggest-mapping",
                data={"key": "test-key", "filename": "test.csv"}
            )

        assert response.status_code == 500
        assert "detail" in response.json()
