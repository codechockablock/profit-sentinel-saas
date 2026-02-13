"""Tests for the legacy-compatible upload & analysis endpoints.

Tests the 3 endpoints added for production frontend compatibility:
    POST /uploads/presign
    POST /uploads/suggest-mapping
    POST /analysis/analyze
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sentinel_agent.sidecar import create_app
from sentinel_agent.sidecar_config import SidecarSettings


@pytest.fixture
def dev_settings():
    """Settings with dev mode enabled (no auth) and S3 configured."""
    return SidecarSettings(
        sidecar_dev_mode=True,
        s3_bucket_name="test-bucket",
        aws_region="us-east-1",
        sentinel_default_store="default",
        sentinel_top_k=20,
    )


@pytest.fixture
def client(dev_settings):
    """Test client with dev mode (no auth required)."""
    app = create_app(settings=dev_settings)
    return TestClient(app)


# ---- Test /uploads/presign ----


class TestPresign:
    """Tests for POST /uploads/presign."""

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_single_file(self, mock_s3_factory, client):
        """Should return a presigned URL for a single file."""
        mock_client = MagicMock()
        mock_client.generate_presigned_post.return_value = {
            "url": "https://s3.amazonaws.com/test-bucket",
            "fields": {"key": "dev-user/uuid-inventory.csv"},
        }
        mock_s3_factory.return_value = mock_client

        response = client.post(
            "/uploads/presign",
            data={"filenames": ["inventory.csv"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "presigned_urls" in data
        assert len(data["presigned_urls"]) == 1
        assert data["presigned_urls"][0]["filename"] == "inventory.csv"
        assert "key" in data["presigned_urls"][0]
        assert "url" in data["presigned_urls"][0]
        assert data["limits"]["max_file_size_mb"] == 50

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_multiple_files(self, mock_s3_factory, client):
        """Should return presigned URLs for multiple files."""
        mock_client = MagicMock()
        mock_client.generate_presigned_post.return_value = {
            "url": "https://s3.example.com",
            "fields": {"key": "signed"},
        }
        mock_s3_factory.return_value = mock_client

        response = client.post(
            "/uploads/presign",
            data={"filenames": ["a.csv", "b.xlsx", "c.xls"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["presigned_urls"]) == 3

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_too_many_files(self, mock_s3_factory, client):
        """Should reject requests with too many files."""
        mock_s3_factory.return_value = MagicMock()

        response = client.post(
            "/uploads/presign",
            data={"filenames": ["a.csv", "b.csv", "c.csv", "d.csv", "e.csv", "f.csv"]},
        )
        assert response.status_code == 400
        assert "Too many files" in response.json()["detail"]

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_invalid_extension(self, mock_s3_factory, client):
        """Should reject files with disallowed extensions."""
        mock_s3_factory.return_value = MagicMock()

        response = client.post(
            "/uploads/presign",
            data={"filenames": ["malware.exe"]},
        )
        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"]

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_empty_filenames(self, mock_s3_factory, client):
        """Should reject empty filename list."""
        mock_s3_factory.return_value = MagicMock()

        response = client.post(
            "/uploads/presign",
            data={"filenames": []},
        )
        # FastAPI will return 422 for missing required field
        assert response.status_code in (400, 422)

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_key_format(self, mock_s3_factory, client):
        """Key should include user_id prefix and UUID."""
        mock_client = MagicMock()
        mock_client.generate_presigned_post.return_value = {
            "url": "https://s3.example.com",
            "fields": {"key": "signed"},
        }
        mock_s3_factory.return_value = mock_client

        response = client.post(
            "/uploads/presign",
            data={"filenames": ["inventory.csv"]},
        )
        key = response.json()["presigned_urls"][0]["key"]
        # In dev mode, user_id is "dev-user" with uploads/ prefix
        assert key.startswith("uploads/dev-user/")
        assert "inventory.csv" in key

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_allowed_extensions(self, mock_s3_factory, client):
        """Should include allowed extensions in response."""
        mock_client = MagicMock()
        mock_client.generate_presigned_post.return_value = {
            "url": "https://s3.example.com",
            "fields": {"key": "signed"},
        }
        mock_s3_factory.return_value = mock_client

        response = client.post(
            "/uploads/presign",
            data={"filenames": ["test.csv"]},
        )
        limits = response.json()["limits"]
        assert ".csv" in limits["allowed_extensions"]
        assert ".xlsx" in limits["allowed_extensions"]


# ---- Test /uploads/suggest-mapping ----


class TestSuggestMapping:
    """Tests for POST /uploads/suggest-mapping."""

    @patch("sentinel_agent.upload_routes.load_dataframe")
    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_suggest_mapping_basic(self, mock_s3_factory, mock_load_df, client):
        """Should return column mapping suggestions."""
        import pandas as pd

        mock_s3_factory.return_value = MagicMock()
        mock_load_df.return_value = pd.DataFrame(
            {
                "SKU": ["ABC-001", "DEF-002"],
                "Qty.": [10, 20],
                "Cost": [5.99, 12.50],
                "Retail": [9.99, 24.99],
                "Sold": [3, 7],
            }
        )

        response = client.post(
            "/uploads/suggest-mapping",
            data={
                "key": "uploads/dev-user/test-uuid-inventory.csv",
                "filename": "inventory.csv",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert "original_columns" in data
        assert "confidences" in data
        # Heuristics should map Qty. -> quantity, Cost -> cost, etc.
        assert data["suggestions"].get("SKU") == "sku"
        assert data["suggestions"].get("Qty.") == "quantity"
        assert data["suggestions"].get("Cost") == "cost"

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_suggest_mapping_access_denied(self, mock_s3_factory, client):
        """Should deny access to files owned by other users."""
        mock_s3_factory.return_value = MagicMock()

        response = client.post(
            "/uploads/suggest-mapping",
            data={
                "key": "other-user/some-file.csv",
                "filename": "inventory.csv",
            },
        )
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    @patch("sentinel_agent.upload_routes.load_dataframe")
    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_suggest_mapping_paladin_columns(
        self, mock_s3_factory, mock_load_df, client
    ):
        """Should handle Paladin POS column names."""
        import pandas as pd

        mock_s3_factory.return_value = MagicMock()
        mock_load_df.return_value = pd.DataFrame(
            {
                "Barcode": ["123456789"],
                "In Stock Qty.": [50],
                "Cost": [10.00],
                "Sug. Retail": [19.99],
                "Sold": [15],
                "Profit Margin %": [50.0],
                "Last Pur.": ["01/15/2026"],
            }
        )

        response = client.post(
            "/uploads/suggest-mapping",
            data={
                "key": "uploads/dev-user/paladin-report.csv",
                "filename": "paladin-report.csv",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["suggestions"]["Barcode"] == "sku"
        assert data["suggestions"]["In Stock Qty."] == "quantity"


# ---- Test /analysis/analyze ----


class TestAnalyze:
    """Tests for POST /analysis/analyze."""

    def test_analyze_missing_mapping(self, client):
        """Should reject requests with invalid mapping JSON."""
        response = client.post(
            "/analysis/analyze",
            data={
                "key": "dev-user/test.csv",
                "mapping": "not-valid-json",
            },
        )
        assert response.status_code == 422
        assert "Invalid mapping JSON" in response.json()["detail"]

    def test_analyze_access_denied(self, client):
        """Should deny access to files owned by other users."""
        response = client.post(
            "/analysis/analyze",
            data={
                "key": "other-user/test.csv",
                "mapping": '{"Qty.": "quantity"}',
            },
        )
        assert response.status_code == 403

    def test_analyze_invalid_mapping_type(self, client):
        """Should reject non-object mapping."""
        response = client.post(
            "/analysis/analyze",
            data={
                "key": "dev-user/test.csv",
                "mapping": '["not", "a", "dict"]',
            },
        )
        assert response.status_code == 400
        assert "JSON object" in response.json()["detail"]


# ---- Test /analysis/primitives ----


class TestPrimitives:
    """Tests for GET /analysis/primitives."""

    def test_list_primitives(self, client):
        """Should return all 11 detection primitives."""
        response = client.get("/analysis/primitives")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 11
        assert "high_margin_leak" in data["primitives"]
        assert "negative_inventory" in data["primitives"]
        assert "dead_item" in data["primitives"]

    def test_primitives_have_metadata(self, client):
        """Each primitive should have title, severity, category."""
        response = client.get("/analysis/primitives")
        data = response.json()
        for key, info in data["primitives"].items():
            assert "title" in info
            assert "severity" in info
            assert "category" in info
            assert "recommendations" in info


# ---- Test /analysis/supported-pos ----


class TestSupportedPOS:
    """Tests for GET /analysis/supported-pos."""

    def test_list_supported_pos(self, client):
        """Should return supported POS systems."""
        response = client.get("/analysis/supported-pos")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 16
        assert "Paladin POS" in data["supported_systems"]
        assert "Square POS" in data["supported_systems"]


# ---- Test column adapter ----


class TestColumnAdapter:
    """Tests for the column adapter module."""

    def test_adapter_converts_dataframe(self):
        """Should convert a post-mapping DataFrame to Rust CSV."""
        import pandas as pd
        from sentinel_agent.column_adapter import ColumnAdapter

        df = pd.DataFrame(
            {
                "sku": ["SKU-001", "SKU-002"],
                "quantity": [10, 20],
                "cost": [5.99, 12.50],
                "revenue": [9.99, 24.99],
                "sold": [3, 7],
            }
        )

        adapter = ColumnAdapter()
        csv_path = adapter.to_rust_csv(df)
        assert csv_path.exists()

        # Read back and verify
        result = pd.read_csv(csv_path)
        assert len(result) == 2
        assert list(result.columns) == [
            "store_id",
            "sku",
            "qty_on_hand",
            "unit_cost",
            "margin_pct",
            "sales_last_30d",
            "days_since_receipt",
            "retail_price",
            "is_damaged",
            "on_order_qty",
            "is_seasonal",
        ]
        assert result["sku"].tolist() == ["SKU-001", "SKU-002"]
        adapter.cleanup()


# ---- Test result adapter ----


class TestResultAdapter:
    """Tests for the result adapter module."""

    def test_adapter_transforms_digest(self):
        """Should transform a Rust digest into legacy API response shape."""
        from sentinel_agent.result_adapter import RustResultAdapter

        digest = {
            "issues": [
                {
                    "issue_type": "NegativeInventory",
                    "store_id": "default",
                    "confidence": 0.9,
                    "dollar_impact": 500.0,
                    "skus": [
                        {
                            "sku_id": "SKU-001",
                            "qty_on_hand": -5,
                            "unit_cost": 10.0,
                            "retail_price": 19.99,
                            "margin_pct": 0.5,
                            "sales_last_30d": 3,
                        }
                    ],
                },
            ],
        }

        adapter = RustResultAdapter()
        result = adapter.transform(
            digest=digest,
            total_rows=100,
            analysis_time=0.5,
        )

        assert "leaks" in result
        assert "summary" in result
        assert result["summary"]["total_rows_analyzed"] == 100
        assert result["summary"]["analysis_time_seconds"] == 0.5
        assert "negative_inventory" in result["leaks"]
        neg_inv = result["leaks"]["negative_inventory"]
        assert neg_inv["count"] == 1
        assert neg_inv["severity"] == "critical"
        assert len(neg_inv["item_details"]) == 1
        assert neg_inv["item_details"][0]["sku"] == "SKU-001"

    def test_adapter_all_11_primitives(self):
        """Should include all 11 primitives in output even if empty."""
        from sentinel_agent.result_adapter import ALL_PRIMITIVES, RustResultAdapter

        adapter = RustResultAdapter()
        result = adapter.transform(digest={"issues": []}, total_rows=0, analysis_time=0)

        assert len(result["leaks"]) == 11
        for prim in ALL_PRIMITIVES:
            assert prim in result["leaks"]
            assert result["leaks"][prim]["count"] == 0

    def test_adapter_impact_estimation(self):
        """Should produce dollar impact estimates."""
        from sentinel_agent.result_adapter import RustResultAdapter

        digest = {
            "issues": [
                {
                    "issue_type": "MarginErosion",
                    "store_id": "default",
                    "confidence": 0.8,
                    "dollar_impact": 1200.0,
                    "skus": [],
                },
            ],
        }

        adapter = RustResultAdapter()
        result = adapter.transform(digest=digest, total_rows=1000, analysis_time=1.0)

        impact = result["summary"]["estimated_impact"]
        assert impact["currency"] == "USD"
        assert impact["low_estimate"] > 0
        assert impact["high_estimate"] > impact["low_estimate"]
        assert "margin_erosion" in impact["breakdown"]
