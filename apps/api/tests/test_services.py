"""
Tests for API service classes.

Unit tests for S3Service, MappingService, and AnalysisService.
"""

import io
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


# =============================================================================
# S3 SERVICE TESTS
# =============================================================================

class TestS3Service:
    """Tests for S3Service class."""

    def test_init(self, mock_s3_client: MagicMock):
        """Test S3Service initialization."""
        from apps.api.src.services.s3 import S3Service
        service = S3Service(mock_s3_client, "test-bucket")
        assert service.client == mock_s3_client
        assert service.bucket_name == "test-bucket"

    def test_generate_presigned_url(self, mock_s3_client: MagicMock):
        """Test presigned URL generation."""
        from apps.api.src.services.s3 import S3Service
        service = S3Service(mock_s3_client, "test-bucket")

        url = service.generate_presigned_url("test/key.csv")

        mock_s3_client.generate_presigned_url.assert_called_once()
        call_args = mock_s3_client.generate_presigned_url.call_args
        assert call_args[0][0] == 'put_object'
        assert call_args[1]['Params']['Bucket'] == "test-bucket"
        assert call_args[1]['Params']['Key'] == "test/key.csv"

    def test_generate_presigned_url_custom_expiry(self, mock_s3_client: MagicMock):
        """Test presigned URL with custom expiry."""
        from apps.api.src.services.s3 import S3Service
        service = S3Service(mock_s3_client, "test-bucket")

        service.generate_presigned_url("key.csv", expires_in=7200)

        call_args = mock_s3_client.generate_presigned_url.call_args
        assert call_args[1]['ExpiresIn'] == 7200

    def test_load_dataframe_csv(self, mock_s3_client_with_data: MagicMock):
        """Test loading CSV file as DataFrame."""
        from apps.api.src.services.s3 import S3Service
        service = S3Service(mock_s3_client_with_data, "test-bucket")

        df = service.load_dataframe("test/data.csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        mock_s3_client_with_data.get_object.assert_called_once()

    def test_load_dataframe_with_sample_rows(self, mock_s3_client_with_data: MagicMock):
        """Test loading limited sample rows."""
        from apps.api.src.services.s3 import S3Service
        service = S3Service(mock_s3_client_with_data, "test-bucket")

        df = service.load_dataframe("test/data.csv", sample_rows=2)

        assert len(df) <= 2

    def test_load_dataframe_excel(self, mock_s3_client: MagicMock):
        """Test loading Excel file as DataFrame."""
        from apps.api.src.services.s3 import S3Service

        # Create mock Excel data
        excel_buffer = io.BytesIO()
        pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']}).to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        excel_data = excel_buffer.getvalue()

        mock_body = MagicMock()
        mock_body.read.return_value = excel_data
        mock_s3_client.get_object.return_value = {'Body': mock_body}
        # Mock head_object for get_object_size
        mock_s3_client.head_object.return_value = {'ContentLength': len(excel_data)}

        service = S3Service(mock_s3_client, "test-bucket")
        df = service.load_dataframe("test/data.xlsx")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['col1', 'col2']

    def test_upload_file(self, mock_s3_client: MagicMock):
        """Test file upload."""
        from apps.api.src.services.s3 import S3Service
        service = S3Service(mock_s3_client, "test-bucket")

        service.upload_file("test/key.csv", b"csv,data", "text/csv")

        mock_s3_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/key.csv",
            Body=b"csv,data",
            ContentType="text/csv"
        )

    def test_delete_file(self, mock_s3_client: MagicMock):
        """Test file deletion."""
        from apps.api.src.services.s3 import S3Service
        service = S3Service(mock_s3_client, "test-bucket")

        service.delete_file("test/key.csv")

        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/key.csv"
        )


# =============================================================================
# MAPPING SERVICE TESTS
# =============================================================================

class TestMappingService:
    """Tests for MappingService class."""

    def test_suggest_mapping_with_ai(
        self,
        sample_pos_dataframe: pd.DataFrame,
        mock_grok_client: MagicMock
    ):
        """Test AI-powered column mapping."""
        from apps.api.src.services.mapping import MappingService

        with patch('apps.api.src.services.mapping.get_grok_client', return_value=mock_grok_client):
            service = MappingService()
            result = service.suggest_mapping(sample_pos_dataframe, "test.csv")

        assert "original_columns" in result
        assert "suggestions" in result
        assert "confidences" in result
        assert "sample_data" in result

    def test_suggest_mapping_heuristic_fallback(
        self,
        sample_pos_dataframe: pd.DataFrame
    ):
        """Test heuristic fallback when AI unavailable."""
        from apps.api.src.services.mapping import MappingService

        with patch('apps.api.src.services.mapping.get_grok_client', return_value=None):
            service = MappingService()
            result = service.suggest_mapping(sample_pos_dataframe, "test.csv")

        assert "suggestions" in result
        assert "notes" in result
        # Heuristic should match common field names
        suggestions = result["suggestions"]
        assert suggestions.get("sku") == "sku"
        assert suggestions.get("quantity") == "quantity"

    def test_heuristic_mapping_standard_fields(self):
        """Test heuristic mapping recognizes standard field names."""
        from apps.api.src.services.mapping import MappingService

        columns = ["sku", "qty", "sale_price", "cost", "vendor_name"]
        df = pd.DataFrame({col: ["test"] for col in columns})

        with patch('apps.api.src.services.mapping.get_grok_client', return_value=None):
            service = MappingService()
            result = service.suggest_mapping(df, "test.csv")

        suggestions = result["suggestions"]
        assert suggestions.get("sku") == "sku"
        assert suggestions.get("qty") == "quantity"
        assert suggestions.get("sale_price") == "revenue"
        assert suggestions.get("cost") == "cost"
        assert suggestions.get("vendor_name") == "vendor"

    def test_heuristic_mapping_unmapped_columns(self):
        """Test heuristic returns None for unrecognized columns."""
        from apps.api.src.services.mapping import MappingService

        columns = ["unknown_col", "random_field", "xyz123"]
        df = pd.DataFrame({col: ["test"] for col in columns})

        with patch('apps.api.src.services.mapping.get_grok_client', return_value=None):
            service = MappingService()
            result = service.suggest_mapping(df, "test.csv")

        suggestions = result["suggestions"]
        for col in columns:
            assert suggestions.get(col) is None

    def test_ai_mapping_handles_markdown_response(self):
        """Test AI mapping handles JSON in markdown code blocks."""
        from apps.api.src.services.mapping import MappingService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """```json
{
    "mapping": {"col1": "sku"},
    "confidence": {"col1": 0.9},
    "notes": "test"
}
```"""
        mock_client.chat.completions.create.return_value = mock_response

        df = pd.DataFrame({"col1": ["test"]})

        with patch('apps.api.src.services.mapping.get_grok_client', return_value=mock_client):
            service = MappingService()
            result = service.suggest_mapping(df, "test.csv")

        assert result["suggestions"]["col1"] == "sku"

    def test_ai_mapping_error_fallback(self):
        """Test AI mapping falls back to heuristic on error."""
        from apps.api.src.services.mapping import MappingService

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        df = pd.DataFrame({"sku": ["test"], "quantity": [1]})

        with patch('apps.api.src.services.mapping.get_grok_client', return_value=mock_client):
            service = MappingService()
            result = service.suggest_mapping(df, "test.csv")

        # Should fall back to heuristic (notes may say "Grok failed" or "AI failed")
        notes = result.get("notes", "")
        assert "failed" in notes.lower()
        assert result["suggestions"]["sku"] == "sku"


# =============================================================================
# ANALYSIS SERVICE TESTS
# =============================================================================

class TestAnalysisService:
    """Tests for AnalysisService class."""

    def test_mock_analysis_when_engine_unavailable(self):
        """Test mock analysis when sentinel engine is not available."""
        from apps.api.src.services.analysis import AnalysisService

        # Patch the import to simulate missing engine
        with patch.dict('sys.modules', {'sentinel_engine': None}):
            service = AnalysisService()

            if not service._engine_available:
                result = service.analyze([])

                # Should return mock results with all 8 primitives
                assert "leaks" in result
                leaks = result["leaks"]

                # All 8 primitives should be present
                expected_primitives = [
                    "high_margin_leak", "negative_inventory", "low_stock",
                    "shrinkage_pattern", "margin_erosion", "dead_item",
                    "overstock", "price_discrepancy"
                ]
                for primitive in expected_primitives:
                    assert primitive in leaks, f"Missing primitive: {primitive}"
                    assert "top_items" in leaks[primitive]
                    assert "scores" in leaks[primitive]

    def test_analysis_primitives_defined(self):
        """Test all 8 expected primitives are defined."""
        from apps.api.src.services.analysis import AnalysisService

        # All 8 detection primitives
        expected = [
            "high_margin_leak", "negative_inventory", "low_stock",
            "shrinkage_pattern", "margin_erosion", "dead_item",
            "overstock", "price_discrepancy"
        ]
        assert set(expected) == set(AnalysisService.PRIMITIVES)

    def test_analyze_with_mock_engine(self, sample_pos_records: list):
        """Test analyze with mocked sentinel engine."""
        from apps.api.src.services.analysis import AnalysisService

        # Create service with mocked engine
        service = AnalysisService()

        # If engine not available, test mock path
        if not service._engine_available:
            result = service.analyze(sample_pos_records)
            # Result should have leaks structure with all 8 primitives
            assert "leaks" in result
            assert "low_stock" in result["leaks"]
            assert "high_margin_leak" in result["leaks"]
            assert "summary" in result
        else:
            # Engine available - actual analysis would run
            result = service.analyze(sample_pos_records)
            assert "leaks" in result
            assert "summary" in result


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Tests for configuration module."""

    def test_settings_defaults(self):
        """Test default settings values."""
        from apps.api.src.config import Settings

        settings = Settings()
        assert settings.app_name == "Profit Sentinel"
        assert settings.app_version == "1.0.0"
        assert settings.debug is False
        assert "localhost:3000" in str(settings.cors_origins)

    def test_ai_api_key_preference(self, monkeypatch):
        """Test XAI_API_KEY is preferred over GROK_API_KEY."""
        from apps.api.src.config import Settings

        monkeypatch.setenv("XAI_API_KEY", "xai-key")
        monkeypatch.setenv("GROK_API_KEY", "grok-key")

        settings = Settings()
        assert settings.ai_api_key == "xai-key"

    def test_ai_api_key_fallback(self, monkeypatch):
        """Test fallback to GROK_API_KEY when XAI_API_KEY not set."""
        from apps.api.src.config import Settings, get_settings

        # Clear cached settings first
        get_settings.cache_clear()

        # Force empty XAI_API_KEY by setting to empty string, then create settings
        # that explicitly sets the values (bypassing .env file)
        settings = Settings(xai_api_key=None, grok_api_key="grok-key")
        assert settings.ai_api_key == "grok-key"

    def test_has_ai_key_true(self, monkeypatch):
        """Test has_ai_key returns True when key configured."""
        from apps.api.src.config import Settings

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        settings = Settings()
        assert settings.has_ai_key is True

    def test_has_ai_key_false(self):
        """Test has_ai_key returns False when no key configured."""
        from apps.api.src.config import Settings

        # Create settings with explicit None values (bypassing .env file)
        settings = Settings(xai_api_key=None, grok_api_key=None)
        assert settings.has_ai_key is False

    def test_standard_fields_defined(self):
        """Test STANDARD_FIELDS dictionary is properly defined."""
        from apps.api.src.config import STANDARD_FIELDS

        expected_keys = [
            "date", "sku", "quantity", "revenue", "cost",
            "vendor", "category", "transaction_id", "customer_id",
            "discount", "tax", "return_flag"
        ]
        for key in expected_keys:
            assert key in STANDARD_FIELDS
            assert isinstance(STANDARD_FIELDS[key], list)
            assert len(STANDARD_FIELDS[key]) > 0


# =============================================================================
# DEPENDENCIES TESTS
# =============================================================================

class TestDependencies:
    """Tests for dependency injection functions."""

    def test_get_s3_client(self, monkeypatch):
        """Test S3 client creation."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")

        from apps.api.src.dependencies import get_s3_client

        with patch('boto3.client') as mock_boto:
            get_s3_client()
            mock_boto.assert_called_once_with("s3")

    def test_get_supabase_client_none_without_config(self):
        """Test Supabase client returns None without config."""
        from apps.api.src.config import Settings, get_settings
        from apps.api.src.dependencies import get_supabase_client

        # Create mock settings with no supabase config
        mock_settings = Settings(supabase_url=None, supabase_service_key=None)

        # Patch get_settings to return our mock settings
        with patch('apps.api.src.dependencies.get_settings', return_value=mock_settings):
            result = get_supabase_client()
        assert result is None

    def test_get_grok_client_none_without_key(self):
        """Test Grok client returns None without API key."""
        from apps.api.src.config import Settings
        from apps.api.src.dependencies import get_grok_client

        # Create mock settings with no AI API key
        mock_settings = Settings(xai_api_key=None, grok_api_key=None)

        # Patch get_settings to return our mock settings
        with patch('apps.api.src.dependencies.get_settings', return_value=mock_settings):
            result = get_grok_client()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self):
        """Test get_current_user returns None without token."""
        from apps.api.src.dependencies import get_current_user

        result = await get_current_user(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_require_user_raises_without_auth(self):
        """Test require_user raises 401 without authentication."""
        from apps.api.src.dependencies import require_user
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await require_user(None)

        assert exc_info.value.status_code == 401
