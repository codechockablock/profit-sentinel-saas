"""
Pytest fixtures for API tests.

Provides mocked dependencies for all external services (S3, Supabase, Grok).
"""

import io
import json
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_pos_csv_data() -> bytes:
    """Sample POS CSV data for testing."""
    csv_content = """sku,description,vendor,category,quantity,cost,revenue,sold
SKU001,Widget A,Vendor X,Electronics,15,10.50,25.00,45
SKU002,Gadget B,Vendor Y,Electronics,0,8.00,20.00,150
SKU003,Part C,Vendor X,Hardware,-5,5.00,12.00,2
SKU004,Item D,Vendor Z,Furniture,50,100.00,95.00,10
SKU005,Tool E,Vendor X,Hardware,8,15.00,40.00,0
"""
    return csv_content.encode('utf-8')


@pytest.fixture
def sample_pos_dataframe(sample_pos_csv_data: bytes) -> pd.DataFrame:
    """Sample POS DataFrame for testing."""
    return pd.read_csv(io.BytesIO(sample_pos_csv_data))


@pytest.fixture
def sample_pos_records(sample_pos_dataframe: pd.DataFrame) -> list:
    """Sample POS records as list of dicts."""
    return sample_pos_dataframe.to_dict(orient='records')


@pytest.fixture
def sample_column_mapping() -> dict[str, str]:
    """Sample column mapping for analysis."""
    return {
        "sku": "sku",
        "description": "description",
        "vendor": "vendor",
        "category": "category",
        "quantity": "quantity",
        "cost": "cost",
        "revenue": "revenue",
        "sold": "sold"
    }


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_s3_client() -> MagicMock:
    """Mock boto3 S3 client."""
    mock = MagicMock()

    # Mock presigned URL generation
    mock.generate_presigned_url.return_value = "https://s3.amazonaws.com/bucket/test-key?presigned=true"

    return mock


@pytest.fixture
def mock_s3_client_with_data(mock_s3_client: MagicMock, sample_pos_csv_data: bytes) -> MagicMock:
    """Mock S3 client with sample CSV data."""
    # Mock get_object to return sample CSV
    mock_body = MagicMock()
    mock_body.read.return_value = sample_pos_csv_data
    mock_s3_client.get_object.return_value = {'Body': mock_body}

    # Mock head_object for get_object_size (returns small size to pass size check)
    mock_s3_client.head_object.return_value = {'ContentLength': len(sample_pos_csv_data)}

    return mock_s3_client


@pytest.fixture
def mock_supabase_client() -> MagicMock:
    """Mock Supabase client."""
    mock = MagicMock()

    # Mock auth
    mock_user = MagicMock()
    mock_user.user.id = "test-user-id"
    mock.auth.get_user.return_value = mock_user

    return mock


@pytest.fixture
def mock_grok_client() -> MagicMock:
    """Mock Grok/OpenAI client."""
    mock = MagicMock()

    # Mock chat completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "mapping": {
            "sku": "sku",
            "description": None,
            "quantity": "quantity",
            "cost": "cost",
            "revenue": "revenue"
        },
        "confidence": {
            "sku": 0.95,
            "description": 0.3,
            "quantity": 0.9,
            "cost": 0.85,
            "revenue": 0.88
        },
        "notes": "Mapped columns based on column names and sample data"
    })
    mock.chat.completions.create.return_value = mock_response

    return mock


@pytest.fixture
def mock_sentinel_engine():
    """Mock sentinel_engine module for VSA analysis."""
    with patch.dict('sys.modules', {
        'sentinel_engine': MagicMock(),
    }):
        mock_module = MagicMock()
        mock_module.bundle_pos_facts.return_value = MagicMock()  # Mock bundle
        mock_module.query_bundle.return_value = (
            ["SKU001", "SKU003", "SKU005"],  # Items
            [0.95, 0.87, 0.72]  # Scores
        )
        yield mock_module


# =============================================================================
# APPLICATION FIXTURES
# =============================================================================

@pytest.fixture
def app():
    """Create FastAPI application instance."""
    from apps.api.src.main import create_app
    return create_app()


@pytest.fixture
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def client_with_mocks(
    app,
    mock_s3_client_with_data: MagicMock,
    mock_supabase_client: MagicMock,
    mock_grok_client: MagicMock
) -> Generator[TestClient, None, None]:
    """Test client with all external services mocked."""
    with patch('apps.api.src.dependencies.get_s3_client', return_value=mock_s3_client_with_data):
        with patch('apps.api.src.dependencies.get_supabase_client', return_value=mock_supabase_client):
            with patch('apps.api.src.dependencies.get_grok_client', return_value=mock_grok_client):
                yield TestClient(app)


@pytest.fixture
def authenticated_client(
    client_with_mocks: TestClient,
    mock_supabase_client: MagicMock
) -> TestClient:
    """Test client with authentication header."""
    client_with_mocks.headers = {"Authorization": "Bearer test-token"}
    return client_with_mocks


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key")
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
