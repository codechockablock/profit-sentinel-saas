"""Regression tests for recent security and config propagation fixes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sentinel_agent.sidecar import create_app
from sentinel_agent.sidecar_config import SidecarSettings


@pytest.fixture
def app_client():
    """Create a sidecar TestClient with deterministic mocked engine."""
    settings = SidecarSettings(
        sidecar_dev_mode=True,
        s3_bucket_name="test-bucket",
        aws_region="us-east-1",
        sentinel_default_store="default",
        sentinel_top_k=20,
    )

    with patch("sentinel_agent.sidecar.SentinelEngine") as mock_engine_ctor:
        mock_engine = MagicMock()
        mock_engine.binary = "/mock/sentinel-server"
        mock_engine_ctor.return_value = mock_engine

        app = create_app(settings=settings)
        state = app.extra["sentinel_state"]
        with TestClient(app) as client:
            yield client, app, state


def test_config_put_applies_to_new_world_model_pipeline(app_client):
    client, _app, state = app_client

    response = client.put(
        "/api/v1/config",
        json={"overrides": {"min_healthy_velocity": 1.75}},
    )
    assert response.status_code == 200
    assert state.user_configs["dev-user"]["min_healthy_velocity"] == 1.75

    if state.world_model is None:
        pytest.skip("Engine 2 unavailable in this environment")

    pipeline = state.get_user_world_model("dev-user")
    assert pipeline is not None
    assert pipeline.dead_stock_config.min_healthy_velocity == 1.75
    assert pipeline.predictive.dead_stock_velocity_threshold == 1.75


def test_config_put_updates_existing_world_model_pipeline(app_client):
    client, _app, state = app_client

    if state.world_model is None:
        pytest.skip("Engine 2 unavailable in this environment")

    pipeline = state.get_user_world_model("dev-user")
    assert pipeline is not None

    response = client.put(
        "/api/v1/config",
        json={"overrides": {"min_healthy_velocity": 2.25}},
    )
    assert response.status_code == 200
    assert pipeline.dead_stock_config.min_healthy_velocity == 2.25
    assert pipeline.predictive.dead_stock_velocity_threshold == 2.25


def test_health_engine2_redacts_exception_details(app_client):
    client, _app, state = app_client

    class ExplodingWorldModel:
        @property
        def entity_history(self):
            raise RuntimeError("sensitive-engine2-error")

    state.world_model = ExplodingWorldModel()

    response = client.get("/health/engine2")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["error"] == "internal_error"
    assert "sensitive-engine2-error" not in response.text


def test_health_supabase_redacts_exception_details(app_client):
    client, _app, state = app_client

    class ExplodingSupabase:
        def table(self, *_args, **_kwargs):
            raise RuntimeError("secret-token-should-not-leak")

    state.supabase_client = ExplodingSupabase()

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["dependencies"]["supabase"]["ok"] is False
    assert data["dependencies"]["supabase"]["detail"] == "unavailable"
    assert "secret-token-should-not-leak" not in response.text


def test_cors_policy_restricts_methods_and_headers(app_client):
    _client, app, _state = app_client

    cors = next(m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert cors.kwargs["allow_methods"] == [
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    ]
    assert cors.kwargs["allow_headers"] == [
        "Authorization",
        "Content-Type",
        "X-Request-ID",
    ]
