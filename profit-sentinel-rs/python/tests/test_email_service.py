"""Tests for the email service module.

Covers HTML template rendering, dollar formatting, and email sending (mocked).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sentinel_agent.email_service import (
    _format_dollar,
    _priority_color,
    _trend_arrow,
    render_digest_html,
    send_digest_email,
    send_email,
)

# ---------------------------------------------------------------------------
# Dollar formatting
# ---------------------------------------------------------------------------


class TestFormatDollar:
    def test_millions(self):
        assert _format_dollar(1_500_000) == "$1.5M"

    def test_thousands(self):
        assert _format_dollar(42_300) == "$42.3K"

    def test_hundreds(self):
        assert _format_dollar(750) == "$750"

    def test_zero(self):
        assert _format_dollar(0) == "$0"

    def test_exact_million(self):
        assert _format_dollar(1_000_000) == "$1.0M"

    def test_exact_thousand(self):
        assert _format_dollar(1_000) == "$1.0K"


# ---------------------------------------------------------------------------
# Priority colors
# ---------------------------------------------------------------------------


class TestPriorityColor:
    def test_critical(self):
        bg, fg = _priority_color(12)
        assert "fef2f2" in bg  # red background
        assert "dc2626" in fg  # red text

    def test_high(self):
        bg, fg = _priority_color(9)
        assert "fff7ed" in bg  # orange background

    def test_medium(self):
        bg, fg = _priority_color(6)
        assert "fefce8" in bg  # yellow background

    def test_low(self):
        bg, fg = _priority_color(3)
        assert "eff6ff" in bg  # blue background

    def test_boundary_10(self):
        bg, _ = _priority_color(10)
        assert "fef2f2" in bg  # 10 is critical

    def test_boundary_8(self):
        bg, _ = _priority_color(8)
        assert "fff7ed" in bg  # 8 is high


# ---------------------------------------------------------------------------
# Trend arrows
# ---------------------------------------------------------------------------


class TestTrendArrow:
    def test_worsening(self):
        arrow = _trend_arrow("Worsening")
        assert "dc2626" in arrow  # red
        assert "9650" in arrow  # up triangle

    def test_improving(self):
        arrow = _trend_arrow("Improving")
        assert "16a34a" in arrow  # green
        assert "9660" in arrow  # down triangle

    def test_stable(self):
        arrow = _trend_arrow("Stable")
        assert "9ca3af" in arrow  # gray
        assert "8212" in arrow  # em dash


# ---------------------------------------------------------------------------
# HTML template rendering
# ---------------------------------------------------------------------------


class TestRenderDigestHtml:
    def test_basic_render(self):
        html = render_digest_html(
            plain_text="Test digest",
            issue_count=3,
            total_impact=15000,
            store_count=2,
            pipeline_ms=45,
        )
        assert "<!DOCTYPE html>" in html
        assert "Profit Sentinel" in html
        assert "Morning Digest" in html
        assert "3" in html  # issue count
        assert "$15.0K" in html  # total impact
        assert "2" in html  # store count
        assert "45ms" in html  # pipeline

    def test_zero_issues_all_clear(self):
        html = render_digest_html(
            plain_text="All clear",
            issue_count=0,
            total_impact=0,
            store_count=0,
            pipeline_ms=10,
        )
        assert "All Clear" in html
        assert "No priority issues" in html

    def test_with_issues_table(self):
        issues = [
            {
                "issue_type": "DeadStock",
                "store_id": "store-7",
                "dollar_impact": 5000,
                "priority_score": 11.0,
                "trend_direction": "Worsening",
                "root_cause": "Theft",
                "skus": [{"sku_id": "SKU-001"}, {"sku_id": "SKU-002"}],
            },
            {
                "issue_type": "MarginErosion",
                "store_id": "store-12",
                "dollar_impact": 2500,
                "priority_score": 7.0,
                "trend_direction": "Stable",
                "root_cause": "",
                "skus": [{"sku_id": "SKU-003"}],
            },
        ]
        html = render_digest_html(
            plain_text="Test",
            issue_count=2,
            total_impact=7500,
            store_count=2,
            pipeline_ms=30,
            issues=issues,
        )
        assert "Dead Stock" in html  # CamelCase converted
        assert "Margin Erosion" in html
        assert "store-7" in html
        assert "2 SKUs" in html
        assert "1 SKU" in html
        assert "$5.0K" in html

    def test_custom_date(self):
        html = render_digest_html(
            plain_text="Test",
            issue_count=1,
            total_impact=100,
            store_count=1,
            pipeline_ms=5,
            generated_at="January 15, 2026",
        )
        assert "January 15, 2026" in html

    def test_dashboard_link(self):
        html = render_digest_html(
            plain_text="Test",
            issue_count=1,
            total_impact=100,
            store_count=1,
            pipeline_ms=5,
        )
        assert "profitsentinel.com/dashboard" in html
        assert "View Full Dashboard" in html

    def test_unsubscribe_link(self):
        html = render_digest_html(
            plain_text="Test",
            issue_count=0,
            total_impact=0,
            store_count=0,
            pipeline_ms=0,
        )
        assert "Unsubscribe" in html

    def test_limits_to_ten_issues(self):
        issues = [
            {
                "issue_type": f"Type{i}",
                "store_id": f"store-{i}",
                "dollar_impact": 100 * i,
                "priority_score": 5,
                "trend_direction": "Stable",
                "root_cause": "",
                "skus": [],
            }
            for i in range(15)
        ]
        html = render_digest_html(
            plain_text="Test",
            issue_count=15,
            total_impact=10000,
            store_count=15,
            pipeline_ms=50,
            issues=issues,
        )
        # Should only render first 10
        assert "store-9" in html
        assert "store-10" not in html


# ---------------------------------------------------------------------------
# Email sending (mocked HTTP)
# ---------------------------------------------------------------------------


class TestSendEmail:
    @pytest.mark.asyncio
    async def test_send_email_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "email-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel_agent.email_service.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await send_email(
                api_key="re_test_key",
                to="test@example.com",
                subject="Test",
                html="<h1>Hello</h1>",
            )

            assert result["id"] == "email-123"
            instance.post.assert_called_once()
            call_kwargs = instance.post.call_args
            assert call_kwargs[1]["headers"]["Authorization"] == "Bearer re_test_key"

    @pytest.mark.asyncio
    async def test_send_email_list_recipients(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "email-456"}
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel_agent.email_service.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await send_email(
                api_key="re_test_key",
                to=["a@test.com", "b@test.com"],
                subject="Multi",
                html="<p>Hi</p>",
            )

            call_args = instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["to"] == ["a@test.com", "b@test.com"]


class TestSendDigestEmail:
    @pytest.mark.asyncio
    async def test_builds_correct_subject_with_issues(self):
        with patch(
            "sentinel_agent.email_service.send_email", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = {"id": "email-789"}

            await send_digest_email(
                api_key="re_test",
                to="user@test.com",
                plain_text="Digest text",
                issue_count=5,
                total_impact=12500,
                store_count=3,
                pipeline_ms=45,
            )

            mock_send.assert_called_once()
            subject = mock_send.call_args[1]["subject"]
            assert "5 issues" in subject
            assert "$12.5K" in subject

    @pytest.mark.asyncio
    async def test_all_clear_subject(self):
        with patch(
            "sentinel_agent.email_service.send_email", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = {"id": "email-000"}

            await send_digest_email(
                api_key="re_test",
                to="user@test.com",
                plain_text="All clear",
                issue_count=0,
                total_impact=0,
                store_count=0,
                pipeline_ms=10,
            )

            subject = mock_send.call_args[1]["subject"]
            assert "All Clear" in subject

    @pytest.mark.asyncio
    async def test_singular_issue_subject(self):
        with patch(
            "sentinel_agent.email_service.send_email", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = {"id": "email-111"}

            await send_digest_email(
                api_key="re_test",
                to="user@test.com",
                plain_text="One issue",
                issue_count=1,
                total_impact=500,
                store_count=1,
                pipeline_ms=5,
            )

            subject = mock_send.call_args[1]["subject"]
            assert "1 issue" in subject
            assert "1 issues" not in subject  # no double plural
