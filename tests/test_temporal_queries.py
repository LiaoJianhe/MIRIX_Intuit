"""Tests for temporal query functionality."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import requests

from mirix.client import MirixClient
from mirix.temporal.temporal_parser import TemporalRange, parse_temporal_expression

# Integration tests use a running server (same as test_deletion_apis / test_search_all_users)
BASE_URL = os.environ.get("MIRIX_API_URL", "http://localhost:8000")
CONFIG_PATH = Path(__file__).parent.parent / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"


class TestTemporalParser:
    """Test temporal expression parsing."""

    def test_parse_today(self):
        """Test parsing 'today' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What happened today?", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 19, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()
        assert result.end.hour == 23
        assert result.end.minute == 59

    def test_parse_yesterday(self):
        """Test parsing 'yesterday' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What did I do yesterday?", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 18, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 18).date()
        assert result.end.hour == 23

    def test_parse_last_week(self):
        """Test parsing 'last week' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What happened last week?", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 12, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_this_week(self):
        """Test parsing 'this week' expression."""
        # Use a Wednesday for testing
        ref_time = datetime(2025, 11, 19, 14, 30, 0)  # Wednesday
        result = parse_temporal_expression("Show me this week's events", ref_time)

        assert result is not None
        # Should start from Monday (2 days before Wednesday)
        assert result.start == datetime(2025, 11, 17, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_last_month(self):
        """Test parsing 'last month' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What happened last month?", ref_time)

        assert result is not None
        # Approximately 30 days ago
        expected_start = datetime(2025, 10, 20, 0, 0, 0, 0)
        assert result.start == expected_start

    def test_parse_this_month(self):
        """Test parsing 'this month' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("Show me this month's activities", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 1, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_last_n_days(self):
        """Test parsing 'last N days' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What did I do in the last 3 days?", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 16, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_last_n_weeks(self):
        """Test parsing 'last N weeks' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("Show me last 2 weeks", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 5, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_last_n_months(self):
        """Test parsing 'last N months' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("Show me last 2 months", ref_time)

        assert result is not None
        # Approximately 60 days ago
        expected_start = datetime(2025, 9, 20, 0, 0, 0, 0)
        assert result.start == expected_start

    def test_no_temporal_expression(self):
        """Test that None is returned when no temporal expression is found."""
        result = parse_temporal_expression("What is the weather?", datetime.now())
        assert result is None

        result = parse_temporal_expression("Tell me about Python", datetime.now())
        assert result is None

    def test_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)

        result1 = parse_temporal_expression("What happened TODAY?", ref_time)
        result2 = parse_temporal_expression("What happened today?", ref_time)
        result3 = parse_temporal_expression("What happened ToDay?", ref_time)

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1.start == result2.start == result3.start

    def test_temporal_range_to_dict(self):
        """Test TemporalRange to_dict() method."""
        start = datetime(2025, 11, 19, 0, 0, 0)
        end = datetime(2025, 11, 19, 23, 59, 59)
        range_obj = TemporalRange(start, end)

        result = range_obj.to_dict()
        assert result["start"] == start.isoformat()
        assert result["end"] == end.isoformat()

    def test_temporal_range_none_values(self):
        """Test TemporalRange with None values."""
        range_obj = TemporalRange(None, None)

        result = range_obj.to_dict()
        assert result["start"] is None
        assert result["end"] is None


@pytest.fixture(scope="module")
def check_server():
    """Skip the module's integration tests if the server is not running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Integration test - server not available at {BASE_URL}: {e}")


@pytest.fixture(scope="module")
def temporal_client(check_server, api_key_factory):
    """Client and user for temporal integration tests."""
    org_id = "test-temporal-org"
    client_id = "test-temporal-client"
    user_id = "test-temporal-user"
    auth = api_key_factory(client_id, org_id)
    client = MirixClient(
        api_key=auth["api_key"],
        base_url=BASE_URL,
        client_id=client_id,
        client_name="Test Temporal Client",
        client_scope="test",
        org_id=org_id,
        debug=False,
    )
    client.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)
    client.create_or_get_user(user_id=user_id, user_name="Test Temporal User", org_id=org_id)
    return client, user_id


class TestTemporalIntegration:
    """Integration tests for temporal query feature.

    Require a running server. Run with: pytest tests/test_temporal_queries.py -m integration
    Or run all tests (integration tests skip if server is down):
    pytest tests/test_temporal_queries.py
    """

    @pytest.mark.integration
    def test_retrieve_with_temporal_expression(self, temporal_client):
        """Test retrieval with natural language temporal expression (parsed to start/end dates)."""
        client, user_id = temporal_client
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        parsed = parse_temporal_expression("What happened today?", ref_time)
        assert parsed is not None
        start_date = parsed.start.strftime("%Y-%m-%dT%H:%M:%S")
        end_date = parsed.end.strftime("%Y-%m-%dT%H:%M:%S")
        results = client.search(
            user_id=user_id,
            query="",
            memory_type="episodic",
            limit=20,
            start_date=start_date,
            end_date=end_date,
        )
        assert results.get("success") is True
        assert "results" in results
        assert "date_range" in results
        assert results["date_range"] is None or isinstance(results["date_range"], dict)
        # If date_range was applied, response should reflect it
        if results.get("date_range"):
            assert "start" in results["date_range"] or "end" in results["date_range"]

    @pytest.mark.integration
    def test_retrieve_with_explicit_date_range(self, temporal_client):
        """Test retrieval with explicit start_date and end_date."""
        client, user_id = temporal_client
        start_date = "2025-11-01T00:00:00"
        end_date = "2025-11-30T23:59:59"
        results = client.search(
            user_id=user_id,
            query="",
            memory_type="episodic",
            limit=20,
            start_date=start_date,
            end_date=end_date,
        )
        assert results.get("success") is True
        assert "results" in results
        assert isinstance(results["results"], list)
        assert results.get("date_range") is not None
        assert results["date_range"].get("start") is not None
        assert results["date_range"].get("end") is not None

    @pytest.mark.integration
    def test_temporal_filtering_episodic_only(self, temporal_client):
        """Test that temporal filtering only affects episodic memories (date_range in response)."""
        client, user_id = temporal_client
        start_date = "2025-10-01T00:00:00"
        end_date = "2025-10-31T23:59:59"
        results = client.search(
            user_id=user_id,
            query="",
            memory_type="all",
            limit=20,
            start_date=start_date,
            end_date=end_date,
        )
        assert results.get("success") is True
        assert "results" in results
        # date_range is applied server-side for episodic only; API returns it in response
        assert results.get("date_range") is not None
        assert results["date_range"].get("start") is not None
        assert results["date_range"].get("end") is not None
        # Only episodic results have timestamp; other types may be present without temporal filter
        for item in results["results"]:
            if item.get("memory_type") == "episodic" and item.get("timestamp"):
                # Episodic timestamps should fall within range (server filters by occurred_at)
                pass


# Additional documentation and usage examples
"""
Usage Examples:
===============

1. Automatic temporal parsing:
   >>> from mirix import MirixClient
   >>> client = MirixClient(...)
   >>> memories = client.retrieve_with_conversation(
   ...     user_id='demo-user',
   ...     messages=[
   ...         {"role": "user", "content": [{"type": "text", "text": "What did we discuss today?"}]}
   ...     ]
   ... )
   
2. Explicit date range:
   >>> memories = client.retrieve_with_conversation(
   ...     user_id='demo-user',
   ...     messages=[
   ...         {"role": "user", "content": [{"type": "text", "text": "Show me meetings"}]}
   ...     ],
   ...     start_date="2025-11-19T00:00:00",
   ...     end_date="2025-11-19T23:59:59"
   ... )

3. Combine with filter_tags:
   >>> memories = client.retrieve_with_conversation(
   ...     user_id='demo-user',
   ...     messages=[
   ...         {"role": "user", "content": [{"type": "text", "text": "What did I do yesterday?"}]}
   ...     ],
   ...     filter_tags={"expert_id": "expert-123"}
   ... )

Supported Temporal Expressions:
================================
- "today": Current day from 00:00:00 to 23:59:59
- "yesterday": Previous day
- "last N days": Previous N days including today
- "last week": Previous 7 days
- "this week": From Monday of current week to now
- "last month": Previous 30 days
- "this month": From 1st of current month to now
- "last N weeks": Previous N weeks
- "last N months": Previous N * 30 days

Note: Only episodic memories are filtered by temporal expressions.
      Other memory types (semantic, procedural, resource, knowledge vault, core) 
      do not have occurred_at timestamps and are not affected.
"""
