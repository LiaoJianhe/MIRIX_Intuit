"""
Integration tests for filter_tags operator queries via the REST API.

Requires a running server: python scripts/start_server.py
Tests the full round-trip: API -> manager -> PG -> response.

Run:
    pytest tests/test_filter_tags_integration.py -v -m integration -s
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path

import pytest
import requests

from mirix.client import MirixClient

pytestmark = [
    pytest.mark.integration,
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("MIRIX_API_URL", "http://localhost:8000")
CONFIG_PATH = Path(__file__).parent.parent / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"


# =================================================================
# FIXTURES
# =================================================================

@pytest.fixture(scope="module")
def client():
    """Create a MirixClient with a unique scope for test isolation."""
    api_key = os.environ.get("MIRIX_API_KEY")
    if not api_key:
        pytest.skip("MIRIX_API_KEY not set")

    c = MirixClient(
        api_key=api_key,
        base_url=BASE_URL,
        debug=True,
    )
    return c


@pytest.fixture(scope="module")
def user_id():
    return f"test-filter-ops-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def setup_memories(client, user_id):
    """Add memories with various filter_tags shapes for testing operators."""
    memories_added = []

    def add_memory(text, filter_tags):
        result = client.add(
            user_id=user_id,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": text}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Acknowledged"}]},
            ],
            filter_tags=filter_tags,
            chaining=True,
        )
        memories_added.append(result)
        return result

    add_memory(
        "Account ABC123 had a transaction for project Alpha",
        {"account_ids": ["ABC123", "DEF456"], "priority": "high"},
    )
    add_memory(
        "Account GHI789 had a transaction for project Beta",
        {"account_ids": ["GHI789"], "priority": "low"},
    )
    add_memory(
        "General note about project Gamma with no account_ids",
        {"priority": "medium"},
    )

    time.sleep(5)
    yield memories_added


# =================================================================
# Tests
# =================================================================

class TestContainsViaApi:
    def test_search_with_contains_operator(self, client, user_id, setup_memories):
        """$contains finds memories where account_ids array includes the value."""
        result = client.search(
            user_id=user_id,
            query="transaction",
            filter_tags={"account_ids": {"$contains": "ABC123"}},
            limit=10,
        )
        assert result["success"]
        results = result.get("results", [])
        texts = [r.get("summary", "") + r.get("details", "") + r.get("context", "") for r in results]
        matching = [t for t in texts if "ABC123" in t]
        assert len(matching) >= 1, f"Expected at least 1 result with ABC123, got {len(matching)}"

    def test_contains_no_match(self, client, user_id, setup_memories):
        """$contains returns no results when value is not in any array."""
        result = client.search(
            user_id=user_id,
            query="transaction",
            filter_tags={"account_ids": {"$contains": "NONEXISTENT"}},
            limit=10,
        )
        assert result["success"]
        assert result.get("count", 0) == 0 or len(result.get("results", [])) == 0


class TestExistsViaApi:
    def test_exists_true(self, client, user_id, setup_memories):
        """$exists: true returns only memories that have the key."""
        result = client.search(
            user_id=user_id,
            query="project",
            filter_tags={"account_ids": {"$exists": True}},
            limit=10,
        )
        assert result["success"]
        for r in result.get("results", []):
            ft = r.get("filter_tags", {})
            if ft:
                assert "account_ids" in ft, f"Expected account_ids in filter_tags, got {ft}"

    def test_exists_false(self, client, user_id, setup_memories):
        """$exists: false returns memories that do NOT have the key."""
        result = client.search(
            user_id=user_id,
            query="project Gamma",
            filter_tags={"account_ids": {"$exists": False}},
            limit=10,
        )
        assert result["success"]
        for r in result.get("results", []):
            ft = r.get("filter_tags", {})
            if ft:
                assert "account_ids" not in ft, f"Expected no account_ids, got {ft}"


class TestInViaApi:
    def test_in_operator(self, client, user_id, setup_memories):
        """$in matches memories where the stored scalar is one of the values."""
        result = client.search(
            user_id=user_id,
            query="project",
            filter_tags={"priority": {"$in": ["high", "medium"]}},
            limit=10,
        )
        assert result["success"]
        for r in result.get("results", []):
            ft = r.get("filter_tags", {})
            if ft and "priority" in ft:
                assert ft["priority"] in ["high", "medium"], f"Unexpected priority: {ft['priority']}"
