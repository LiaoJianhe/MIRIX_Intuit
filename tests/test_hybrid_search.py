"""
Tests for mirix.services.hybrid_search_helper.

Usage:
    pytest tests/test_hybrid_search.py -v
"""

from unittest.mock import AsyncMock

import pytest

from mirix.services.hybrid_search_helper import (
    DEFAULT_RECENT_CAP,
    fetch_recent_window,
    get_hybrid_window_seconds,
    set_hybrid_window_seconds,
)


@pytest.fixture(autouse=True)
def hybrid_window_five_seconds():
    set_hybrid_window_seconds(5)
    yield


@pytest.mark.asyncio
class TestFetchRecentWindow:
    """The save-flow helper fetches just-written rows from the Relational
    provider's recent window. The ranked ("relevant") bucket is obtained
    separately by the caller via the manager ``list_*`` call, so this helper
    does not issue a Search call. De-dup against the ranked list happens in the
    caller (``Agent._merge_recent_into_relevant``), not here.
    """

    async def test_returns_relational_rows_as_list(self):
        relational = AsyncMock()
        relational.list = AsyncMock(
            return_value=[{"id": "r1", "updated_at": "2025-06-01T00:00:00+00:00"}]
        )

        recent = await fetch_recent_window(
            "semantic_memory",
            relational,
            user_id="u1",
        )

        assert [r["id"] for r in recent] == ["r1"]
        relational.list.assert_awaited_once()

    async def test_does_not_call_a_search_provider(self):
        """The helper only touches the Relational provider; it takes no Search
        provider argument and issues no Search call."""
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        await fetch_recent_window("semantic_memory", relational, user_id="u1")

        # Only the relational list call was made.
        relational.list.assert_awaited_once()

    async def test_relational_call_uses_recent_window(self):
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        await fetch_recent_window(
            "episodic_memory",
            relational,
            user_id="u1",
        )

        call_kw = relational.list.await_args
        assert call_kw[0][0] == "episodic_memory"
        assert call_kw[1]["user_id"] == "u1"
        tr = call_kw[1]["time_range"]
        assert "updated_at__gte" in tr
        assert "created_at__gte" in tr
        assert call_kw[1]["time_range_or_null_updated"] is True

    async def test_recent_cap_applied_to_relational_list(self):
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        await fetch_recent_window(
            "semantic_memory",
            relational,
            user_id="u1",
            recent_cap=42,
        )

        assert relational.list.await_args[1]["limit"] == 42

    async def test_recent_cap_defaults_to_500(self):
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        await fetch_recent_window("semantic_memory", relational, user_id="u1")

        assert relational.list.await_args[1]["limit"] == DEFAULT_RECENT_CAP == 500

    async def test_relational_raises_propagates(self):
        relational = AsyncMock()
        relational.list = AsyncMock(side_effect=RuntimeError("relational down"))

        with pytest.raises(RuntimeError, match="relational down"):
            await fetch_recent_window("episodic_memory", relational)


class TestWindowAccessors:
    def test_set_and_get_round_trip(self):
        original = get_hybrid_window_seconds()
        try:
            set_hybrid_window_seconds(42)
            assert get_hybrid_window_seconds() == 42
        finally:
            set_hybrid_window_seconds(original)
