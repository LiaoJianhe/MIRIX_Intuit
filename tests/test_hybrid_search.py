"""
Tests for mirix.services.hybrid_search_helper.

Usage:
    pytest tests/test_hybrid_search.py -v
"""

from unittest.mock import AsyncMock

import pytest

from mirix.database.call_context import set_hybrid_window_seconds
from mirix.services.hybrid_search_helper import (
    HYBRID_COUNT_RECENT_LIMIT,
    _merge_and_deduplicate,
    hybrid_count,
    hybrid_search,
)


@pytest.fixture(autouse=True)
def hybrid_window_five_seconds():
    set_hybrid_window_seconds(5)
    yield


@pytest.mark.asyncio
class TestHybridSearch:
    async def test_merges_search_and_relational_recent_window(self):
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=(
                [{"id": "s1", "updated_at": "2024-01-01T00:00:00+00:00"}],
                None,
            )
        )
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[{"id": "r1", "updated_at": "2025-06-01T00:00:00+00:00"}])

        merged, next_cursor = await hybrid_search(
            "episodic_memory",
            search,
            relational,
            user_id="u1",
            limit=10,
        )

        assert {r["id"] for r in merged} == {"s1", "r1"}
        assert next_cursor is None
        search.search.assert_awaited_once()
        relational.list.assert_awaited_once()
        call_kw = relational.list.await_args
        assert call_kw[0][0] == "episodic_memory"
        assert call_kw[1]["user_id"] == "u1"
        tr = call_kw[1]["time_range"]
        assert "updated_at__gte" in tr
        assert "created_at__gte" in tr
        assert call_kw[1]["time_range_or_null_updated"] is True

    async def test_deduplicates_by_id_relational_wins(self):
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=(
                [{"id": "1", "updated_at": "2025-01-01T00:00:00+00:00", "src": "search"}],
                None,
            )
        )
        relational = AsyncMock()
        relational.list = AsyncMock(
            return_value=[{"id": "1", "updated_at": "2025-01-02T00:00:00+00:00", "src": "relational"}]
        )

        merged, _next_cursor = await hybrid_search(
            "raw_memory", search, relational, limit=10
        )

        assert len(merged) == 1
        assert merged[0]["src"] == "relational"

    async def test_deduplicates_by_id_search_wins_if_newer(self):
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=(
                [{"id": "1", "updated_at": "2025-01-03T00:00:00+00:00", "src": "search"}],
                None,
            )
        )
        relational = AsyncMock()
        relational.list = AsyncMock(
            return_value=[{"id": "1", "updated_at": "2025-01-02T00:00:00+00:00", "src": "relational"}]
        )

        merged, _next_cursor = await hybrid_search(
            "raw_memory", search, relational, limit=10
        )

        assert len(merged) == 1
        assert merged[0]["src"] == "search"

    async def test_respects_limit(self):
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=(
                [
                    {"id": "a", "updated_at": "2025-01-03T00:00:00+00:00"},
                    {"id": "b", "updated_at": "2025-01-02T00:00:00+00:00"},
                    {"id": "c", "updated_at": "2025-01-01T00:00:00+00:00"},
                ],
                None,
            )
        )
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        merged, _next_cursor = await hybrid_search(
            "semantic_memory", search, relational, limit=2
        )
        assert len(merged) == 2

    async def test_search_raises_propagates(self):
        search = AsyncMock()
        search.search = AsyncMock(side_effect=RuntimeError("search down"))
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        with pytest.raises(RuntimeError, match="search down"):
            await hybrid_search("episodic_memory", search, relational)

        relational.list.assert_not_awaited()

    async def test_relational_raises_propagates(self):
        search = AsyncMock()
        search.search = AsyncMock(return_value=([], None))
        relational = AsyncMock()
        relational.list = AsyncMock(side_effect=RuntimeError("relational down"))

        with pytest.raises(RuntimeError, match="relational down"):
            await hybrid_search("episodic_memory", search, relational)

    async def test_next_cursor_propagates_from_search_provider(self):
        """The cursor returned by search_provider.search() is propagated to the caller."""
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=(
                [{"id": "s1", "updated_at": "2024-01-01T00:00:00+00:00"}],
                "opaque-cursor-token",
            )
        )
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        merged, next_cursor = await hybrid_search(
            "episodic_memory",
            search,
            relational,
            user_id="u1",
            limit=10,
        )

        assert next_cursor == "opaque-cursor-token"
        assert len(merged) == 1


@pytest.mark.asyncio
class TestHybridCount:
    async def test_returns_search_plus_relational_not_in_search(self):
        search = AsyncMock()
        search.count = AsyncMock(return_value=10)
        search.get_by_id = AsyncMock(
            side_effect=lambda table, rid, user_id=None: ({"id": rid} if rid == "in_both" else None)
        )
        relational = AsyncMock()
        relational.list = AsyncMock(
            return_value=[
                {"id": "only_relational"},
                {"id": "in_both"},
            ]
        )

        total = await hybrid_count(
            "knowledge_vault",
            search,
            relational,
            user_id="u1",
        )

        assert total == 11
        search.count.assert_awaited_once()
        relational.list.assert_awaited_once()

    async def test_hybrid_count_uses_configurable_limit(self):
        search = AsyncMock()
        search.count = AsyncMock(return_value=0)
        search.get_by_id = AsyncMock(return_value=None)
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        await hybrid_count("knowledge_vault", search, relational, user_id="u1")

        call_kw = relational.list.await_args
        assert call_kw[1]["limit"] == HYBRID_COUNT_RECENT_LIMIT == 500


class TestMergeAndDeduplicate:
    def test_sorts_by_timestamp_descending(self):
        search_results = [
            {"id": "old", "updated_at": "2020-01-01T00:00:00+00:00"},
        ]
        recent = [
            {"id": "new", "updated_at": "2025-01-01T00:00:00+00:00"},
        ]
        out = _merge_and_deduplicate(search_results, recent, limit=10)
        assert [r["id"] for r in out] == ["new", "old"]

    def test_fallback_created_at_for_sort(self):
        out = _merge_and_deduplicate(
            [{"id": "a", "created_at": "2021-01-01T00:00:00+00:00"}],
            [{"id": "b", "created_at": "2022-01-01T00:00:00+00:00"}],
            limit=10,
        )
        assert [r["id"] for r in out] == ["b", "a"]
