"""
Tests for mirix.services.hybrid_search_helper.

Usage:
    pytest tests/test_hybrid_search.py -v
"""

from unittest.mock import AsyncMock

import pytest

from mirix.services.hybrid_search_helper import (
    DEFAULT_RECENT_CAP,
    DedupCandidates,
    fetch_and_dedup_candidates,
    get_hybrid_window_seconds,
    set_hybrid_window_seconds,
)


@pytest.fixture(autouse=True)
def hybrid_window_five_seconds():
    set_hybrid_window_seconds(5)
    yield


@pytest.mark.asyncio
class TestFetchAndDedupCandidates:
    """The save-flow helper returns labeled buckets without merging.

    The dedup judgment is left to the calling sub-agent's LLM, which sees
    ``relevant`` (Search-provider ranked) and ``recent`` (Relational
    just-written) separately so it can reason about each bucket on its own
    terms.
    """

    async def test_returns_labeled_buckets_unmerged(self):
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=(
                [{"id": "s1", "updated_at": "2024-01-01T00:00:00+00:00"}],
                None,
            )
        )
        relational = AsyncMock()
        relational.list = AsyncMock(
            return_value=[{"id": "r1", "updated_at": "2025-06-01T00:00:00+00:00"}]
        )

        candidates = await fetch_and_dedup_candidates(
            "episodic_memory",
            search,
            relational,
            user_id="u1",
            limit=10,
        )

        assert isinstance(candidates, DedupCandidates)
        assert [r["id"] for r in candidates.relevant] == ["s1"]
        assert [r["id"] for r in candidates.recent] == ["r1"]
        search.search.assert_awaited_once()
        relational.list.assert_awaited_once()

    async def test_overlap_appears_in_both_buckets(self):
        """A row that appears in both Search and the Relational recent window
        is returned in both buckets verbatim. No dedup is applied; the
        labeled-bucket contract is that the caller decides what to do with
        overlapping IDs."""
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=(
                [{"id": "1", "updated_at": "2025-01-01T00:00:00+00:00", "src": "search"}],
                None,
            )
        )
        relational = AsyncMock()
        relational.list = AsyncMock(
            return_value=[
                {"id": "1", "updated_at": "2025-01-02T00:00:00+00:00", "src": "relational"},
            ]
        )

        candidates = await fetch_and_dedup_candidates(
            "raw_memory", search, relational, limit=10
        )

        assert len(candidates.relevant) == 1
        assert candidates.relevant[0]["src"] == "search"
        assert len(candidates.recent) == 1
        assert candidates.recent[0]["src"] == "relational"

    async def test_relational_call_uses_recent_window(self):
        search = AsyncMock()
        search.search = AsyncMock(return_value=([], None))
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        await fetch_and_dedup_candidates(
            "episodic_memory",
            search,
            relational,
            user_id="u1",
            limit=10,
        )

        call_kw = relational.list.await_args
        assert call_kw[0][0] == "episodic_memory"
        assert call_kw[1]["user_id"] == "u1"
        tr = call_kw[1]["time_range"]
        assert "updated_at__gte" in tr
        assert "created_at__gte" in tr
        assert call_kw[1]["time_range_or_null_updated"] is True

    async def test_recent_cap_applied_to_relational_list(self):
        search = AsyncMock()
        search.search = AsyncMock(return_value=([], None))
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        await fetch_and_dedup_candidates(
            "semantic_memory",
            search,
            relational,
            user_id="u1",
            limit=10,
            recent_cap=42,
        )

        assert relational.list.await_args[1]["limit"] == 42

    async def test_recent_cap_defaults_to_500(self):
        search = AsyncMock()
        search.search = AsyncMock(return_value=([], None))
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        await fetch_and_dedup_candidates(
            "semantic_memory", search, relational, user_id="u1"
        )

        assert relational.list.await_args[1]["limit"] == DEFAULT_RECENT_CAP == 500

    async def test_search_raises_propagates(self):
        search = AsyncMock()
        search.search = AsyncMock(side_effect=RuntimeError("search down"))
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        with pytest.raises(RuntimeError, match="search down"):
            await fetch_and_dedup_candidates("episodic_memory", search, relational)

    async def test_relational_raises_propagates(self):
        search = AsyncMock()
        search.search = AsyncMock(return_value=([], None))
        relational = AsyncMock()
        relational.list = AsyncMock(side_effect=RuntimeError("relational down"))

        with pytest.raises(RuntimeError, match="relational down"):
            await fetch_and_dedup_candidates("episodic_memory", search, relational)

    async def test_relevant_preserves_search_order(self):
        """The ranked Search-provider order is preserved as-is in
        ``relevant``; the helper deliberately does not re-sort by
        timestamp."""
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=(
                [
                    {"id": "c", "updated_at": "2025-01-01T00:00:00+00:00"},
                    {"id": "a", "updated_at": "2025-01-03T00:00:00+00:00"},
                    {"id": "b", "updated_at": "2025-01-02T00:00:00+00:00"},
                ],
                None,
            )
        )
        relational = AsyncMock()
        relational.list = AsyncMock(return_value=[])

        candidates = await fetch_and_dedup_candidates(
            "semantic_memory", search, relational, limit=10
        )
        assert [r["id"] for r in candidates.relevant] == ["c", "a", "b"]


class TestWindowAccessors:
    def test_set_and_get_round_trip(self):
        original = get_hybrid_window_seconds()
        try:
            set_hybrid_window_seconds(42)
            assert get_hybrid_window_seconds() == 42
        finally:
            set_hybrid_window_seconds(original)
