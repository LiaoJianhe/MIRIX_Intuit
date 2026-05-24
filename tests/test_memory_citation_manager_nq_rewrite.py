"""Unit tests for VEPAGE-1107 manager rewrite changes.

Covers:
- ``_pydantic_citation_from_row`` defensive constructor that handles rows
  missing ``memory_source_id`` (defensive fallback for code paths still
  using the generic ``provider.list`` against ``memory_citations``).
- ``get_citations_for_memories`` rewrite that routes through
  ``memory_citation_manager.get_citations_for_memory`` named query
  instead of ``provider.list``.

These tests pin the behavior the FST citation tests depend on: when a
search response is being enriched with citations, the row hydration must
not crash even if a row arrives without the FK column populated.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.services.memory_citation_manager import (
    MemoryCitationManager,
    _pydantic_citation_from_row,
)


# ---------------------------------------------------------------------------
# _pydantic_citation_from_row
# ---------------------------------------------------------------------------


def _well_formed_row() -> Dict[str, Any]:
    """A row as the named-query path produces it — fully populated.

    IDs use the format Pydantic models enforce (``^cit-[a-fA-F0-9]{8}`` etc).
    """
    return {
        "id": "cit-abcdef01-2345-6789-abcd-ef0123456789",
        "memory_source_id": "src-abcdef01-2345-6789-abcd-ef0123456789",
        "memory_type": "episodic",
        "memory_id": "ep-abcdef01",
        "citation_type": "created",
        "occurred_at": None,
        "external_thread_id": None,
    }


class TestPydanticCitationFromRow:
    def test_well_formed_row_constructs_directly(self):
        """Rows with all required fields should construct without warning."""
        row = _well_formed_row()
        citation = _pydantic_citation_from_row(row)
        assert citation.id == "cit-abcdef01-2345-6789-abcd-ef0123456789"
        assert citation.memory_source_id == "src-abcdef01-2345-6789-abcd-ef0123456789"
        assert citation.memory_type == "episodic"
        assert citation.memory_id == "ep-abcdef01"

    def test_missing_memory_source_id_falls_back_to_empty_string(self):
        """Row missing the FK should synthesize an empty string,
        not crash. Empty string is intentionally a sentinel — downstream
        consumers can detect and skip these rather than crash the whole
        search response. The function also logs a warning (verified by
        manual inspection of mirix.log — pytest's caplog/capsys don't
        intercept the MIRIX logger's stream)."""
        row = _well_formed_row()
        del row["memory_source_id"]
        citation = _pydantic_citation_from_row(row)
        assert citation.memory_source_id == ""
        assert citation.id == "cit-abcdef01-2345-6789-abcd-ef0123456789"

    def test_explicit_none_memory_source_id_falls_back(self):
        """Same as missing, but the key is present with a None value
        (this is what the legacy ``provider.list`` path actually
        produced — ``memory_source`` was populated as None on the
        entity, ``from_entity`` skipped the None, and the row arrived
        without the field set or with a None)."""
        row = _well_formed_row()
        row["memory_source_id"] = None
        citation = _pydantic_citation_from_row(row)
        assert citation.memory_source_id == ""

    def test_does_not_mutate_input_row(self):
        """The helper should not mutate the caller's row dict — it
        makes a copy when synthesizing the missing field. This matters
        if the caller is iterating over a list and re-reading the row
        elsewhere (e.g., logging)."""
        row = _well_formed_row()
        del row["memory_source_id"]
        original_keys = set(row.keys())
        _pydantic_citation_from_row(row)
        assert set(row.keys()) == original_keys
        assert "memory_source_id" not in row


# ---------------------------------------------------------------------------
# get_citations_for_memories — NQ routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetCitationsForMemoriesNQ:
    async def test_routes_through_named_query_not_provider_list(self):
        """The rewrite must call ``find_using_named_query`` with the
        ``memory_citation_manager.get_citations_for_memory`` query name —
        NOT the legacy ``provider.list`` path. This is the heart of the
        VEPAGE-1107 fix: NQ projects the FK column explicitly so the
        row arrives with ``memory_source_id`` populated.
        """
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(
            return_value=[_well_formed_row()]
        )
        # If the manager fell back to provider.list, the test would
        # call it instead — ensure it doesn't.
        mock_provider.list = AsyncMock(
            side_effect=AssertionError(
                "get_citations_for_memories must use find_using_named_query, "
                "not provider.list — see VEPAGE-1107"
            )
        )

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = MemoryCitationManager()
            result = await mgr.get_citations_for_memories(
                [("episodic", "ep-abcdef01")]
            )

        mock_provider.find_using_named_query.assert_awaited_once()
        call = mock_provider.find_using_named_query.await_args
        # Positional args: (table, query_name)
        assert call.args[0] == "memory_citations"
        assert call.args[1] == "memory_citation_manager.get_citations_for_memory"
        # kwargs: params + page_size
        assert call.kwargs["params"] == {
            "memoryType": "episodic",
            "memoryId": "ep-abcdef01",
        }
        assert call.kwargs["page_size"] == 1000
        assert ("episodic", "ep-abcdef01") in result
        assert len(result[("episodic", "ep-abcdef01")]) == 1

    async def test_fans_out_per_memory_key(self):
        """Multiple ``(memory_type, memory_id)`` pairs result in one
        named-query call per key — the existing per-key loop pattern
        is preserved (a tuple-IN NQ would be a future optimization)."""
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(
            return_value=[_well_formed_row()]
        )

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = MemoryCitationManager()
            await mgr.get_citations_for_memories(
                [("episodic", "ep-1"), ("semantic", "sem-1"), ("core", "block-1")]
            )

        assert mock_provider.find_using_named_query.await_count == 3

    async def test_row_missing_fk_doesnt_crash_response(self):
        """When IPSR returns a row without ``memory_source_id`` (e.g.,
        if a stale code path is involved, or if the NQ deploy hasn't
        landed), the manager must NOT raise — that would empty the
        whole search response. Instead, the helper synthesizes an
        empty string and logs a warning."""
        bad_row = _well_formed_row()
        del bad_row["memory_source_id"]
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[bad_row])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = MemoryCitationManager()
            result = await mgr.get_citations_for_memories(
                [("episodic", "ep-abcdef01")]
            )

        # Returned without raising — the citation is present with the
        # sentinel empty memory_source_id.
        citations = result[("episodic", "ep-abcdef01")]
        assert len(citations) == 1
        assert citations[0].memory_source_id == ""

    async def test_empty_memory_keys_returns_empty(self):
        """Empty input → empty output, no provider calls."""
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock()

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = MemoryCitationManager()
            result = await mgr.get_citations_for_memories([])

        assert result == {}
        mock_provider.find_using_named_query.assert_not_awaited()
