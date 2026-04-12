"""Unit tests for S11: citation inclusion in search results.

Verifies:
1. MemoryCitationManager.get_citations_for_memories() batch query
2. _attach_citations_to_results() helper wires citations into result dicts
3. search_memory and search_memory_all_users pass include_citations through

These are fast unit tests — they mock the DB/manager and require no running server.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from mirix.schemas.memory_citation import MemoryCitation as PydanticMemoryCitation

pytestmark = pytest.mark.asyncio


# --- Helpers ---


_COUNTER = 0


_SENTINEL = object()


def _make_citation(
    memory_type="episodic",
    memory_id="mem-1",
    memory_source_id="src-1",
    citation_type="created",
    external_thread_id=None,
    occurred_at=_SENTINEL,
):
    if occurred_at is _SENTINEL:
        occurred_at = datetime(2026, 4, 10, tzinfo=timezone.utc)
    return PydanticMemoryCitation(
        id=PydanticMemoryCitation._generate_id(),
        memory_type=memory_type,
        memory_id=memory_id,
        memory_source_id=memory_source_id,
        citation_type=citation_type,
        external_thread_id=external_thread_id,
        occurred_at=occurred_at,
    )


# --- get_citations_for_memories ---


class TestGetCitationsForMemories:
    """Tests for MemoryCitationManager.get_citations_for_memories()."""

    @pytest.fixture
    def mock_session(self):
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return session

    @pytest.fixture
    def citation_mgr(self, mock_session):
        with patch("mirix.services.memory_citation_manager.MemoryCitationManager.__init__", return_value=None):
            from mirix.services.memory_citation_manager import MemoryCitationManager

            mgr = MemoryCitationManager.__new__(MemoryCitationManager)
            mgr.session_maker = MagicMock(return_value=mock_session)
            return mgr

    async def test_empty_keys_returns_empty_dict(self, citation_mgr):
        result = await citation_mgr.get_citations_for_memories([])
        assert result == {}

    async def test_groups_by_memory_key(self, citation_mgr, mock_session):
        """Citations are grouped by (memory_type, memory_id) tuple."""
        row1 = MagicMock(
            id=PydanticMemoryCitation._generate_id(),
            memory_type="episodic",
            memory_id="mem-1",
            memory_source_id="src-a",
            external_thread_id=None,
            occurred_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            citation_type="created",
            message_ids=None,
            created_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            updated_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        )
        row2 = MagicMock(
            id=PydanticMemoryCitation._generate_id(),
            memory_type="episodic",
            memory_id="mem-1",
            memory_source_id="src-b",
            external_thread_id="thread-1",
            occurred_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
            citation_type="updated",
            message_ids=None,
            created_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
            updated_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
        )
        row3 = MagicMock(
            id=PydanticMemoryCitation._generate_id(),
            memory_type="semantic",
            memory_id="mem-2",
            memory_source_id="src-a",
            external_thread_id=None,
            occurred_at=None,
            citation_type="created",
            message_ids=None,
            created_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            updated_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        )

        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = [row1, row2, row3]
        mock_session.execute = AsyncMock(return_value=execute_result)

        result = await citation_mgr.get_citations_for_memories([("episodic", "mem-1"), ("semantic", "mem-2")])

        assert ("episodic", "mem-1") in result
        assert ("semantic", "mem-2") in result
        assert len(result[("episodic", "mem-1")]) == 2
        assert len(result[("semantic", "mem-2")]) == 1
        assert result[("episodic", "mem-1")][0].memory_source_id == "src-a"
        assert result[("episodic", "mem-1")][1].citation_type == "updated"

    async def test_missing_keys_not_in_result(self, citation_mgr, mock_session):
        """Memory keys with no citations are simply absent from the dict."""
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=execute_result)

        result = await citation_mgr.get_citations_for_memories([("episodic", "mem-missing")])
        assert result == {}


# --- _attach_citations_to_results ---


class TestAttachCitationsToResults:
    """Tests for the _attach_citations_to_results helper in rest_api."""

    async def test_attaches_citations_to_matching_results(self):
        results = [
            {"memory_type": "episodic", "id": "mem-1", "summary": "test"},
            {"memory_type": "semantic", "id": "mem-2", "summary": "other"},
        ]

        mock_grouped = {
            ("episodic", "mem-1"): [
                _make_citation(memory_type="episodic", memory_id="mem-1", memory_source_id="src-a"),
            ],
        }

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            instance = MockMgr.return_value
            instance.get_citations_for_memories = AsyncMock(return_value=mock_grouped)

            from mirix.server.rest_api import _attach_citations_to_results

            enriched = await _attach_citations_to_results(results)

        assert len(enriched[0]["citations"]) == 1
        assert enriched[0]["citations"][0]["memory_source_id"] == "src-a"
        assert enriched[0]["citations"][0]["citation_type"] == "created"
        assert enriched[1]["citations"] == []

    async def test_core_memory_gets_citations(self):
        """Core memory results get citations like any other memory type."""
        results = [
            {"memory_type": "core", "id": "block-1", "label": "persona", "value": "friendly"},
        ]

        mock_grouped = {
            ("core", "block-1"): [
                _make_citation(
                    memory_type="core", memory_id="block-1", memory_source_id="src-x", citation_type="updated"
                ),
            ],
        }

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            instance = MockMgr.return_value
            instance.get_citations_for_memories = AsyncMock(return_value=mock_grouped)

            from mirix.server.rest_api import _attach_citations_to_results

            enriched = await _attach_citations_to_results(results)

        assert len(enriched[0]["citations"]) == 1
        assert enriched[0]["citations"][0]["memory_source_id"] == "src-x"
        assert enriched[0]["citations"][0]["citation_type"] == "updated"

    async def test_empty_results_returns_empty(self):
        from mirix.server.rest_api import _attach_citations_to_results

        enriched = await _attach_citations_to_results([])
        assert enriched == []

    async def test_occurred_at_serialized_as_iso(self):
        ts = datetime(2026, 4, 10, 12, 30, 0, tzinfo=timezone.utc)
        results = [{"memory_type": "episodic", "id": "mem-1"}]

        mock_grouped = {
            ("episodic", "mem-1"): [
                _make_citation(memory_id="mem-1", occurred_at=ts),
            ],
        }

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            instance = MockMgr.return_value
            instance.get_citations_for_memories = AsyncMock(return_value=mock_grouped)

            from mirix.server.rest_api import _attach_citations_to_results

            enriched = await _attach_citations_to_results(results)

        assert enriched[0]["citations"][0]["occurred_at"] == "2026-04-10T12:30:00+00:00"

    async def test_null_occurred_at_serialized_as_none(self):
        results = [{"memory_type": "episodic", "id": "mem-1"}]

        mock_grouped = {
            ("episodic", "mem-1"): [
                _make_citation(memory_id="mem-1", occurred_at=None),
            ],
        }

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            instance = MockMgr.return_value
            instance.get_citations_for_memories = AsyncMock(return_value=mock_grouped)

            from mirix.server.rest_api import _attach_citations_to_results

            enriched = await _attach_citations_to_results(results)

        assert enriched[0]["citations"][0]["occurred_at"] is None


# --- _attach_citations_to_memories_dict ---


class TestAttachCitationsToMemoriesDict:
    """Tests for citation attachment to the nested memories dict (conversation/topic mode)."""

    async def test_attaches_to_episodic_recent_and_relevant(self):
        memories = {
            "episodic": {
                "total_count": 2,
                "recent": [{"id": "ep-1", "summary": "recent event"}],
                "relevant": [{"id": "ep-2", "summary": "relevant event"}],
            },
        }

        mock_grouped = {
            ("episodic", "ep-1"): [
                _make_citation(memory_type="episodic", memory_id="ep-1", memory_source_id="src-a"),
            ],
        }

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            instance = MockMgr.return_value
            instance.get_citations_for_memories = AsyncMock(return_value=mock_grouped)

            from mirix.server.rest_api import _attach_citations_to_memories_dict

            result = await _attach_citations_to_memories_dict(memories)

        assert len(result["episodic"]["recent"][0]["citations"]) == 1
        assert result["episodic"]["recent"][0]["citations"][0]["memory_source_id"] == "src-a"
        assert result["episodic"]["relevant"][0]["citations"] == []

    async def test_attaches_to_items_list(self):
        memories = {
            "semantic": {
                "total_count": 1,
                "items": [{"id": "sem-1", "summary": "a fact"}],
            },
            "resource": {
                "total_count": 1,
                "items": [{"id": "res-1", "title": "a doc"}],
            },
        }

        mock_grouped = {
            ("semantic", "sem-1"): [
                _make_citation(memory_type="semantic", memory_id="sem-1", memory_source_id="src-b"),
            ],
            ("resource", "res-1"): [
                _make_citation(memory_type="resource", memory_id="res-1", memory_source_id="src-c"),
                _make_citation(
                    memory_type="resource", memory_id="res-1", memory_source_id="src-d", citation_type="updated"
                ),
            ],
        }

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            instance = MockMgr.return_value
            instance.get_citations_for_memories = AsyncMock(return_value=mock_grouped)

            from mirix.server.rest_api import _attach_citations_to_memories_dict

            result = await _attach_citations_to_memories_dict(memories)

        assert len(result["semantic"]["items"][0]["citations"]) == 1
        assert len(result["resource"]["items"][0]["citations"]) == 2

    async def test_attaches_to_core_scopes(self):
        """Core memory items nested under scopes get citations."""
        memories = {
            "core": {
                "total_count": 2,
                "scopes": {
                    "default": {"items": [{"id": "block-1", "label": "persona"}]},
                    "other": {"items": [{"id": "block-2", "label": "human"}]},
                },
            },
        }

        mock_grouped = {
            ("core", "block-1"): [
                _make_citation(
                    memory_type="core", memory_id="block-1", memory_source_id="src-z", citation_type="updated"
                ),
            ],
        }

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            instance = MockMgr.return_value
            instance.get_citations_for_memories = AsyncMock(return_value=mock_grouped)

            from mirix.server.rest_api import _attach_citations_to_memories_dict

            result = await _attach_citations_to_memories_dict(memories)

        assert len(result["core"]["scopes"]["default"]["items"][0]["citations"]) == 1
        assert result["core"]["scopes"]["default"]["items"][0]["citations"][0]["memory_source_id"] == "src-z"
        assert result["core"]["scopes"]["other"]["items"][0]["citations"] == []

    async def test_empty_memories_dict(self):
        from mirix.server.rest_api import _attach_citations_to_memories_dict

        result = await _attach_citations_to_memories_dict({})
        assert result == {}

    async def test_same_id_in_recent_and_relevant_both_get_citations(self):
        """If the same memory appears in both recent and relevant, both get citations."""
        memories = {
            "episodic": {
                "total_count": 1,
                "recent": [{"id": "ep-dup", "summary": "event"}],
                "relevant": [{"id": "ep-dup", "summary": "event"}],
            },
        }

        mock_grouped = {
            ("episodic", "ep-dup"): [
                _make_citation(memory_type="episodic", memory_id="ep-dup", memory_source_id="src-x"),
            ],
        }

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            instance = MockMgr.return_value
            instance.get_citations_for_memories = AsyncMock(return_value=mock_grouped)

            from mirix.server.rest_api import _attach_citations_to_memories_dict

            result = await _attach_citations_to_memories_dict(memories)

        assert len(result["episodic"]["recent"][0]["citations"]) == 1
        assert len(result["episodic"]["relevant"][0]["citations"]) == 1
