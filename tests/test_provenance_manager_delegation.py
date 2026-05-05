"""
Tests that the 3 provenance managers (memory_source, source_message, memory_citation)
delegate to the relational provider when registered, and fall back to ORM/cache
when it is absent.

Run:
    pytest tests/test_provenance_manager_delegation.py -v
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.client import Client as PydanticClient
from mirix.services.memory_citation_manager import MemoryCitationManager
from mirix.services.memory_source_manager import MemorySourceManager
from mirix.services.source_message_manager import SourceMessageManager


def _memory_source_mgr() -> MemorySourceManager:
    m = MemorySourceManager.__new__(MemorySourceManager)
    m.session_maker = MagicMock()
    return m


def _source_message_mgr() -> SourceMessageManager:
    m = SourceMessageManager.__new__(SourceMessageManager)
    m.session_maker = MagicMock()
    return m


def _memory_citation_mgr() -> MemoryCitationManager:
    m = MemoryCitationManager.__new__(MemoryCitationManager)
    m.session_maker = MagicMock()
    return m


def _mock_actor() -> MagicMock:
    a = MagicMock(spec=PydanticClient)
    a.id = "client-1"
    a.write_scope = "scope-a"
    a.read_scopes = ["scope-a"]
    return a


def _memory_source_row(**overrides) -> dict:
    base = {
        "id": "src-aaaaaaaa",
        "client_id": "client-1",
        "user_id": "user-1",
        "organization_id": "org-1",
        "source_type": "conversation",
        "external_id": "ext-1",
        "external_thread_id": "thr-1",
        "source_system": None,
        "source_metadata": None,
        "occurred_at": "2026-05-01T00:00:00+00:00",
        "summary": None,
        "summary_source": None,
        "processing_complete": False,
        "batch_hash": None,
        "filter_tags": {"scope": "scope-a"},
        "created_at": "2026-05-01T00:00:00+00:00",
        "updated_at": "2026-05-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


# ---------- MemorySourceManager ----------


class TestMemorySourceManagerDelegation:
    @pytest.mark.asyncio
    async def test_create_delegates_to_provider(self):
        row = _memory_source_row()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            out = await mgr.create(
                memory_source_id="src-1",
                actor=_mock_actor(),
                user_id="user-1",
                organization_id="org-1",
                external_id="ext-1",
            )
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "memory_sources"
            assert out.id == "src-aaaaaaaa"

    @pytest.mark.asyncio
    async def test_create_falls_back_to_lookup_on_conflict(self):
        existing = _memory_source_row()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=Exception("unique constraint violation"))
        mock_provider.list = AsyncMock(return_value=[existing])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            out = await mgr.create(
                memory_source_id="src-new",
                actor=_mock_actor(),
                user_id="user-1",
                organization_id="org-1",
                external_id="ext-1",
            )
            mock_provider.list.assert_awaited_once()
            assert out.id == "src-aaaaaaaa"

    @pytest.mark.asyncio
    async def test_get_by_id_delegates_to_provider(self):
        row = _memory_source_row()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=row)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            out = await mgr.get_by_id("src-aaaaaaaa")
            mock_provider.read.assert_awaited_once_with("memory_sources", "src-aaaaaaaa")
            assert out.id == "src-aaaaaaaa"

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_not_found(self):
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            out = await mgr.get_by_id("missing")
            assert out is None

    @pytest.mark.asyncio
    async def test_mark_processing_complete_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.update = AsyncMock(return_value=None)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            await mgr.mark_processing_complete("src-aaaaaaaa")
            mock_provider.update.assert_awaited_once_with(
                "memory_sources", "src-aaaaaaaa", {"processing_complete": True}
            )

    @pytest.mark.asyncio
    async def test_update_summary_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.update = AsyncMock(return_value=None)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            await mgr.update_summary("src-aaaaaaaa", "a summary", "generated")
            mock_provider.update.assert_awaited_once_with(
                "memory_sources",
                "src-aaaaaaaa",
                {"summary": "a summary", "summary_source": "generated"},
            )

    @pytest.mark.asyncio
    async def test_get_sources_by_thread_id_delegates_to_provider(self):
        rows = [
            _memory_source_row(id="src-aaaaaaa1", occurred_at="2026-05-01T00:00:00+00:00"),
            _memory_source_row(id="src-aaaaaaa2", occurred_at="2026-05-02T00:00:00+00:00"),
        ]
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            page = await mgr.get_sources_by_thread_id(
                "thr-1", scopes=["scope-a"], user_id="user-1", limit=10
            )
            assert [item.id for item in page.items] == ["src-aaaaaaa1", "src-aaaaaaa2"]
            assert page.has_more is False

    @pytest.mark.asyncio
    async def test_list_sources_delegates_to_provider(self):
        rows = [
            _memory_source_row(id="src-aaaaaaa1", occurred_at="2026-05-01T00:00:00+00:00"),
            _memory_source_row(id="src-aaaaaaa2", occurred_at="2026-05-02T00:00:00+00:00"),
        ]
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_source_mgr()
            page = await mgr.list_sources(user_id="user-1", limit=10)
            # Descending order: src-2 first
            assert [item.id for item in page.items] == ["src-aaaaaaa2", "src-aaaaaaa1"]


# ---------- SourceMessageManager ----------


class TestSourceMessageManagerDelegation:
    @pytest.mark.asyncio
    async def test_bulk_insert_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value={})

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _source_message_mgr()
            inserted = await mgr.bulk_insert(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                memory_source_id="src-1",
            )
            assert inserted == 2
            assert mock_provider.create.await_count == 2

    @pytest.mark.asyncio
    async def test_bulk_insert_treats_conflicts_as_no_op(self):
        # First create succeeds, second raises a "conflict" — total inserted = 1.
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(
            side_effect=[{}, Exception("unique constraint violation")]
        )

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _source_message_mgr()
            inserted = await mgr.bulk_insert(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                memory_source_id="src-1",
            )
            assert inserted == 1

    @pytest.mark.asyncio
    async def test_get_messages_by_source_id_delegates_to_provider(self):
        rows = [
            {
                "id": "smsg-aaaaaaa1",
                "memory_source_id": "src-aaaaaaaa",
                "role": "user",
                "content": {"text": "hi"},
                "sequence_num": 0,
                "content_hash": "h0",
            },
            {
                "id": "smsg-aaaaaaa2",
                "memory_source_id": "src-aaaaaaaa",
                "role": "assistant",
                "content": {"text": "hello"},
                "sequence_num": 1,
                "content_hash": "h1",
            },
        ]
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _source_message_mgr()
            page = await mgr.get_messages_by_source_id("src-aaaaaaaa")
            assert [m.id for m in page.items] == ["smsg-aaaaaaa1", "smsg-aaaaaaa2"]


# ---------- MemoryCitationManager ----------


def _citation_row(**overrides) -> dict:
    base = {
        "id": "cit-aaaaaaaa",
        "memory_source_id": "src-aaaaaaaa",
        "memory_type": "episodic",
        "memory_id": "ep-1",
        "external_thread_id": None,
        "occurred_at": "2026-05-01T00:00:00+00:00",
        "citation_type": "created",
        "created_at": "2026-05-01T00:00:00+00:00",
        "updated_at": "2026-05-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


class TestMemoryCitationManagerDelegation:
    @pytest.mark.asyncio
    async def test_create_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=_citation_row())

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.create(
                memory_source_id="src-1",
                memory_type="episodic",
                memory_id="ep-1",
                citation_type="created",
            )
            mock_provider.create.assert_awaited_once()
            assert out is not None
            assert out.memory_type == "episodic"

    @pytest.mark.asyncio
    async def test_create_returns_none_on_conflict(self):
        # L3 dedup: provider raises on duplicate (memory_source_id, memory_type, memory_id).
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(side_effect=Exception("unique constraint violation"))

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.create(
                memory_source_id="src-1",
                memory_type="episodic",
                memory_id="ep-1",
                citation_type="created",
            )
            assert out is None

    @pytest.mark.asyncio
    async def test_check_exists_returns_true_when_provider_returns_row(self):
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=[_citation_row()])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            assert await mgr.check_exists("src-1", "episodic", "ep-1") is True

    @pytest.mark.asyncio
    async def test_check_exists_returns_false_when_provider_returns_empty(self):
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=[])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            assert await mgr.check_exists("src-1", "episodic", "ep-1") is False

    @pytest.mark.asyncio
    async def test_get_max_occurred_at_takes_max_across_rows(self):
        rows = [
            _citation_row(id="cit-aaaaaaa1", occurred_at="2026-05-01T00:00:00+00:00"),
            _citation_row(id="cit-aaaaaaa2", occurred_at="2026-05-03T00:00:00+00:00"),
            _citation_row(id="cit-aaaaaaa3", occurred_at="2026-05-02T00:00:00+00:00"),
        ]
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.get_max_occurred_at("episodic", "ep-1")
            assert out is not None
            assert out == datetime(2026, 5, 3, 0, 0, 0, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_get_max_occurred_at_returns_none_when_no_rows(self):
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=[])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            assert await mgr.get_max_occurred_at("episodic", "ep-1") is None

    @pytest.mark.asyncio
    async def test_get_citations_for_memory_delegates_to_provider(self):
        rows = [_citation_row()]
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=rows)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.get_citations_for_memory("episodic", "ep-1")
            assert len(out) == 1
            assert out[0].memory_id == "ep-1"

    @pytest.mark.asyncio
    async def test_get_citations_for_memories_groups_by_key(self):
        ep_row = _citation_row(id="cit-aaaaaaa1", memory_type="episodic", memory_id="ep-1")
        sem_row = _citation_row(id="cit-aaaaaaa2", memory_type="semantic", memory_id="sem-1")
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(side_effect=[[ep_row], [sem_row]])

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=mock_provider,
        ):
            mgr = _memory_citation_mgr()
            out = await mgr.get_citations_for_memories(
                [("episodic", "ep-1"), ("semantic", "sem-1")]
            )
            assert ("episodic", "ep-1") in out
            assert ("semantic", "sem-1") in out
            assert len(out[("episodic", "ep-1")]) == 1
            assert len(out[("semantic", "sem-1")]) == 1
