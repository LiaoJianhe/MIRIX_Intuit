"""Unit tests for S9 retrieval endpoints — memory sources and threads.

Tests the manager retrieval methods and API route handlers using mocks.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.memory_source import MemorySource
from mirix.schemas.source_message import SourceMessage


# --- Test data factories ---

SRC_ID_1 = "src-aaaaaaaa-1111-2222-3333-444444444444"
SRC_ID_2 = "src-bbbbbbbb-1111-2222-3333-444444444444"
SMSG_ID_1 = "smsg-aaaaaaaa-1111-2222-3333-444444444444"
SMSG_ID_2 = "smsg-bbbbbbbb-1111-2222-3333-444444444444"


def _make_source(id=SRC_ID_1, client_id="client-1", user_id="user-1", external_thread_id="thread-1", occurred_at=None):
    return MemorySource(
        id=id,
        client_id=client_id,
        user_id=user_id,
        organization_id="org-1",
        external_thread_id=external_thread_id,
        source_type="conversation",
        processing_complete=False,
        occurred_at=occurred_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_message(id=SMSG_ID_1, memory_source_id=SRC_ID_1, sequence_num=0, role="user", content=None):
    return SourceMessage(
        id=id,
        memory_source_id=memory_source_id,
        role=role,
        content=content or {"text": "hello"},
        sequence_num=sequence_num,
        content_hash="abc123",
    )


# --- MemorySourceManager.get_sources_by_thread_id ---

class TestGetSourcesByThreadId:

    @pytest.mark.asyncio
    async def test_returns_sources_for_thread(self):
        from mirix.services.memory_source_manager import MemorySourceManager

        src1 = _make_source(id=SRC_ID_1)
        src2 = _make_source(id=SRC_ID_2)

        mock_record1 = MagicMock()
        mock_record1.to_pydantic.return_value = src1
        mock_record2 = MagicMock()
        mock_record2.to_pydantic.return_value = src2

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_record1, mock_record2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_context():
            yield mock_session

        mgr = MemorySourceManager.__new__(MemorySourceManager)
        mgr.session_maker = mock_context

        result = await mgr.get_sources_by_thread_id(
            external_thread_id="thread-1",
            client_id="client-1",
            user_id="user-1",
        )

        assert len(result) == 2
        assert result[0].id == SRC_ID_1
        assert result[1].id == SRC_ID_2


# --- SourceMessageManager.get_messages_by_source_id ---

class TestGetMessagesBySourceId:

    @pytest.mark.asyncio
    async def test_returns_messages_for_source(self):
        from mirix.services.source_message_manager import SourceMessageManager

        msg1 = _make_message(id=SMSG_ID_1, sequence_num=0)
        msg2 = _make_message(id=SMSG_ID_2, sequence_num=1)

        mock_record1 = MagicMock()
        mock_record1.to_pydantic.return_value = msg1
        mock_record2 = MagicMock()
        mock_record2.to_pydantic.return_value = msg2

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_record1, mock_record2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_context():
            yield mock_session

        mgr = SourceMessageManager.__new__(SourceMessageManager)
        mgr.session_maker = mock_context

        result = await mgr.get_messages_by_source_id(memory_source_id="src-1")

        assert len(result) == 2
        assert result[0].id == SMSG_ID_1
        assert result[1].id == SMSG_ID_2


# --- SourceMessageManager.get_messages_by_thread_id ---

class TestGetMessagesByThreadId:

    @pytest.mark.asyncio
    async def test_returns_messages_for_thread(self):
        from mirix.services.source_message_manager import SourceMessageManager

        msg1 = _make_message(id=SMSG_ID_1, memory_source_id=SRC_ID_1, sequence_num=0)
        msg2 = _make_message(id=SMSG_ID_2, memory_source_id=SRC_ID_2, sequence_num=0)

        mock_record1 = MagicMock()
        mock_record1.to_pydantic.return_value = msg1
        mock_record2 = MagicMock()
        mock_record2.to_pydantic.return_value = msg2

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_record1, mock_record2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_context():
            yield mock_session

        mgr = SourceMessageManager.__new__(SourceMessageManager)
        mgr.session_maker = mock_context

        result = await mgr.get_messages_by_thread_id(external_thread_id="thread-1")

        assert len(result) == 2


# --- REST API route handlers ---

class TestGetMemorySourceRoute:

    @pytest.mark.asyncio
    async def test_returns_source(self):
        from mirix.server.rest_api import get_memory_source

        source = _make_source()
        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_by_id = AsyncMock(return_value=source)

            with patch("mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")):
                with patch("mirix.server.rest_api.get_server"):
                    result = await get_memory_source(source_id=SRC_ID_1, x_client_id="client-1")

        assert result.id == SRC_ID_1

    @pytest.mark.asyncio
    async def test_404_when_not_found(self):
        from mirix.server.rest_api import get_memory_source

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_by_id = AsyncMock(return_value=None)

            with patch("mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")):
                with patch("mirix.server.rest_api.get_server"):
                    from fastapi import HTTPException as FastHTTPException
                    with pytest.raises(FastHTTPException) as exc_info:
                        await get_memory_source(source_id="src-cccccccc-0000-0000-0000-000000000000", x_client_id="client-1")

                    assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_wrong_client(self):
        from mirix.server.rest_api import get_memory_source

        source = _make_source(client_id="client-1")
        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_by_id = AsyncMock(return_value=source)

            with patch("mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("other-client", "org-1")):
                with patch("mirix.server.rest_api.get_server"):
                    from fastapi import HTTPException as FastHTTPException
                    with pytest.raises(FastHTTPException) as exc_info:
                        await get_memory_source(source_id=SRC_ID_1, x_client_id="other-client")

                    assert exc_info.value.status_code == 404


class TestGetThreadSourcesRoute:

    @pytest.mark.asyncio
    async def test_returns_sources(self):
        from mirix.server.rest_api import get_thread_sources

        sources = [_make_source(id=SRC_ID_1), _make_source(id=SRC_ID_2)]
        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_sources_by_thread_id = AsyncMock(return_value=sources)

            with patch("mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")):
                with patch("mirix.server.rest_api.get_server"):
                    result = await get_thread_sources(
                        external_thread_id="thread-1",
                        user_id="user-1",
                        x_client_id="client-1",
                    )

        assert len(result) == 2


class TestGetThreadMessagesRoute:

    @pytest.mark.asyncio
    async def test_404_when_thread_not_found(self):
        from mirix.server.rest_api import get_thread_messages

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockSourceMgr:
            mock_source_instance = MockSourceMgr.return_value
            mock_source_instance.get_sources_by_thread_id = AsyncMock(return_value=[])

            with patch("mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")):
                with patch("mirix.server.rest_api.get_server"):
                    from fastapi import HTTPException as FastHTTPException
                    with pytest.raises(FastHTTPException) as exc_info:
                        await get_thread_messages(
                            external_thread_id="missing-thread",
                            user_id="user-1",
                            x_client_id="client-1",
                        )

                    assert exc_info.value.status_code == 404
