"""Tests that AddMemoryRequest accepts direct_writes and add_memory threads them to put_messages."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_add_memory_request_accepts_direct_writes():
    """AddMemoryRequest constructs with direct_writes=[DirectWriteInput(...)]."""
    from mirix.server.rest_api import AddMemoryRequest, DirectWriteInput

    req = AddMemoryRequest(
        meta_agent_id="agent-1",
        messages=[],
        direct_writes=[
            DirectWriteInput(
                memory_type="episodic",
                payload={
                    "event_type": "e",
                    "summary": "s",
                    "details": "d",
                    "event_actor": "system",
                },
            ),
        ],
    )
    assert req.direct_writes is not None
    assert len(req.direct_writes) == 1
    assert req.direct_writes[0].memory_type == "episodic"
    assert req.direct_writes[0].payload["event_type"] == "e"
    assert req.direct_writes[0].payload["event_actor"] == "system"


@pytest.mark.asyncio
async def test_add_memory_request_direct_writes_defaults_to_none():
    """direct_writes is optional and defaults to None when omitted."""
    from mirix.server.rest_api import AddMemoryRequest

    req = AddMemoryRequest(meta_agent_id="agent-1", messages=[])
    assert req.direct_writes is None


@pytest.mark.asyncio
async def test_add_memory_threads_direct_writes_to_put_messages():
    """add_memory(...) with direct_writes set passes them as a list of dicts to put_messages."""
    from mirix.server.rest_api import (
        AddMemoryRequest,
        DirectWriteInput,
        add_memory,
    )

    # Build a request that carries direct_writes and one dummy message turn
    req = AddMemoryRequest(
        meta_agent_id="agent-1",
        user_id="user-1",
        messages=[{"role": "user", "content": "hello"}],
        filter_tags={"k": "v"},
        external_id="ext-1",
        source_type="engagement",
        source_system="test-system",
        source_metadata={"display_id": "D1"},
        direct_writes=[
            DirectWriteInput(
                memory_type="episodic",
                payload={
                    "event_type": "e",
                    "summary": "s",
                    "details": "d",
                    "event_actor": "system",
                    "occurred_at": "2026-04-17T10:00:00Z",
                },
            ),
        ],
    )

    # Stub client + server + meta_agent so we don't need real DB
    fake_client = MagicMock()
    fake_client.id = "client-1"
    fake_client.write_scope = "test-scope"

    fake_meta_agent = MagicMock()
    fake_meta_agent.id = "agent-1"

    fake_server = MagicMock()
    fake_server.client_manager = MagicMock()
    fake_server.client_manager.get_client_by_id = AsyncMock(return_value=fake_client)
    fake_server.agent_manager = MagicMock()
    fake_server.agent_manager.get_agent_by_id = AsyncMock(return_value=fake_meta_agent)

    with patch("mirix.server.rest_api.get_server", return_value=fake_server), patch(
        "mirix.server.rest_api.get_client_and_org",
        new_callable=AsyncMock,
        return_value=("client-1", "org-1"),
    ), patch(
        "mirix.server.rest_api.put_messages", new_callable=AsyncMock
    ) as mock_put:
        result = await add_memory(req, x_org_id="org-1", x_client_id="client-1")

    assert result["success"] is True
    mock_put.assert_awaited_once()
    kwargs = mock_put.call_args.kwargs

    assert "direct_writes" in kwargs
    assert kwargs["direct_writes"] == [
        {
            "memory_type": "episodic",
            "payload": {
                "event_type": "e",
                "summary": "s",
                "details": "d",
                "event_actor": "system",
                "occurred_at": "2026-04-17T10:00:00Z",
            },
        }
    ]


@pytest.mark.asyncio
async def test_add_memory_direct_writes_none_passes_none_to_put_messages():
    """When request.direct_writes is None, put_messages is called with direct_writes=None."""
    from mirix.server.rest_api import AddMemoryRequest, add_memory

    req = AddMemoryRequest(
        meta_agent_id="agent-1",
        user_id="user-1",
        messages=[{"role": "user", "content": "hello"}],
    )

    fake_client = MagicMock()
    fake_client.id = "client-1"
    fake_client.write_scope = "test-scope"

    fake_meta_agent = MagicMock()
    fake_meta_agent.id = "agent-1"

    fake_server = MagicMock()
    fake_server.client_manager = MagicMock()
    fake_server.client_manager.get_client_by_id = AsyncMock(return_value=fake_client)
    fake_server.agent_manager = MagicMock()
    fake_server.agent_manager.get_agent_by_id = AsyncMock(return_value=fake_meta_agent)

    with patch("mirix.server.rest_api.get_server", return_value=fake_server), patch(
        "mirix.server.rest_api.get_client_and_org",
        new_callable=AsyncMock,
        return_value=("client-1", "org-1"),
    ), patch(
        "mirix.server.rest_api.put_messages", new_callable=AsyncMock
    ) as mock_put:
        await add_memory(req, x_org_id="org-1", x_client_id="client-1")

    kwargs = mock_put.call_args.kwargs
    assert kwargs.get("direct_writes") is None
