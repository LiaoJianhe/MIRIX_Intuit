"""Unit + DB-level tests for direct_write_episodic in direct_write_handlers.py.

The handler is a thin shim over episodic_memory_insert. These tests verify:
  1. The DIRECT_WRITE_HANDLERS registry exposes it.
  2. The handler reshapes payload kwargs into the list-of-items the tool expects
     and delegates correctly.
  3. When occurred_at is missing from both payload and agent, the handler fills
     in now() so the tool's timestamp parsing doesn't crash.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_direct_write_handlers_registry_has_episodic():
    from mirix.functions.direct_write_handlers import (
        DIRECT_WRITE_HANDLERS,
        direct_write_episodic,
    )

    assert DIRECT_WRITE_HANDLERS["episodic"] is direct_write_episodic


async def test_direct_write_episodic_delegates_to_tool():
    from mirix.functions.direct_write_handlers import direct_write_episodic

    agent = SimpleNamespace(occurred_at=None)

    with patch(
        "mirix.functions.direct_write_handlers.episodic_memory_insert",
        new_callable=AsyncMock,
    ) as mock_tool:
        await direct_write_episodic(
            agent,
            event_type="engagement_created",
            summary="s",
            details="d",
            event_actor="system",
            occurred_at="2026-04-17T10:00:00Z",
        )

    mock_tool.assert_awaited_once()
    call_args = mock_tool.call_args
    assert call_args.args[0] is agent
    items = call_args.kwargs["items"]
    assert len(items) == 1
    item = items[0]
    assert item["event_type"] == "engagement_created"
    assert item["summary"] == "s"
    assert item["details"] == "d"
    assert item["actor"] == "system"
    assert item["occurred_at"] == "2026-04-17T10:00:00Z"


async def test_direct_write_episodic_defaults_occurred_at_to_now():
    """When neither payload nor agent has occurred_at, fill in now() so the
    tool's `datetime.fromisoformat(...)` doesn't crash.
    """
    from mirix.functions.direct_write_handlers import direct_write_episodic

    agent = SimpleNamespace(occurred_at=None)

    with patch(
        "mirix.functions.direct_write_handlers.episodic_memory_insert",
        new_callable=AsyncMock,
    ) as mock_tool:
        await direct_write_episodic(
            agent,
            event_type="e",
            summary="s",
            details="d",
            event_actor="user",
            occurred_at=None,
        )

    items = mock_tool.call_args.kwargs["items"]
    assert items[0]["occurred_at"] is not None
    datetime.fromisoformat(items[0]["occurred_at"])


async def test_direct_write_episodic_prefers_payload_over_agent():
    """Payload occurred_at wins over agent.occurred_at."""
    from mirix.functions.direct_write_handlers import direct_write_episodic

    agent = SimpleNamespace(occurred_at="2020-01-01T00:00:00Z")

    with patch(
        "mirix.functions.direct_write_handlers.episodic_memory_insert",
        new_callable=AsyncMock,
    ) as mock_tool:
        await direct_write_episodic(
            agent,
            event_type="e",
            summary="s",
            details="d",
            event_actor="user",
            occurred_at="2026-06-01T12:00:00Z",
        )

    items = mock_tool.call_args.kwargs["items"]
    assert items[0]["occurred_at"] == "2026-06-01T12:00:00Z"


async def test_direct_write_episodic_falls_back_to_agent_occurred_at():
    """When payload.occurred_at is None, agent.occurred_at is used."""
    from mirix.functions.direct_write_handlers import direct_write_episodic

    agent = SimpleNamespace(occurred_at="2026-03-15T09:30:00Z")

    with patch(
        "mirix.functions.direct_write_handlers.episodic_memory_insert",
        new_callable=AsyncMock,
    ) as mock_tool:
        await direct_write_episodic(
            agent,
            event_type="e",
            summary="s",
            details="d",
            event_actor="user",
            occurred_at=None,
        )

    items = mock_tool.call_args.kwargs["items"]
    assert items[0]["occurred_at"] == "2026-03-15T09:30:00Z"
