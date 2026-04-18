"""Unit + DB-level tests for direct_write_episodic in direct_write_handlers.py.

Exercises both the meta_memory_agent insert_event path and the non-meta
create_episodic_memory path. The citation write hits the real DB via
_write_citation → MemoryCitationManager.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio(loop_scope="module")

TEST_ORG_ID = "direct-write-handler-org"
TEST_CLIENT_ID = "direct-write-handler-client"
TEST_USER_ID = "direct-write-handler-user"


def _unique(prefix: str = "src") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _provision_org_client_and_user():
    """Create the org, client, and user needed by all tests in this module."""
    from conftest import _create_client_and_key

    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    await _create_client_and_key(
        TEST_CLIENT_ID, TEST_ORG_ID, org_name="Direct Write Handler Org"
    )

    user_mgr = UserManager()
    try:
        await user_mgr.get_user_by_id(TEST_USER_ID)
    except Exception:
        await user_mgr.create_user(
            PydanticUser(
                id=TEST_USER_ID,
                name="Direct Write Handler User",
                organization_id=TEST_ORG_ID,
                timezone="UTC",
            )
        )
    yield


async def _get_actor():
    from mirix.services.client_manager import ClientManager

    return await ClientManager().get_client_by_id(TEST_CLIENT_ID)


async def _create_source(source_id=None):
    from mirix.services.memory_source_manager import MemorySourceManager

    sid = source_id or _unique("src")
    actor = await _get_actor()
    await MemorySourceManager().create(
        memory_source_id=sid,
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_type="conversation",
        use_cache=False,
    )
    return sid


def _build_agent_stub(
    *,
    actor,
    memory_source_id,
    is_meta: bool,
    insert_return_id: str,
):
    """Build an Agent-like object that direct_write_episodic expects.

    Uses real actor/user_id but a mocked episodic_memory_manager so we can
    assert on the insert call and make it return a deterministic id.
    """
    from mirix.schemas.agent import AgentType

    # Mock the episodic_memory_manager
    ep_mgr = MagicMock()
    ep_mgr.insert_event = AsyncMock(return_value=SimpleNamespace(id=insert_return_id))
    ep_mgr.create_episodic_memory = AsyncMock(return_value=SimpleNamespace(id=insert_return_id))

    agent_state = MagicMock()
    agent_state.id = "meta-1" if is_meta else "sub-1"
    agent_state.parent_id = None if is_meta else "meta-1"
    agent_state.name = "agent"
    if is_meta:
        agent_state.is_type = lambda t: t == AgentType.meta_memory_agent
    else:
        agent_state.is_type = lambda t: False

    return SimpleNamespace(
        actor=actor,
        user=SimpleNamespace(id=TEST_USER_ID, organization_id=TEST_ORG_ID),
        user_id=TEST_USER_ID,
        agent_state=agent_state,
        filter_tags={"scope": actor.write_scope or "test"},
        use_cache=False,
        occurred_at=None,
        memory_source_id=memory_source_id,
        external_thread_id=None,
        episodic_memory_manager=ep_mgr,
    )


# ---------------------------------------------------------------------------
# DIRECT_WRITE_HANDLERS registry is correctly populated
# ---------------------------------------------------------------------------


async def test_direct_write_handlers_registry_has_episodic():
    from mirix.functions.direct_write_handlers import (
        DIRECT_WRITE_HANDLERS,
        direct_write_episodic,
    )

    assert DIRECT_WRITE_HANDLERS["episodic"] is direct_write_episodic


# ---------------------------------------------------------------------------
# Meta-agent path (uses insert_event)
# ---------------------------------------------------------------------------


async def test_direct_write_episodic_meta_agent_uses_insert_event():
    """meta_memory_agent: calls episodic_memory_manager.insert_event with correct args."""
    from mirix.functions.direct_write_handlers import direct_write_episodic
    from mirix.services.memory_citation_manager import MemoryCitationManager

    source_id = await _create_source()
    memory_id = _unique("ep")
    actor = await _get_actor()

    agent = _build_agent_stub(
        actor=actor,
        memory_source_id=source_id,
        is_meta=True,
        insert_return_id=memory_id,
    )

    ts = "2026-04-17T10:00:00Z"
    await direct_write_episodic(
        agent,
        event_type="engagement_created",
        summary="a summary",
        details="some details",
        event_actor="system",
        occurred_at=ts,
    )

    # insert_event called (meta-agent path)
    agent.episodic_memory_manager.insert_event.assert_awaited_once()
    agent.episodic_memory_manager.create_episodic_memory.assert_not_called()

    kwargs = agent.episodic_memory_manager.insert_event.call_args.kwargs
    assert kwargs["event_type"] == "engagement_created"
    assert kwargs["summary"] == "a summary"
    assert kwargs["details"] == "some details"
    assert kwargs["event_actor"] == "system"
    assert kwargs["organization_id"] == TEST_ORG_ID
    assert kwargs["client_id"] == actor.id
    assert kwargs["user_id"] == TEST_USER_ID
    # timestamp was parsed + made naive-UTC for timestamptz
    assert kwargs["timestamp"] == datetime(2026, 4, 17, 10, 0, 0)

    # Citation was written to the real DB
    exists = await MemoryCitationManager().check_exists(
        memory_source_id=source_id,
        memory_type="episodic",
        memory_id=memory_id,
        use_cache=False,
    )
    assert exists is True


# ---------------------------------------------------------------------------
# Non-meta path (uses create_episodic_memory)
# ---------------------------------------------------------------------------


async def test_direct_write_episodic_non_meta_uses_create_episodic_memory():
    """Non-meta agent: calls episodic_memory_manager.create_episodic_memory instead."""
    from mirix.functions.direct_write_handlers import direct_write_episodic
    from mirix.services.memory_citation_manager import MemoryCitationManager

    source_id = await _create_source()
    memory_id = _unique("ep")
    actor = await _get_actor()

    agent = _build_agent_stub(
        actor=actor,
        memory_source_id=source_id,
        is_meta=False,
        insert_return_id=memory_id,
    )

    await direct_write_episodic(
        agent,
        event_type="e",
        summary="s",
        details="d",
        event_actor="user",
        occurred_at=None,  # handler should fall back to now()
    )

    agent.episodic_memory_manager.create_episodic_memory.assert_awaited_once()
    agent.episodic_memory_manager.insert_event.assert_not_called()

    # Citation should still be written
    exists = await MemoryCitationManager().check_exists(
        memory_source_id=source_id,
        memory_type="episodic",
        memory_id=memory_id,
        use_cache=False,
    )
    assert exists is True


async def test_direct_write_episodic_preserves_existing_scope_in_filter_tags():
    """If agent.filter_tags already has scope, handler does NOT overwrite it."""
    from mirix.functions.direct_write_handlers import direct_write_episodic

    source_id = await _create_source()
    actor = await _get_actor()

    agent = _build_agent_stub(
        actor=actor,
        memory_source_id=source_id,
        is_meta=True,
        insert_return_id=_unique("ep"),
    )
    agent.filter_tags = {"scope": "already-set-scope", "other": "tag"}

    await direct_write_episodic(
        agent,
        event_type="e",
        summary="s",
        details="d",
        event_actor="system",
    )

    kwargs = agent.episodic_memory_manager.insert_event.call_args.kwargs
    # Scope preserved, other tag preserved
    assert kwargs["filter_tags"]["scope"] == "already-set-scope"
    assert kwargs["filter_tags"]["other"] == "tag"
