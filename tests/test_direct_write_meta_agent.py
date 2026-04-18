"""Tests for the direct-write branch in Agent.step and _apply_direct_write dispatch."""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio(loop_scope="module")

TEST_ORG_ID = "direct-write-meta-org"
TEST_CLIENT_ID = "direct-write-meta-client"
TEST_USER_ID = "direct-write-meta-user"


def _unique(prefix: str = "src") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _provision_org_client_and_user():
    """Create the org, client, and user needed by all tests in this module."""
    from conftest import _create_client_and_key

    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    await _create_client_and_key(TEST_CLIENT_ID, TEST_ORG_ID, org_name="Direct Write Meta Org")

    user_mgr = UserManager()
    try:
        await user_mgr.get_user_by_id(TEST_USER_ID)
    except Exception:
        await user_mgr.create_user(
            PydanticUser(
                id=TEST_USER_ID,
                name="Direct Write Meta User",
                organization_id=TEST_ORG_ID,
                timezone="UTC",
            )
        )
    yield


async def _get_actor():
    from mirix.services.client_manager import ClientManager

    return await ClientManager().get_client_by_id(TEST_CLIENT_ID)


async def _get_user():
    from mirix.services.user_manager import UserManager

    return await UserManager().get_user_by_id(TEST_USER_ID)


# ---------------------------------------------------------------------------
# Unit tests for _apply_direct_write (dispatch + error path)
# ---------------------------------------------------------------------------


async def test_apply_direct_write_dispatches_to_registered_handler():
    """_apply_direct_write looks up handler in DIRECT_WRITE_HANDLERS and calls with payload kwargs."""
    from mirix.agent.agent import Agent
    from mirix.functions.function_sets import memory_tools

    agent = Agent.__new__(Agent)

    called = {}

    async def _fake_handler(ag, **kwargs):
        called["agent"] = ag
        called["kwargs"] = kwargs

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        await agent._apply_direct_write("episodic", {"event_type": "t", "summary": "s"})
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    assert called["agent"] is agent
    assert called["kwargs"] == {"event_type": "t", "summary": "s"}


async def test_apply_direct_write_unknown_memory_type_raises():
    """Unknown memory_type → ValueError."""
    from mirix.agent.agent import Agent

    agent = Agent.__new__(Agent)
    with pytest.raises(ValueError, match="No direct-write handler registered for memory_type: bogus"):
        await agent._apply_direct_write("bogus", {})


# ---------------------------------------------------------------------------
# Integration test for Agent.step direct-write branch:
# Uses a real Agent instance built via __new__ + manually wired attrs so we
# can exercise the branch path without booting the full meta-agent init.
# ---------------------------------------------------------------------------


def _make_meta_agent_stub(
    *,
    memory_source_id=None,
    direct_writes=None,
    filter_tags=None,
    use_cache=False,
    source_exists_and_complete=False,
    source_deduped=False,
):
    """Build an Agent-like object that Agent.step will run against.

    We stub out the heavy deps (message_manager retention, block_manager, etc.)
    and only wire up what the pre-LLM path actually reaches.
    """
    from mirix.agent.agent import Agent
    from mirix.schemas.agent import AgentType

    agent = Agent.__new__(Agent)

    # Agent state must be a proper meta_memory_agent
    agent_state = MagicMock()
    agent_state.id = "meta-1"
    agent_state.parent_id = None
    agent_state.name = "meta-agent"
    agent_state.is_type = lambda t: t == AgentType.meta_memory_agent
    agent_state.llm_config = MagicMock()
    agent_state.llm_config.model = "gpt-4"
    agent.agent_state = agent_state

    # Runtime fields
    agent.user_id = TEST_USER_ID
    agent.filter_tags = filter_tags or {"scope": "test"}
    agent.block_filter_tags = None
    agent.use_cache = use_cache
    agent.occurred_at = None
    agent.memory_source_id = memory_source_id
    agent.direct_writes = direct_writes
    agent.external_id = None
    agent.external_thread_id = None
    agent.source_type = None
    agent.source_system = None
    agent.source_metadata = None
    agent.source_summary = None
    agent.source_summary_source = None
    agent.summarize = False
    agent.source_messages = None
    agent._block_scopes = ["test"]
    agent._source_deduped = source_deduped

    # Managers — stubbed to record calls; message_manager retention read must return []
    agent.message_manager = MagicMock()
    agent.message_manager.get_messages_for_agent_user = AsyncMock(return_value=[])

    agent.memory_source_manager = MagicMock()
    if source_exists_and_complete:
        agent.memory_source_manager.get_by_id = AsyncMock(
            return_value=SimpleNamespace(processing_complete=True)
        )
    else:
        agent.memory_source_manager.get_by_id = AsyncMock(return_value=None)
    agent.memory_source_manager.mark_processing_complete = AsyncMock(return_value=None)

    # _persist_memory_source is a method we want to no-op unless dedup drives it
    async def _fake_persist(memory_source_id, input_messages):
        # Let tests control whether dedup fires via source_deduped flag
        if source_deduped:
            agent._source_deduped = True

    agent._persist_memory_source = _fake_persist
    return agent


async def test_step_direct_writes_skips_llm_and_calls_handler():
    """Meta-agent with direct_writes set runs handler, marks processing complete, returns step_count=0."""
    from mirix.functions.function_sets import memory_tools

    actor = await _get_actor()
    user = await _get_user()
    memory_source_id = _unique("src")

    agent = _make_meta_agent_stub(
        memory_source_id=memory_source_id,
        direct_writes=[{"memory_type": "episodic", "payload": {"event_type": "e", "summary": "s"}}],
    )

    handler_calls = []

    async def _fake_handler(ag, **kwargs):
        handler_calls.append(kwargs)

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        result = await agent.step(
            input_messages=[],
            actor=actor,
            user=user,
        )
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    # Handler was called with the payload
    assert handler_calls == [{"event_type": "e", "summary": "s"}]
    # Processing marked complete
    agent.memory_source_manager.mark_processing_complete.assert_awaited_once_with(memory_source_id)
    # step_count == 0 (direct-write branch returned early)
    assert result.step_count == 0


async def test_step_direct_writes_respects_source_dedup():
    """If source dedupes at _persist_memory_source, direct_writes branch does NOT run."""
    from mirix.functions.function_sets import memory_tools

    actor = await _get_actor()
    user = await _get_user()
    memory_source_id = _unique("src")

    agent = _make_meta_agent_stub(
        memory_source_id=memory_source_id,
        direct_writes=[{"memory_type": "episodic", "payload": {}}],
        source_deduped=True,
    )

    called = []

    async def _fake_handler(ag, **kwargs):
        called.append(kwargs)

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        result = await agent.step(
            input_messages=[],
            actor=actor,
            user=user,
        )
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    assert called == [], "Direct-write handler must NOT run when source deduped"
    # mark_processing_complete must NOT be called on dedup
    agent.memory_source_manager.mark_processing_complete.assert_not_called()
    assert result.step_count == 0


async def test_step_direct_writes_respects_processing_complete():
    """If source already has processing_complete=True, direct_writes branch does NOT run."""
    from mirix.functions.function_sets import memory_tools

    actor = await _get_actor()
    user = await _get_user()
    memory_source_id = _unique("src")

    agent = _make_meta_agent_stub(
        memory_source_id=memory_source_id,
        direct_writes=[{"memory_type": "episodic", "payload": {}}],
        source_exists_and_complete=True,
    )

    called = []

    async def _fake_handler(ag, **kwargs):
        called.append(kwargs)

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        result = await agent.step(
            input_messages=[],
            actor=actor,
            user=user,
        )
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    assert called == [], "Direct-write handler must NOT run when source already processed"
    agent.memory_source_manager.mark_processing_complete.assert_not_called()
    assert result.step_count == 0


async def test_step_direct_writes_unknown_type_raises_value_error():
    """Unknown memory_type in direct_writes → ValueError (does not mark processing complete)."""
    actor = await _get_actor()
    user = await _get_user()

    agent = _make_meta_agent_stub(
        memory_source_id=_unique("src"),
        direct_writes=[{"memory_type": "bogus", "payload": {}}],
    )

    with pytest.raises(ValueError, match="No direct-write handler registered"):
        await agent.step(input_messages=[], actor=actor, user=user)

    agent.memory_source_manager.mark_processing_complete.assert_not_called()
