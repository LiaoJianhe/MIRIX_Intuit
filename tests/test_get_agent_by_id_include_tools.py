"""Unit tests for AgentManager.get_agent_by_id(include_tools=...) (VEPAGE-1310).

The save path (``add_memory`` in ``mirix/server/rest_api.py``) resolves the meta
agent purely to read ``meta_agent.id`` (and assert it exists) — it never touches
``meta_agent.tools``. Yet ``get_agent_by_id`` hardcoded
``include_relationships=["tools"]``, so every save eagerly hydrated the
meta-agent's tools: a fatter IPS-R payload / extra per-agent tool round-trip on
the relational path and dead Redis tool-pipeline work on the cache path — pure
overhead before the message is even queued.

``include_tools`` (mirroring ``list_agents``) lets the save path opt out. These
tests pin the contract that ``include_tools=False`` suppresses the tools
relationship request entirely while still returning a valid AgentState and
preserving the not-found contract. The end-to-end "no per-agent tool work on the
save path" behaviour is guarded by the ECMS full-stack test
``test_sdk_save_perf.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.orm.errors import NoResultFound
from mirix.schemas.client import Client
from mirix.services.agent_manager import AgentManager


def _make_actor():
    return Client(
        id="client-1",
        organization_id="org-1",
        name="Test Client",
        status="active",
    )


def _tools_less_row():
    """A realistic flat agent row as the provider produces it when tools are
    NOT hydrated: no ``tools`` key at all."""
    return {
        "id": "agent-1",
        "name": "meta_memory_agent",
        "agent_type": "meta_memory_agent",
        "system": "you are a meta agent",
        "llm_config": {
            "model": "gpt-4o-mini",
            "model_endpoint_type": "openai",
            "context_window": 8192,
        },
        "embedding_config": {
            "embedding_endpoint_type": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
        },
        "organization_id": "org-1",
        "created_by_id": "client-1",
    }


async def _call_get_agent_by_id(rows, **kwargs):
    """Invoke get_agent_by_id against a recording fake relational provider (no
    cache, so we land on the IPS-R path) and return (agents_or_exc, provider).

    Returns the fake provider so callers can assert on the
    ``include_relationships`` kwarg the named query received.
    """
    am = AgentManager()
    fake_rp = AsyncMock()
    fake_rp.find_using_named_query = AsyncMock(return_value=rows)

    # No cache provider / redis client -> skip the cache-hit branch and go
    # straight to the relational provider path.
    with patch(
        "mirix.database.cache_provider.get_cache_provider", return_value=None
    ), patch(
        "mirix.database.redis_client.get_redis_client", return_value=None
    ), patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        agent = await am.get_agent_by_id(
            agent_id="agent-1", actor=_make_actor(), **kwargs
        )

    return agent, fake_rp


@pytest.mark.asyncio
async def test_include_tools_false_skips_tool_relationship():
    """include_tools=False must NOT request the tools relationship, so the
    relational provider does not fire a per-agent tool hydration round-trip."""
    _, fake_rp = await _call_get_agent_by_id([_tools_less_row()], include_tools=False)
    assert fake_rp.find_using_named_query.await_count == 1
    kwargs = fake_rp.find_using_named_query.await_args.kwargs
    assert kwargs.get("include_relationships") is None


@pytest.mark.asyncio
async def test_include_tools_false_yields_valid_agent_state_with_empty_tools():
    """With include_tools=False the provider returns a row WITHOUT a ``tools``
    key, but AgentState.tools is a required field. get_agent_by_id must still
    produce a valid AgentState (tools defaulted to []), not raise a
    ValidationError."""
    agent, _ = await _call_get_agent_by_id([_tools_less_row()], include_tools=False)
    assert agent.id == "agent-1"
    assert agent.tools == []


@pytest.mark.asyncio
async def test_include_tools_default_hydrates_tools():
    """The default preserves existing behaviour for callers that rely on
    populated ``tools`` (e.g. agent construction)."""
    _, fake_rp = await _call_get_agent_by_id([_tools_less_row()])
    assert fake_rp.find_using_named_query.await_count == 1
    kwargs = fake_rp.find_using_named_query.await_args.kwargs
    assert kwargs.get("include_relationships") == ["tools"]


@pytest.mark.asyncio
async def test_include_tools_true_explicit_hydrates_tools():
    """Passing include_tools=True explicitly matches the default."""
    _, fake_rp = await _call_get_agent_by_id([_tools_less_row()], include_tools=True)
    kwargs = fake_rp.find_using_named_query.await_args.kwargs
    assert kwargs.get("include_relationships") == ["tools"]


@pytest.mark.asyncio
async def test_missing_agent_raises_regardless_of_include_tools():
    """The not-found contract is unchanged: an empty provider result still
    raises NoResultFound whether or not tools are requested."""
    for include_tools in (True, False):
        with pytest.raises(NoResultFound):
            await _call_get_agent_by_id([], include_tools=include_tools)
