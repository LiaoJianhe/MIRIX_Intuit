"""Unit tests for AgentManager._bulk_update_children_configs (VEPAGE-1283).

Re-initializing a meta agent (force_update path, via update_meta_agent) applies
the SAME llm_config / embedding_config to every sub-agent. Previously this was a
per-child read+update+re-read N+1; now it is a single bulk write keyed by
parent:
  * provider mode  -> one mutate_using_named_query
    (agent_manager.update_children_configs_by_parent), NOT one update per child.
  * SQLAlchemy fallback -> one bulk UPDATE statement.

agents is an engine table (no domain events), so a mutation NQ is the right
tool — there is no per-record EventContext to preserve.
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.schemas.client import Client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.services.agent_manager import AgentManager


def _actor():
    return Client(id="client-1", organization_id="org-1", name="Test Client", status="active")


def _llm():
    return LLMConfig(model="gpt-4o-mini", model_endpoint_type="openai", context_window=8192)


def _emb():
    return EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
    )


@pytest.mark.asyncio
async def test_provider_mode_uses_single_mutation_nq():
    """Provider mode issues exactly ONE mutation NQ for all children (not N)."""
    am = AgentManager()
    fake_rp = AsyncMock()
    fake_rp.mutate_using_named_query = AsyncMock(return_value=6)

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        await am._bulk_update_children_configs(
            parent_id="meta-1", actor=_actor(), llm_config=_llm(), embedding_config=_emb()
        )

    assert fake_rp.mutate_using_named_query.await_count == 1
    args, kwargs = fake_rp.mutate_using_named_query.await_args
    assert args[0] == "agents"
    assert args[1] == "agent_manager.update_children_configs_by_parent"
    params = kwargs["params"]
    assert params["parentId"] == "meta-1"
    assert params["organizationId"] == "org-1"
    assert params["createdById"] == "client-1"
    # Configs serialized as JSON strings for the jsonb columns.
    assert isinstance(params["llmConfig"], str) and "gpt-4o-mini" in params["llmConfig"]
    assert isinstance(params["embeddingConfig"], str)


@pytest.mark.asyncio
async def test_noop_when_no_config_provided():
    """No write when neither config is supplied."""
    am = AgentManager()
    fake_rp = AsyncMock()
    fake_rp.mutate_using_named_query = AsyncMock()

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        await am._bulk_update_children_configs(parent_id="meta-1", actor=_actor())

    fake_rp.mutate_using_named_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_mode_issues_single_bulk_update():
    """With no relational provider, fall back to ONE bulk UPDATE statement."""
    am = AgentManager()

    captured = {}

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt):
            captured["stmt"] = stmt
            captured["calls"] = captured.get("calls", 0) + 1

        async def commit(self):
            captured["committed"] = True

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=None,
    ):
        with patch.object(am, "session_maker", return_value=_FakeSession()):
            await am._bulk_update_children_configs(
                parent_id="meta-1",
                actor=_actor(),
                llm_config=_llm(),
                embedding_config=_emb(),
            )

    # Exactly one statement executed and committed (a single bulk UPDATE, not N).
    assert captured.get("calls") == 1
    assert captured.get("committed") is True
