"""
Tests for resilience of memory inserts when embedding generation fails.

Before the fix: if embedding generation raised (e.g. sustained OpenAI 429),
the exception propagated out of the manager's insert_* method before the DB
write happened, silently dropping the memory.

After the fix: embedding failures are caught, the memory is persisted with
NULL embedding columns, and a warning is logged. The row is still searchable
via keyword/text columns and can be backfilled later.

These tests use the real Postgres test database (via the conftest fixtures)
and monkey-patch `mirix.embeddings.embedding_model` to raise.
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
import yaml

pytestmark = [pytest.mark.asyncio(loop_scope="module")]

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.schemas.agent import CreateAgent
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.schemas.user import User as PydanticUser
from mirix.services.agent_manager import AgentManager
from mirix.services.client_manager import ClientManager
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.resource_memory_manager import ResourceMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.user_manager import UserManager

# =================================================================
# FIXTURES
# =================================================================


_ORG_ID = "test-org-embedding-fail"
_CLIENT_ID = "test-client-embedding-fail"
_USER_ID = "test-user-embedding-fail"
_AGENT_ID = "test-agent-embedding-fail"


@pytest_asyncio.fixture(scope="module")
async def test_actor():
    org_mgr = OrganizationManager()
    client_mgr = ClientManager()
    try:
        await org_mgr.get_organization_by_id(_ORG_ID)
    except Exception:
        await org_mgr.create_organization(PydanticOrganization(id=_ORG_ID, name="Embedding Fail Test Org"))
    try:
        return await client_mgr.get_client_by_id(_CLIENT_ID)
    except Exception:
        return await client_mgr.create_client(
            PydanticClient(
                id=_CLIENT_ID,
                organization_id=_ORG_ID,
                name="Embedding Fail Client",
                write_scope="test",
                read_scopes=["test"],
            )
        )


@pytest_asyncio.fixture(scope="module")
async def test_user(test_actor):
    user_mgr = UserManager()
    try:
        return await user_mgr.get_user_by_id(_USER_ID)
    except Exception:
        return await user_mgr.create_user(
            PydanticUser(id=_USER_ID, organization_id=_ORG_ID, name="Embedding Fail User", timezone="UTC")
        )


@pytest_asyncio.fixture(scope="module")
async def test_agent(test_actor):
    user_mgr = UserManager()
    await user_mgr.get_admin_user()
    agent_mgr = AgentManager()
    try:
        return await agent_mgr.get_agent_by_id(_AGENT_ID, actor=test_actor)
    except Exception:
        config_path = Path("mirix/configs/examples/mirix_gemini.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return await agent_mgr.create_agent(
            CreateAgent(
                name="Embedding Fail Test Agent",
                description="Agent to verify memory persistence when embedding fails",
                llm_config=LLMConfig(**config["llm_config"]),
                embedding_config=EmbeddingConfig(**config["embedding_config"]),
            ),
            actor=test_actor,
        )


@pytest.fixture
def failing_embedding_model(monkeypatch):
    """Patch embedding_model so calling get_text_embedding raises (simulating a 429)."""
    from unittest.mock import Mock

    async def _failing_get_text_embedding(text):
        raise RuntimeError("Simulated OpenAI 429 after retries exhausted")

    mock_embed_model = Mock()
    mock_embed_model.get_text_embedding = _failing_get_text_embedding

    async def factory(config):
        return mock_embed_model

    # Patch every place the manager modules imported `embedding_model` from.
    monkeypatch.setattr("mirix.embeddings.embedding_model", factory)
    monkeypatch.setattr("mirix.services.episodic_memory_manager.embedding_model", factory)
    monkeypatch.setattr("mirix.services.semantic_memory_manager.embedding_model", factory)
    monkeypatch.setattr("mirix.services.knowledge_vault_manager.embedding_model", factory)
    monkeypatch.setattr("mirix.services.procedural_memory_manager.embedding_model", factory)
    monkeypatch.setattr("mirix.services.resource_memory_manager.embedding_model", factory)
    return factory


# =================================================================
# EPISODIC
# =================================================================


async def test_episodic_insert_persists_memory_when_embedding_fails(
    test_actor, test_user, test_agent, failing_embedding_model
):
    manager = EpisodicMemoryManager()
    unique_summary = f"episodic-embed-fail-{uuid.uuid4()}"

    event = await manager.insert_event(
        actor=test_actor,
        agent_state=test_agent,
        agent_id=test_agent.id,
        timestamp=datetime.now(timezone.utc),
        event_type="user_message",
        event_actor="user",
        summary=unique_summary,
        details="details body",
        organization_id=test_actor.organization_id,
        user_id=test_user.id,
    )

    assert event is not None, "Memory must not be dropped when embedding fails"
    assert event.id, "Persisted memory must have a DB id"
    assert event.summary == unique_summary
    assert event.summary_embedding is None, "Embedding should be NULL when generation failed"
    assert event.details_embedding is None


# =================================================================
# SEMANTIC
# =================================================================


async def test_semantic_insert_persists_memory_when_embedding_fails(
    test_actor, test_user, test_agent, failing_embedding_model
):
    manager = SemanticMemoryManager()
    unique_name = f"semantic-embed-fail-{uuid.uuid4()}"

    item = await manager.insert_semantic_item(
        agent_state=test_agent,
        agent_id=test_agent.id,
        name=unique_name,
        summary="summary body",
        details="details body",
        source="test",
        organization_id=test_actor.organization_id,
        actor=test_actor,
        user_id=test_user.id,
    )

    assert item is not None, "Memory must not be dropped when embedding fails"
    assert item.id
    assert item.name == unique_name
    assert item.name_embedding is None
    assert item.summary_embedding is None
    assert item.details_embedding is None


# =================================================================
# KNOWLEDGE VAULT
# =================================================================


async def test_knowledge_vault_insert_persists_memory_when_embedding_fails(
    test_actor, test_user, test_agent, failing_embedding_model
):
    manager = KnowledgeVaultManager()
    unique_caption = f"kv-embed-fail-{uuid.uuid4()}"

    item = await manager.insert_knowledge(
        actor=test_actor,
        agent_state=test_agent,
        agent_id=test_agent.id,
        entry_type="credential",
        source="test",
        sensitivity="low",
        secret_value="secret",
        caption=unique_caption,
        organization_id=test_actor.organization_id,
        user_id=test_user.id,
    )

    assert item is not None, "Memory must not be dropped when embedding fails"
    assert item.id
    assert item.caption == unique_caption
    assert item.caption_embedding is None


# =================================================================
# PROCEDURAL
# =================================================================


async def test_procedural_insert_persists_memory_when_embedding_fails(
    test_actor, test_user, test_agent, failing_embedding_model
):
    manager = ProceduralMemoryManager()
    unique_summary = f"procedural-embed-fail-{uuid.uuid4()}"

    item = await manager.insert_procedure(
        agent_state=test_agent,
        agent_id=test_agent.id,
        entry_type="workflow",
        summary=unique_summary,
        steps=["step 1", "step 2"],
        actor=test_actor,
        organization_id=test_actor.organization_id,
        user_id=test_user.id,
    )

    assert item is not None, "Memory must not be dropped when embedding fails"
    assert item.id
    assert item.summary == unique_summary
    assert item.summary_embedding is None
    assert item.steps_embedding is None


# =================================================================
# RESOURCE
# =================================================================


async def test_resource_insert_persists_memory_when_embedding_fails(
    test_actor, test_user, test_agent, failing_embedding_model
):
    manager = ResourceMemoryManager()
    unique_title = f"resource-embed-fail-{uuid.uuid4()}"

    item = await manager.insert_resource(
        agent_state=test_agent,
        agent_id=test_agent.id,
        actor=test_actor,
        title=unique_title,
        summary="summary body",
        resource_type="doc",
        content="content body",
        organization_id=test_actor.organization_id,
        user_id=test_user.id,
    )

    assert item is not None, "Memory must not be dropped when embedding fails"
    assert item.id
    assert item.title == unique_title
    assert item.summary_embedding is None
