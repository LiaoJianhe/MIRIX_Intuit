"""
Tests that memory managers delegate reads/writes to IPS relational and search providers
when registered, and use the ORM/cache path when providers are absent.

Run:
    pytest tests/test_manager_delegation.py -v
"""

import sys
import types

# episodic/semantic managers import embedding_model at module load; that pulls llm/observability
# and optional Langfuse (pydantic v1), which breaks on Python 3.14+. Stub for unit collection.
if "mirix.embeddings" not in sys.modules:
    from unittest.mock import MagicMock as _MagicMock

    _emb_mod = types.ModuleType("mirix.embeddings")

    async def _stub_embedding_model(*_args, **_kwargs):
        return _MagicMock()

    _emb_mod.embedding_model = _stub_embedding_model
    sys.modules["mirix.embeddings"] = _emb_mod

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.database.call_context import CALL_ORIGIN_CLIENT_API, CALL_ORIGIN_ENGINE
from mirix.orm.errors import NoResultFound
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.block import Block, BlockUpdate
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.episodic_memory import EpisodicEvent as PydanticEpisodicEvent
from mirix.schemas.enums import ToolType
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.schemas.semantic_memory import SemanticMemoryItem as PydanticSemanticMemoryItem
from mirix.schemas.semantic_memory import SemanticMemoryItemUpdate
from mirix.schemas.tool import Tool as PydanticTool
from mirix.schemas.user import User as PydanticUser
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.services.agent_manager import AgentManager
from mirix.services.block_manager import BlockManager
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.tool_manager import ToolManager
from mirix.services.user_manager import UserManager


def _episodic_mgr() -> EpisodicMemoryManager:
    m = EpisodicMemoryManager.__new__(EpisodicMemoryManager)
    m.session_maker = MagicMock()
    return m


def _semantic_mgr() -> SemanticMemoryManager:
    m = SemanticMemoryManager.__new__(SemanticMemoryManager)
    m.session_maker = MagicMock()
    return m


def _block_mgr() -> BlockManager:
    m = BlockManager.__new__(BlockManager)
    m.session_maker = MagicMock()
    return m


def _naive_utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _mock_user() -> MagicMock:
    u = MagicMock(spec=PydanticUser)
    u.id = "user-1"
    u.organization_id = "org-1"
    return u


def _mock_actor() -> MagicMock:
    c = MagicMock(spec=PydanticClient)
    c.id = "client-1"
    c.organization_id = "org-1"
    c.name = "test-client"
    c.write_scope = "scope-a"
    return c


def _minimal_agent_state() -> AgentState:
    return AgentState.model_construct(
        id="agent-a1b2c3d4",
        name="delegation-test-agent",
        system="sys",
        agent_type=AgentType.episodic_memory_agent,
        llm_config=LLMConfig(model="m", model_endpoint_type="openai", context_window=8000),
        embedding_config=EmbeddingConfig.default_config("text-embedding-3-small"),
        organization_id="org-1",
        tools=[],
    )


def _session_maker_cm(mock_session: MagicMock):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


def _episodic_row_dict() -> dict:
    ts = _naive_utc_now()
    return {
        "id": "ep-123",
        "event_type": "test",
        "summary": "sum",
        "details": "det",
        "actor": "actor-1",
        "user_id": "user-1",
        "organization_id": "org-1",
        "client_id": "client-1",
        "occurred_at": ts,
        "created_at": ts,
        "last_modify": {"timestamp": ts.isoformat(), "operation": "created"},
        "filter_tags": None,
        "agent_id": None,
        "summary_embedding": None,
        "details_embedding": None,
        "embedding_config": None,
        "updated_at": None,
    }


def _semantic_row_dict() -> dict:
    ts = _naive_utc_now()
    return {
        "id": "sem-123",
        "name": "n",
        "summary": "s",
        "details": "d",
        "source": "src",
        "user_id": "user-1",
        "organization_id": "org-1",
        "client_id": "client-1",
        "created_at": ts,
        "last_modify": {"timestamp": ts.isoformat(), "operation": "created"},
        "filter_tags": None,
        "agent_id": None,
        "name_embedding": None,
        "summary_embedding": None,
        "details_embedding": None,
        "embedding_config": None,
        "updated_at": None,
    }


def _block_row_dict(bid: str = "block-c0ffee00") -> dict:
    return {
        "id": bid,
        "label": "human",
        "value": "hello",
        "limit": 8000,
        "user_id": "user-1",
        "organization_id": "org-1",
        "filter_tags": {"scope": "scope-a"},
        "created_by_id": "user-1",
        "last_updated_by_id": "client-1",
    }


def _organization_row_dict(oid: str = "org-1") -> dict:
    ts = _naive_utc_now()
    return {"id": oid, "name": "test-org", "created_at": ts}


def _user_row_dict(uid: str = "user-1") -> dict:
    ts = _naive_utc_now()
    return {
        "id": uid,
        "name": "test-user",
        "status": "active",
        "timezone": "UTC (UTC+00:00)",
        "organization_id": "org-1",
        "is_admin": False,
        "created_at": ts,
        "updated_at": ts,
        "is_deleted": False,
    }


def _tool_source_with_docstring() -> str:
    return '''def test_delegation_tool():
    """Minimal tool for manager delegation tests."""
    pass
'''


def _tool_row_dict(tid: str = "tool-c0ffee00") -> dict:
    return {
        "id": tid,
        "name": "test_tool",
        "tool_type": ToolType.CUSTOM,
        "description": "d",
        "source_type": None,
        "organization_id": "org-1",
        "tags": [],
        "source_code": _tool_source_with_docstring(),
        "json_schema": {"type": "object", "description": "d"},
        "return_char_limit": 10000,
        "created_by_id": "client-1",
        "last_updated_by_id": None,
    }


def _agent_row_dict(aid: str = "agent-1") -> dict:
    emb = EmbeddingConfig.default_config("text-embedding-3-small")
    return {
        "id": aid,
        "name": "test-agent",
        "system": "sys",
        "agent_type": AgentType.meta_memory_agent,
        "created_by_id": "client-1",
        "organization_id": "org-1",
        "llm_config": {"model": "gpt-4", "model_endpoint_type": "openai", "context_window": 8000},
        "embedding_config": emb.model_dump(),
        "tools": [],
        "parent_id": None,
        "tool_rules": None,
        "mcp_tools": [],
        "description": None,
        "children": None,
    }


def _org_mgr() -> OrganizationManager:
    m = OrganizationManager.__new__(OrganizationManager)
    m.session_maker = MagicMock()
    return m


def _user_mgr() -> UserManager:
    m = UserManager.__new__(UserManager)
    m.session_maker = MagicMock()
    return m


def _tool_mgr() -> ToolManager:
    m = ToolManager.__new__(ToolManager)
    m.session_maker = MagicMock()
    return m


def _agent_mgr() -> AgentManager:
    m = AgentManager.__new__(AgentManager)
    m.session_maker = MagicMock()
    return m


class TestEpisodicMemoryManagerDelegation:
    @pytest.mark.asyncio
    async def test_get_by_id_delegates_to_relational_provider(self):
        row = _episodic_row_dict()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            out = await mgr.get_episodic_memory_by_id("ep-123", _mock_user())
            mock_provider.read.assert_awaited_once_with("episodic_memory", "ep-123")
            assert out.id == "ep-123"
            assert out.summary == "sum"

    @pytest.mark.asyncio
    async def test_get_by_id_raises_when_provider_returns_none(self):
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            with pytest.raises(NoResultFound):
                await mgr.get_episodic_memory_by_id("missing", _mock_user())

    @pytest.mark.asyncio
    async def test_get_by_id_falls_back_when_no_relational_provider(self):
        pyd = PydanticEpisodicEvent(**_episodic_row_dict())
        mock_orm = MagicMock()
        mock_orm.to_pydantic = MagicMock(return_value=pyd)
        mock_session = MagicMock()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=None):
            with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
                with patch(
                    "mirix.services.episodic_memory_manager.EpisodicEvent.read",
                    new_callable=AsyncMock,
                    return_value=mock_orm,
                ):
                    mgr = _episodic_mgr()
                    mgr.session_maker = _session_maker_cm(mock_session)
                    out = await mgr.get_episodic_memory_by_id("ep-123", _mock_user())
                    assert out.id == "ep-123"

    @pytest.mark.asyncio
    async def test_create_episodic_memory_delegates_to_provider(self):
        row = _episodic_row_dict()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)

        ev = PydanticEpisodicEvent(**{**row, "id": "ep-new"})
        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            out = await mgr.create_episodic_memory(ev, _mock_actor(), client_id="client-1", user_id="user-1")
            mock_provider.create.assert_awaited_once()
            call_kw = mock_provider.create.await_args
            assert call_kw[0][0] == "episodic_memory"
            assert call_kw[0][1]["user_id"] == "user-1"
            assert out.id == "ep-123"

    @pytest.mark.asyncio
    async def test_delete_event_by_id_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.delete = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            await mgr.delete_event_by_id("ep-123", _mock_actor())
            mock_provider.delete.assert_awaited_once_with("episodic_memory", "ep-123")

    @pytest.mark.asyncio
    async def test_insert_event_delegates_to_provider(self):
        row = _episodic_row_dict()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)
        ts = _naive_utc_now()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            await mgr.insert_event(
                actor=_mock_actor(),
                agent_state=_minimal_agent_state(),
                agent_id="ag-1",
                event_type="test",
                timestamp=ts,
                event_actor="bot",
                details="d",
                summary="s",
                organization_id="org-1",
                client_id="client-1",
                user_id="user-1",
            )
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "episodic_memory"

    @pytest.mark.asyncio
    async def test_list_episodic_memory_delegates_search_client_api(self):
        row = _episodic_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_CLIENT_API):
                mgr = _episodic_mgr()
                out = await mgr.list_episodic_memory(
                    _minimal_agent_state(),
                    _mock_user(),
                    query="q",
                    search_method="bm25",
                    limit=10,
                )
                mock_search.search.assert_awaited_once()
                assert mock_search.search.await_args[0][0] == "episodic_memory"
                assert len(out) == 1
                assert out[0].id == "ep-123"

    @pytest.mark.asyncio
    async def test_list_episodic_memory_delegates_hybrid_engine_origin(self):
        row = _episodic_row_dict()
        mock_search = MagicMock()

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_ENGINE):
                with patch(
                    "mirix.services.hybrid_search_helper.hybrid_search",
                    new_callable=AsyncMock,
                    return_value=[row],
                ):
                    mgr = _episodic_mgr()
                    out = await mgr.list_episodic_memory(
                        _minimal_agent_state(),
                        _mock_user(),
                        query="q",
                        limit=5,
                    )
                    assert len(out) == 1

    @pytest.mark.asyncio
    async def test_get_total_number_delegates_count_client_api(self):
        mock_search = MagicMock()
        mock_search.count = AsyncMock(return_value=42)

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_CLIENT_API):
                mgr = _episodic_mgr()
                n = await mgr.get_total_number_of_items(_mock_user())
                mock_search.count.assert_awaited_once_with(
                    "episodic_memory",
                    user_id="user-1",
                    organization_id="org-1",
                )
                assert n == 42

    @pytest.mark.asyncio
    async def test_get_total_number_delegates_hybrid_count_engine(self):
        mock_search = MagicMock()

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_ENGINE):
                with patch(
                    "mirix.services.hybrid_search_helper.hybrid_count",
                    new_callable=AsyncMock,
                    return_value=99,
                ):
                    mgr = _episodic_mgr()
                    n = await mgr.get_total_number_of_items(_mock_user())
                    assert n == 99


class TestSemanticMemoryManagerDelegation:
    @pytest.mark.asyncio
    async def test_get_semantic_item_by_id_delegates(self):
        row = _semantic_row_dict()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            out = await mgr.get_semantic_item_by_id("sem-123", _mock_user(), timezone_str="UTC")
            mock_provider.read.assert_awaited_once_with("semantic_memory", "sem-123")
            assert out.id == "sem-123"

    @pytest.mark.asyncio
    async def test_get_semantic_item_by_id_raises_when_none(self):
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            with pytest.raises(NoResultFound):
                await mgr.get_semantic_item_by_id("x", _mock_user(), timezone_str="UTC")

    @pytest.mark.asyncio
    async def test_get_by_id_falls_back_when_no_relational_provider(self):
        row = _semantic_row_dict()
        pyd = PydanticSemanticMemoryItem(**row)
        mock_orm = MagicMock()
        mock_orm.to_pydantic = MagicMock(return_value=pyd)
        mock_session = MagicMock()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=None):
            with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
                with patch(
                    "mirix.services.semantic_memory_manager.SemanticMemoryItem.read",
                    new_callable=AsyncMock,
                    return_value=mock_orm,
                ):
                    mgr = _semantic_mgr()
                    mgr.session_maker = _session_maker_cm(mock_session)
                    out = await mgr.get_semantic_item_by_id("sem-123", _mock_user(), timezone_str="UTC")
                    assert out.name == "n"

    @pytest.mark.asyncio
    async def test_create_item_delegates(self):
        row = _semantic_row_dict()
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)
        item = PydanticSemanticMemoryItem(**row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            out = await mgr.create_item(item, _mock_actor(), client_id="client-1", user_id="user-1")
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "semantic_memory"
            assert out.id == "sem-123"

    @pytest.mark.asyncio
    async def test_update_item_delegates(self):
        row = _semantic_row_dict()
        row["summary"] = "updated"
        mock_provider = MagicMock()
        mock_provider.update = AsyncMock(return_value=row)
        upd = SemanticMemoryItemUpdate(id="sem-123", summary="updated")

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            out = await mgr.update_item(upd, _mock_user(), _mock_actor())
            mock_provider.update.assert_awaited_once_with("semantic_memory", "sem-123", {"summary": "updated"})
            assert out.summary == "updated"

    @pytest.mark.asyncio
    async def test_delete_semantic_item_by_id_delegates(self):
        mock_provider = MagicMock()
        mock_provider.delete = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _semantic_mgr()
            await mgr.delete_semantic_item_by_id("sem-123", _mock_actor())
            mock_provider.delete.assert_awaited_once_with("semantic_memory", "sem-123")

    @pytest.mark.asyncio
    async def test_list_semantic_items_search_client_api(self):
        row = _semantic_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_CLIENT_API):
                mgr = _semantic_mgr()
                out = await mgr.list_semantic_items(
                    _minimal_agent_state(),
                    _mock_user(),
                    query="q",
                    limit=3,
                )
                mock_search.search.assert_awaited_once()
                args, kwargs = mock_search.search.await_args
                assert args[0] == "semantic_memory"
                assert len(out) == 1

    @pytest.mark.asyncio
    async def test_list_semantic_items_hybrid_engine(self):
        row = _semantic_row_dict()
        mock_search = MagicMock()

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_ENGINE):
                with patch(
                    "mirix.services.hybrid_search_helper.hybrid_search",
                    new_callable=AsyncMock,
                    return_value=[row],
                ):
                    mgr = _semantic_mgr()
                    out = await mgr.list_semantic_items(_minimal_agent_state(), _mock_user())
                    assert len(out) == 1

    @pytest.mark.asyncio
    async def test_get_total_number_count_client_api(self):
        mock_search = MagicMock()
        mock_search.count = AsyncMock(return_value=7)

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_CLIENT_API):
                mgr = _semantic_mgr()
                n = await mgr.get_total_number_of_items(_mock_user())
                mock_search.count.assert_awaited_once_with(
                    "semantic_memory",
                    user_id="user-1",
                    organization_id="org-1",
                )
                assert n == 7

    @pytest.mark.asyncio
    async def test_get_total_number_hybrid_count_engine(self):
        mock_search = MagicMock()

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_ENGINE):
                with patch(
                    "mirix.services.hybrid_search_helper.hybrid_count",
                    new_callable=AsyncMock,
                    return_value=11,
                ):
                    mgr = _semantic_mgr()
                    assert await mgr.get_total_number_of_items(_mock_user()) == 11


class TestBlockManagerDelegation:
    @pytest.mark.asyncio
    async def test_get_block_by_id_delegates(self):
        row = _block_row_dict()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _block_mgr()
            out = await mgr.get_block_by_id("block-c0ffee00", user=_mock_user())
            mock_provider.read.assert_awaited_once_with("block", "block-c0ffee00")
            assert out.id == "block-c0ffee00"

    @pytest.mark.asyncio
    async def test_get_block_by_id_returns_none_when_provider_returns_none(self):
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _block_mgr()
            assert await mgr.get_block_by_id("missing") is None

    @pytest.mark.asyncio
    async def test_create_or_update_block_creates_via_provider(self):
        row = _block_row_dict("block-deadb33f")
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=None)
        mock_provider.create = AsyncMock(return_value=row)

        blk = Block(
            id="block-deadb33f",
            label="human",
            value="v",
            limit=8000,
            user_id="user-1",
            organization_id="org-1",
            filter_tags={"scope": "scope-a"},
            created_by_id="user-1",
            last_updated_by_id="client-1",
        )

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _block_mgr()
            out = await mgr.create_or_update_block(blk, _mock_actor(), user=_mock_user())
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "block"
            assert out.id == "block-deadb33f"

    @pytest.mark.asyncio
    async def test_update_block_delegates(self):
        row = _block_row_dict()
        row["value"] = "patched"
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=_block_row_dict())
        mock_provider.update = AsyncMock(return_value=row)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _block_mgr()
            out = await mgr.update_block(
                "block-c0ffee00",
                BlockUpdate(value="patched"),
                _mock_actor(),
                user=_mock_user(),
            )
            mock_provider.update.assert_awaited_once()
            assert mock_provider.update.await_args[0][0] == "block"
            assert mock_provider.update.await_args[0][1] == "block-c0ffee00"
            assert out.value == "patched"

    @pytest.mark.asyncio
    async def test_delete_block_delegates(self):
        existing = _block_row_dict()
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=existing)
        mock_provider.delete = AsyncMock(return_value=None)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            with patch.object(BlockManager, "_invalidate_block_cache", new_callable=AsyncMock):
                mgr = _block_mgr()
                out = await mgr.delete_block("block-c0ffee00", _mock_actor())
                mock_provider.delete.assert_awaited_once_with("block", "block-c0ffee00")
                assert out.id == "block-c0ffee00"

    @pytest.mark.asyncio
    async def test_get_blocks_search_client_api(self):
        row = _block_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_CLIENT_API):
                mgr = _block_mgr()
                out = await mgr.get_blocks(
                    user=_mock_user(),
                    any_scopes=["scope-a"],
                    auto_create_from_default=False,
                )
                mock_search.search.assert_awaited_once()
                args, _kwargs = mock_search.search.await_args
                assert args[0] == "block"
                assert len(out) == 1

    @pytest.mark.asyncio
    async def test_get_blocks_hybrid_engine_origin(self):
        row = _block_row_dict()
        mock_search = MagicMock()

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_ENGINE):
                with patch(
                    "mirix.services.hybrid_search_helper.hybrid_search",
                    new_callable=AsyncMock,
                    return_value=[row],
                ):
                    mgr = _block_mgr()
                    out = await mgr.get_blocks(
                        user=_mock_user(),
                        any_scopes=["scope-a"],
                        auto_create_from_default=False,
                    )
                    assert len(out) == 1


@pytest.mark.asyncio
class TestOrganizationManagerDelegation:
    async def test_get_org_by_id_delegates_to_provider(self):
        row = _organization_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _org_mgr()
                out = await mgr.get_organization_by_id("org-1")
                mock_provider.find_using_named_query.assert_awaited_once()
                args, kwargs = mock_provider.find_using_named_query.call_args
                assert args[0] == "organizations"
                assert args[1] == "organization_manager.get_organization_by_id"
                assert kwargs["params"] == {"id": "org-1"}
                assert isinstance(out, PydanticOrganization)
                assert out.id == "org-1"
                assert out.name == "test-org"

    async def test_create_organization_delegates(self):
        row = _organization_row_dict("org-new")
        mock_provider = MagicMock()
        # _create_organization reads the row by id first; provide None to force create.
        mock_provider.find_using_named_query = AsyncMock(return_value=[])
        mock_provider.read = AsyncMock(return_value=None)
        mock_provider.create = AsyncMock(return_value=row)
        pyd = PydanticOrganization(id="org-new", name="new-org")

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _org_mgr()
                out = await mgr.create_organization(pyd)
                mock_provider.create.assert_awaited_once()
                assert mock_provider.create.await_args[0][0] == "organizations"
                assert mock_provider.create.await_args[0][1]["id"] == "org-new"
                assert out.id == "org-new"

    async def test_list_organizations_delegates(self):
        row = _organization_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _org_mgr()
            out = await mgr.list_organizations(limit=25)
            mock_provider.find_using_named_query.assert_awaited_once()
            args, kwargs = mock_provider.find_using_named_query.call_args
            assert args[0] == "organizations"
            assert args[1] == "organization_manager.list_organizations"
            assert kwargs["params"]["limit"] == 25
            assert len(out) == 1
            assert out[0].id == "org-1"


@pytest.mark.asyncio
class TestUserManagerDelegation:
    async def test_get_user_by_id_delegates_to_provider(self):
        row = _user_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _user_mgr()
                out = await mgr.get_user_by_id("user-1")
                args, kwargs = mock_provider.find_using_named_query.call_args
                assert args[0] == "users"
                assert args[1] == "user_manager.get_user_by_id"
                assert kwargs["params"] == {"id": "user-1"}
                assert out.id == "user-1"
                assert out.name == "test-user"

    async def test_create_user_delegates(self):
        row = _user_row_dict("user-new")
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)
        pyd = PydanticUser(
            id="user-new",
            name="new-user",
            status="active",
            timezone="UTC (UTC+00:00)",
            organization_id="org-1",
        )

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _user_mgr()
            out = await mgr.create_user(pyd)
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "users"
            assert mock_provider.create.await_args[0][1]["id"] == "user-new"
            assert out.id == "user-new"

    async def test_list_users_delegates(self):
        row = _user_row_dict()
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=[row])

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _user_mgr()
            out = await mgr.list_users(organization_id="org-1", limit=10)
            mock_provider.list.assert_awaited_once_with("users", organization_id="org-1", limit=10)
            assert len(out) == 1
            assert out[0].id == "user-1"


@pytest.mark.asyncio
class TestToolManagerDelegation:
    async def test_get_tool_by_id_delegates_to_provider(self):
        row = _tool_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _tool_mgr()
                out = await mgr.get_tool_by_id("tool-c0ffee00", _mock_actor())
                args, kwargs = mock_provider.find_using_named_query.call_args
                assert args[0] == "tools"
                assert args[1] == "tool_manager.get_tool_by_id"
                assert kwargs["params"]["id"] == "tool-c0ffee00"
                assert out.id == "tool-c0ffee00"
                assert out.name == "test_tool"

    async def test_create_tool_delegates(self):
        row = _tool_row_dict("tool-deadb33f")
        mock_provider = MagicMock()
        mock_provider.create = AsyncMock(return_value=row)
        tool = PydanticTool(
            id="tool-deadb33f",
            name="new_tool",
            json_schema={"type": "object", "description": "x"},
            source_code='''def new_tool_fn():
    """Create-tool delegation test."""
    pass
''',
        )
        actor = _mock_actor()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _tool_mgr()
            out = await mgr.create_tool(tool, actor=actor)
            mock_provider.create.assert_awaited_once()
            assert mock_provider.create.await_args[0][0] == "tools"
            sent = mock_provider.create.await_args[0][1]
            assert sent["_created_by_id"] == actor.id
            assert sent["organization_id"] == actor.organization_id
            assert out.id == "tool-deadb33f"

    async def test_list_tools_delegates(self):
        row = _tool_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])
        actor = _mock_actor()

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _tool_mgr()
            out = await mgr.list_tools(actor, limit=30)
            args, kwargs = mock_provider.find_using_named_query.call_args
            assert args[0] == "tools"
            assert args[1] == "tool_manager.list_tools"
            assert kwargs["params"] == {"organizationId": actor.organization_id, "limit": 30}
            assert len(out) == 1
            assert out[0].id == "tool-c0ffee00"


@pytest.mark.asyncio
class TestAgentManagerDelegation:
    async def test_get_agent_by_id_delegates_to_provider(self):
        row = _agent_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _agent_mgr()
                actor = _mock_actor()
                out = await mgr.get_agent_by_id("agent-1", actor)
                args, kwargs = mock_provider.find_using_named_query.call_args
                assert args[0] == "agents"
                assert args[1] == "agent_manager.get_agent_by_id"
                assert kwargs["params"] == {
                    "id": "agent-1",
                    "organizationId": actor.organization_id,
                    "createdById": actor.id,
                }
                assert kwargs["include_relationships"] == ["tools"]
                assert out.id == "agent-1"
                assert out.name == "test-agent"
                assert out.created_by_id == "client-1"

    async def test_delete_agent_delegates(self):
        existing = {**_agent_row_dict(), "parent_id": None}
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=existing)
        mock_provider.hard_delete = AsyncMock(return_value=None)

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _agent_mgr()
                await mgr.delete_agent("agent-1", _mock_actor())
                mock_provider.read.assert_awaited_once_with("agents", "agent-1")
                mock_provider.hard_delete.assert_awaited_once_with("agents", "agent-1")

    async def test_get_agent_by_id_populates_cache_on_ips_path(self):
        """GAP J: read-through cache should be populated after IPS read."""
        row = _agent_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        mock_cache = MagicMock()
        mock_cache.get_hash = AsyncMock(return_value=None)
        mock_cache.set_hash = AsyncMock()
        mock_cache.AGENT_PREFIX = "agent:"
        mock_cache.TOOL_PREFIX = "tool:"

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=mock_cache):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _agent_mgr()
                out = await mgr.get_agent_by_id("agent-1", _mock_actor())
                assert out.id == "agent-1"
                mock_cache.set_hash.assert_awaited()


# ---------------------------------------------------------------------------
# GAP C: update_event delegates to IPS relational + skips embedding
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestEpisodicUpdateEventDelegation:
    async def test_update_event_delegates_to_provider(self):
        existing = _episodic_row_dict()
        updated = {**existing, "summary": "new summary"}
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=existing)
        mock_provider.update = AsyncMock(return_value=updated)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            result = await mgr.update_event(
                event_id="ep-123",
                new_summary="new summary",
                actor=_mock_actor(),
            )
            mock_provider.update.assert_awaited_once()
            assert "episodic_memory" == mock_provider.update.await_args[0][0]
            assert "ep-123" == mock_provider.update.await_args[0][1]
            assert result.id == "ep-123"

    async def test_update_event_appends_details(self):
        existing = _episodic_row_dict()
        expected_details = existing["details"] + "\n" + "more info"
        updated = {**existing, "details": expected_details}
        mock_provider = MagicMock()
        mock_provider.read = AsyncMock(return_value=existing)
        mock_provider.update = AsyncMock(return_value=updated)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            result = await mgr.update_event(
                event_id="ep-123",
                new_details="more info",
                actor=_mock_actor(),
                update_mode="append",
            )
            call_data = mock_provider.update.await_args[0][2]
            assert "more info" in call_data["details"]


# ---------------------------------------------------------------------------
# GAP D: list_episodic_memory_by_org delegates to search provider
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestEpisodicListByOrgDelegation:
    async def test_list_by_org_uses_search_provider_client_api(self):
        row = _episodic_row_dict()
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=([row], None))

        agent_state = MagicMock(spec=AgentState)
        agent_state.embedding_config = MagicMock()

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_CLIENT_API):
                mgr = _episodic_mgr()
                results = await mgr.list_episodic_memory_by_org(
                    agent_state=agent_state,
                    organization_id="org-1",
                    query="test",
                )
                mock_search.search.assert_awaited_once()
                assert len(results) == 1
                assert results[0].id == "ep-123"

    async def test_list_by_org_uses_hybrid_for_engine(self):
        row = _episodic_row_dict()
        mock_search = MagicMock()

        agent_state = MagicMock(spec=AgentState)
        agent_state.embedding_config = MagicMock()

        with patch("mirix.database.search_provider.get_search_provider", return_value=mock_search):
            with patch("mirix.database.call_context.get_call_origin", return_value=CALL_ORIGIN_ENGINE):
                with patch("mirix.database.relational_provider.get_relational_provider", return_value=MagicMock()):
                    with patch(
                        "mirix.services.hybrid_search_helper.hybrid_search",
                        new_callable=AsyncMock,
                        return_value=[row],
                    ) as mock_hybrid:
                        mgr = _episodic_mgr()
                        results = await mgr.list_episodic_memory_by_org(
                            agent_state=agent_state,
                            organization_id="org-1",
                        )
                        mock_hybrid.assert_awaited_once()
                        assert len(results) == 1


# ---------------------------------------------------------------------------
# GAP E: list_episodic_memory_around_timestamp delegates to IPS relational
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestEpisodicAroundTimestampDelegation:
    async def test_delegates_to_relational_provider(self):
        row = _episodic_row_dict()
        mock_provider = MagicMock()
        mock_provider.list = AsyncMock(return_value=[row])

        agent_state = MagicMock(spec=AgentState)
        agent_state.embedding_config = MagicMock()

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)

        with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
            mgr = _episodic_mgr()
            results = await mgr.list_episodic_memory_around_timestamp(
                agent_state=agent_state,
                start_time=start,
                end_time=end,
                user=_mock_user(),
            )
            mock_provider.list.assert_awaited_once()
            call_kwargs = mock_provider.list.await_args
            assert call_kwargs[0][0] == "episodic_memory"
            assert len(results) == 1
            assert results[0].id == "ep-123"


# ---------------------------------------------------------------------------
# GAP J: tool_manager read-through cache on IPS path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestToolManagerCacheDelegation:
    async def test_get_tool_by_id_populates_cache_on_ips_path(self):
        row = _tool_row_dict()
        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock(return_value=[row])

        mock_cache = MagicMock()
        mock_cache.get_hash = AsyncMock(return_value=None)
        mock_cache.set_hash = AsyncMock()
        mock_cache.TOOL_PREFIX = "tool:"

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=mock_cache):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _tool_mgr()
                out = await mgr.get_tool_by_id("tool-c0ffee00", _mock_actor())
                assert out.id == "tool-c0ffee00"
                mock_cache.set_hash.assert_awaited()

    async def test_get_tool_by_id_returns_from_cache_hit(self):
        cached_data = _tool_row_dict()
        mock_cache = MagicMock()
        mock_cache.get_hash = AsyncMock(return_value=cached_data)
        mock_cache.TOOL_PREFIX = "tool:"

        mock_provider = MagicMock()
        mock_provider.find_using_named_query = AsyncMock()

        with patch("mirix.database.cache_provider.get_cache_provider", return_value=mock_cache):
            with patch("mirix.database.relational_provider.get_relational_provider", return_value=mock_provider):
                mgr = _tool_mgr()
                out = await mgr.get_tool_by_id("tool-c0ffee00", _mock_actor())
                assert out.id == "tool-c0ffee00"
                mock_provider.find_using_named_query.assert_not_awaited()
