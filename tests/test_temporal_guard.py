"""Unit tests for the temporal guard (S8) — prevents backdated sources from overwriting memories.

Tests verify that:
- _should_update_memory returns False when occurred_at < max existing citation occurred_at
- _should_update_memory returns True when occurred_at >= max, or when occurred_at/memory_source_id absent
- Update functions (core, semantic, procedural, resource, knowledge_vault) skip when guarded
- Episodic functions are NOT guarded (events slot by timestamp)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.functions.function_sets.memory_tools import (
    _should_update_memory,
    core_memory_append,
    core_memory_rewrite,
    episodic_memory_insert,
    episodic_memory_merge,
    episodic_memory_replace,
    knowledge_vault_update,
    procedural_memory_update,
    resource_memory_update,
    semantic_memory_update,
)


# --- Helpers ---

def _make_agent(memory_source_id=None, use_cache=True, external_thread_id=None, occurred_at=None):
    """Create a minimal mock agent with memory source fields."""
    agent = MagicMock()
    agent.memory_source_id = memory_source_id
    agent.use_cache = use_cache
    agent.external_thread_id = external_thread_id
    agent.occurred_at = occurred_at
    agent.filter_tags = None
    agent.client_id = "client-1"
    agent.user_id = "user-1"
    agent.actor = MagicMock()
    agent.actor.id = "client-1"
    agent.actor.organization_id = "org-1"
    agent.user = MagicMock()
    agent.user.id = "user-1"
    agent.user.organization_id = "org-1"
    agent.agent_state = MagicMock()
    agent.agent_state.parent_id = None
    agent.agent_state.id = "agent-1"
    agent.interface = MagicMock()
    return agent


OLDER = datetime(2026, 1, 1, tzinfo=timezone.utc)
NEWER = datetime(2026, 6, 1, tzinfo=timezone.utc)


# --- _should_update_memory ---

class TestShouldUpdateMemory:
    """Tests for the temporal guard helper."""

    @pytest.mark.asyncio
    async def test_allows_when_no_occurred_at(self):
        """No occurred_at on agent → arrival-order fallback → allow."""
        agent = _make_agent(memory_source_id="src-1", occurred_at=None)
        assert await _should_update_memory(agent, "semantic", "sem-1") is True

    @pytest.mark.asyncio
    async def test_allows_when_no_memory_source_id(self):
        """No memory_source_id → backward compat → allow."""
        agent = _make_agent(memory_source_id=None, occurred_at=OLDER)
        assert await _should_update_memory(agent, "semantic", "sem-1") is True

    @pytest.mark.asyncio
    async def test_allows_when_no_existing_citations(self):
        """No prior citations for this memory → allow."""
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=None)
            assert await _should_update_memory(agent, "semantic", "sem-1") is True

    @pytest.mark.asyncio
    async def test_allows_when_occurred_at_is_newer(self):
        """Source occurred_at > existing max → allow."""
        agent = _make_agent(memory_source_id="src-1", occurred_at=NEWER)
        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=OLDER)
            assert await _should_update_memory(agent, "semantic", "sem-1") is True

    @pytest.mark.asyncio
    async def test_allows_when_occurred_at_is_equal(self):
        """Source occurred_at == existing max → allow (not strictly older)."""
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=OLDER)
            assert await _should_update_memory(agent, "semantic", "sem-1") is True

    @pytest.mark.asyncio
    async def test_blocks_when_occurred_at_is_older(self):
        """Source occurred_at < existing max → block."""
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)
            assert await _should_update_memory(agent, "semantic", "sem-1") is False


# --- core_memory_append ---

class TestCoreMemoryAppendTemporalGuard:

    @pytest.mark.asyncio
    async def test_skips_append_when_backdated(self):
        """core_memory_append should skip when temporal guard blocks."""
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        blocks = MagicMock()
        block = MagicMock()
        block.id = "block-1"
        block.value = "existing"
        block.limit = 5000
        blocks.get_block.return_value = block

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)
            mock_instance.create = AsyncMock(return_value=None)

            result = await core_memory_append(agent, blocks, "human", "new content")

            assert result is None
            blocks.update_block_value.assert_not_called()
            mock_instance.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_proceeds_when_not_backdated(self):
        """core_memory_append should proceed when temporal guard allows."""
        agent = _make_agent(memory_source_id="src-1", occurred_at=NEWER)
        blocks = MagicMock()
        block = MagicMock()
        block.id = "block-1"
        block.value = "existing"
        block.limit = 5000
        blocks.get_block.return_value = block

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=OLDER)
            mock_instance.create = AsyncMock(return_value=None)

            result = await core_memory_append(agent, blocks, "human", "new content")

            assert result is None
            blocks.update_block_value.assert_called_once()
            mock_instance.create.assert_awaited_once()


# --- core_memory_rewrite ---

class TestCoreMemoryRewriteTemporalGuard:

    @pytest.mark.asyncio
    async def test_skips_rewrite_when_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        blocks = MagicMock()
        block = MagicMock()
        block.id = "block-1"
        block.value = "old content"
        block.limit = 5000
        blocks.get_block.return_value = block

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)
            mock_instance.create = AsyncMock(return_value=None)

            result = await core_memory_rewrite(agent, blocks, "human", "new content")

            assert result is None
            blocks.update_block_value.assert_not_called()
            mock_instance.create.assert_not_awaited()


# --- semantic_memory_update ---

class TestSemanticMemoryUpdateTemporalGuard:

    @pytest.mark.asyncio
    async def test_skips_update_when_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        agent.semantic_memory_manager = MagicMock()
        agent.semantic_memory_manager.delete_semantic_item_by_id = AsyncMock()
        agent.semantic_memory_manager.insert_semantic_item = AsyncMock()

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)

            result = await semantic_memory_update(agent, ["sem-old-1"], [{"name": "n", "summary": "s", "details": "d", "source": "src"}])

            assert "temporal guard" in result.lower()
            agent.semantic_memory_manager.delete_semantic_item_by_id.assert_not_awaited()
            agent.semantic_memory_manager.insert_semantic_item.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_proceeds_when_not_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=NEWER)
        mock_item = MagicMock()
        mock_item.id = "sem-new-1"
        agent.semantic_memory_manager = MagicMock()
        agent.semantic_memory_manager.delete_semantic_item_by_id = AsyncMock()
        agent.semantic_memory_manager.insert_semantic_item = AsyncMock(return_value=mock_item)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=OLDER)
            mock_instance.create = AsyncMock(return_value=None)

            result = await semantic_memory_update(agent, ["sem-old-1"], [{"name": "n", "summary": "s", "details": "d", "source": "src"}])

            agent.semantic_memory_manager.delete_semantic_item_by_id.assert_awaited_once()
            agent.semantic_memory_manager.insert_semantic_item.assert_awaited_once()
            assert "sem-new-1" in result


# --- procedural_memory_update ---

class TestProceduralMemoryUpdateTemporalGuard:

    @pytest.mark.asyncio
    async def test_skips_update_when_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        agent.procedural_memory_manager = MagicMock()
        agent.procedural_memory_manager.delete_procedure_by_id = AsyncMock()
        agent.procedural_memory_manager.insert_procedure = AsyncMock()

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)

            result = await procedural_memory_update(agent, ["proc-old-1"], [{"entry_type": "t", "summary": "s", "steps": []}])

            assert result is None
            agent.procedural_memory_manager.delete_procedure_by_id.assert_not_awaited()
            agent.procedural_memory_manager.insert_procedure.assert_not_awaited()


# --- resource_memory_update ---

class TestResourceMemoryUpdateTemporalGuard:

    @pytest.mark.asyncio
    async def test_skips_update_when_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        agent.resource_memory_manager = MagicMock()
        agent.resource_memory_manager.delete_resource_by_id = AsyncMock()
        agent.resource_memory_manager.insert_resource = AsyncMock()

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)

            result = await resource_memory_update(agent, ["res-old-1"], [{"title": "t", "summary": "s", "resource_type": "r", "content": "c"}])

            assert result is None
            agent.resource_memory_manager.delete_resource_by_id.assert_not_awaited()
            agent.resource_memory_manager.insert_resource.assert_not_awaited()


# --- knowledge_vault_update ---

class TestKnowledgeVaultUpdateTemporalGuard:

    @pytest.mark.asyncio
    async def test_skips_update_when_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        agent.knowledge_vault_manager = MagicMock()
        agent.knowledge_vault_manager.delete_knowledge_by_id = AsyncMock()
        agent.knowledge_vault_manager.insert_knowledge = AsyncMock()

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)

            result = await knowledge_vault_update(agent, ["kv-old-1"], [{"entry_type": "t", "source": "s", "sensitivity": "low", "secret_value": "v", "caption": "c"}])

            assert result is None
            agent.knowledge_vault_manager.delete_knowledge_by_id.assert_not_awaited()
            agent.knowledge_vault_manager.insert_knowledge.assert_not_awaited()


# --- Episodic insert is NOT guarded (append-only) ---

class TestEpisodicInsertNotGuarded:
    """episodic_memory_insert is append-only — no guard needed."""

    @pytest.mark.asyncio
    async def test_episodic_insert_has_no_guard(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        mock_event = MagicMock()
        mock_event.id = "event-1"
        agent.episodic_memory_manager = MagicMock()
        agent.episodic_memory_manager.insert_event = AsyncMock(return_value=mock_event)

        with patch("mirix.functions.function_sets.memory_tools._should_update_memory") as mock_guard:
            with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
                mock_instance = MockMgr.return_value
                mock_instance.create = AsyncMock(return_value=None)

                await episodic_memory_insert(
                    agent,
                    [{"occurred_at": "2026-01-01T00:00:00Z", "event_type": "test", "actor": "user", "summary": "s", "details": "d"}],
                )

                mock_guard.assert_not_called()
                agent.episodic_memory_manager.insert_event.assert_awaited_once()


# --- Episodic merge IS guarded ---

class TestEpisodicMergeTemporalGuard:

    @pytest.mark.asyncio
    async def test_skips_merge_when_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        agent.episodic_memory_manager = MagicMock()
        agent.episodic_memory_manager.update_event = AsyncMock()

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)

            result = await episodic_memory_merge(agent, "event-1", "new summary", "new details")

            assert result is None
            agent.episodic_memory_manager.update_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_proceeds_merge_when_not_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=NEWER)
        mock_event = MagicMock()
        mock_event.id = "event-1"
        mock_event.summary = "merged"
        mock_event.details = "details"
        agent.episodic_memory_manager = MagicMock()
        agent.episodic_memory_manager.update_event = AsyncMock(return_value=mock_event)

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=OLDER)
            mock_instance.create = AsyncMock(return_value=None)

            result = await episodic_memory_merge(agent, "event-1", "merged", "details")

            agent.episodic_memory_manager.update_event.assert_awaited_once()
            mock_instance.create.assert_awaited_once()


# --- Episodic replace IS guarded ---

class TestEpisodicReplaceTemporalGuard:

    @pytest.mark.asyncio
    async def test_skips_replace_when_backdated(self):
        agent = _make_agent(memory_source_id="src-1", occurred_at=OLDER)
        agent.episodic_memory_manager = MagicMock()
        agent.episodic_memory_manager.get_episodic_memory_by_id = AsyncMock(return_value=MagicMock())
        agent.episodic_memory_manager.delete_event_by_id = AsyncMock()
        agent.episodic_memory_manager.insert_event = AsyncMock()

        with patch("mirix.services.memory_citation_manager.MemoryCitationManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_max_occurred_at = AsyncMock(return_value=NEWER)

            result = await episodic_memory_replace(
                agent, ["event-1"],
                [{"occurred_at": "2026-01-01T00:00:00Z", "event_type": "test", "actor": "user", "summary": "s", "details": "d"}],
            )

            assert result is None
            agent.episodic_memory_manager.delete_event_by_id.assert_not_awaited()
            agent.episodic_memory_manager.insert_event.assert_not_awaited()
