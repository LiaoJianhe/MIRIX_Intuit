"""Tests for fail-fast behavior on LLM 422 errors.

Covers:
- `_get_ai_reply` re-raises LLMUnprocessableEntityError immediately, with no
  retries and no second_try fallback.
- `_get_ai_reply` continues to retry LLMServerError(DEPENDENCY_TIMEOUT) (the 424
  case) up to the retry limit — regression protection for the 424 path.
- `step()` catches LLMUnprocessableEntityError from sub-agent dispatch, emits a
  skip span, marks the memory source complete, and returns a clean
  MirixUsageStatistics(step_count=0) without propagating the exception.
- Edge case: 422 raised but memory_source_id is None — no mark_processing_complete
  call, function returns cleanly.
- Edge case: 422 raised and mark_processing_complete itself raises — re-raise.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.agent.agent import Agent
from mirix.errors import ErrorCode, LLMServerError, LLMUnprocessableEntityError
from mirix.schemas.usage import MirixUsageStatistics


def _make_agent_state(agent_type_name="meta_memory_agent"):
    from mirix.schemas.agent import AgentType

    state = MagicMock()
    state.id = "agent-123"
    state.name = agent_type_name
    state.created_by_id = "client-1"
    state.parent_id = None
    state.agent_type = getattr(AgentType, agent_type_name)
    state.is_type = lambda t: t == getattr(AgentType, agent_type_name)
    state.llm_config = MagicMock()
    state.llm_config.model = "gpt-4o-mini"
    state.llm_config.model_endpoint_type = "openai"
    state.llm_config.context_window = 128000
    state.tools = []
    state.tool_rules = []
    state.system = "test system prompt"
    state.embedding_config = MagicMock()
    return state


def _make_actor():
    actor = MagicMock()
    actor.id = "client-1"
    actor.organization_id = "org-1"
    actor.message_set_retention_count = 0
    return actor


def _make_user():
    user = MagicMock()
    user.id = "user-1"
    user.name = "test-user"
    return user


def _make_pydantic_source(processing_complete=False):
    source = MagicMock()
    source.processing_complete = processing_complete
    return source


def _setup_step_agent(memory_source_id="src-abc123"):
    """Build an Agent wired for step() unit tests."""
    agent_state = _make_agent_state("meta_memory_agent")
    user = _make_user()
    actor = _make_actor()

    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.user = user
    agent.actor = actor
    agent.user_id = user.id
    agent.memory_source_id = memory_source_id
    agent.direct_writes = None
    agent.memory_source_manager = MagicMock()
    agent.memory_source_manager.get_by_id = AsyncMock(return_value=_make_pydantic_source(processing_complete=False))
    agent.memory_source_manager.mark_processing_complete = AsyncMock()
    agent._persist_memory_source = AsyncMock()
    agent.interface = MagicMock()
    agent.filter_tags = None
    agent.block_filter_tags = None
    agent.use_cache = True
    agent.client_id = "client-1"
    agent.logger = MagicMock()
    agent.model = "gpt-4o-mini"
    agent.summarize = False
    agent.source_summary = None
    agent.source_summary_source = None
    agent._extract_topics_from_messages = AsyncMock(return_value=None)

    agent.message_manager = MagicMock()
    agent.message_manager.get_messages_for_agent = AsyncMock(return_value=[])
    agent.message_manager.get_messages_for_agent_user = AsyncMock(return_value=[])
    agent.message_manager.create_many_messages = AsyncMock()
    agent.message_manager.hard_delete_user_messages_for_agent = AsyncMock()
    agent.source_message_manager = MagicMock()

    return agent, actor, user


def _setup_get_ai_reply_agent():
    """Build an Agent wired for _get_ai_reply() unit tests."""
    agent_state = _make_agent_state("meta_memory_agent")
    agent_state.tools = []

    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.logger = MagicMock()
    agent.interface = MagicMock()
    agent.last_function_response = None
    agent.supports_structured_output = False
    agent.tool_rules_solver = SimpleNamespace(
        get_allowed_tool_names=lambda last_function_response=None: [],
        tool_call_history=[],
        init_tool_rules=[],
    )
    return agent


class TestGetAiReplyFastFailOn422:
    @pytest.mark.asyncio
    async def test_unprocessable_entity_propagates_after_one_call(self):
        """LLMUnprocessableEntityError must propagate immediately — no retries, no second_try."""
        agent = _setup_get_ai_reply_agent()

        call_count = 0

        async def raise_422(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise LLMUnprocessableEntityError("Toxicity detected")

        mock_client = MagicMock()
        mock_client.send_llm_request = AsyncMock(side_effect=raise_422)

        with pytest.raises(LLMUnprocessableEntityError):
            await agent._get_ai_reply(
                message_sequence=[MagicMock()],
                empty_response_retry_limit=3,
                backoff_factor=0.0,
                max_delay=0.0,
                step_count=0,
                llm_client=mock_client,
            )

        assert call_count == 1, f"Expected 1 LLM call, got {call_count}"

    @pytest.mark.asyncio
    async def test_dependency_timeout_still_retries_then_second_try(self):
        """424 LLMServerError(DEPENDENCY_TIMEOUT) must continue to retry up to the limit."""
        agent = _setup_get_ai_reply_agent()

        call_count = 0

        async def raise_424(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise LLMServerError("Upstream dependency timeout", code=ErrorCode.DEPENDENCY_TIMEOUT)

        mock_client = MagicMock()
        mock_client.send_llm_request = AsyncMock(side_effect=raise_424)

        retry_limit = 2

        with pytest.raises(Exception):
            await agent._get_ai_reply(
                message_sequence=[MagicMock(), MagicMock()],
                empty_response_retry_limit=retry_limit,
                backoff_factor=0.0,
                max_delay=0.0,
                step_count=0,
                llm_client=mock_client,
            )

        # First pass: retry_limit attempts; second_try: another retry_limit attempts.
        assert call_count == retry_limit * 2, (
            f"Expected {retry_limit * 2} LLM calls (initial retries + second_try retries), got {call_count}"
        )


class TestStepFastFailOn422:
    @pytest.mark.asyncio
    async def test_422_in_sub_agent_marks_source_complete_and_returns_clean(self):
        """422 from sub-agent dispatch should mark source complete and return step_count=0."""
        agent, actor, user = _setup_step_agent(memory_source_id="src-abc123")
        agent.inner_step = AsyncMock(side_effect=LLMUnprocessableEntityError("Suspicious language detected"))

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="bad input")

        with patch("mirix.agent.agent.LLMClient"), patch("mirix.agent.agent.emit_idempotency_skip_span") as mock_skip:
            result = await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                actor=actor,
                user=user,
            )

        assert isinstance(result, MirixUsageStatistics)
        assert result.step_count == 0

        agent.memory_source_manager.mark_processing_complete.assert_awaited_once_with("src-abc123")

        mock_skip.assert_called_once()
        kwargs = mock_skip.call_args.kwargs
        assert kwargs["reason"] == "llm-422-content-rejected"
        assert kwargs["name"] == "Source Rejected: LLM 422"
        assert kwargs["metadata"]["memory_source_id"] == "src-abc123"
        assert "error" in kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_422_with_no_memory_source_id_returns_cleanly(self):
        """422 with memory_source_id=None: no mark_processing_complete call, no AttributeError."""
        agent, actor, user = _setup_step_agent(memory_source_id=None)
        agent.inner_step = AsyncMock(side_effect=LLMUnprocessableEntityError("Toxicity detected"))

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="bad input")

        with patch("mirix.agent.agent.LLMClient"), patch("mirix.agent.agent.emit_idempotency_skip_span") as mock_skip:
            result = await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                actor=actor,
                user=user,
            )

        assert isinstance(result, MirixUsageStatistics)
        assert result.step_count == 0
        agent.memory_source_manager.mark_processing_complete.assert_not_called()
        mock_skip.assert_called_once()
        assert mock_skip.call_args.kwargs["reason"] == "llm-422-content-rejected"

    @pytest.mark.asyncio
    async def test_422_then_mark_complete_raises_propagates(self):
        """If mark_processing_complete raises after a 422, the outer handler re-raises."""
        agent, actor, user = _setup_step_agent(memory_source_id="src-abc123")
        agent.inner_step = AsyncMock(side_effect=LLMUnprocessableEntityError("Toxicity detected"))
        agent.memory_source_manager.mark_processing_complete = AsyncMock(side_effect=RuntimeError("DB unavailable"))

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="bad input")

        with patch("mirix.agent.agent.LLMClient"), patch("mirix.agent.agent.emit_idempotency_skip_span"):
            with pytest.raises(RuntimeError, match="DB unavailable"):
                await agent.step(
                    input_messages=[input_msg],
                    chaining=False,
                    max_chaining_steps=1,
                    actor=actor,
                    user=user,
                )
