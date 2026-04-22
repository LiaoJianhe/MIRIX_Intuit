import json
from unittest.mock import MagicMock

from mirix.agent.tool_call_consolidation import consolidate_memory_agent_tool_calls
from mirix.schemas.agent import AgentType
from mirix.schemas.openai.chat_completion_response import (
    FunctionCall,
    ToolCall,
)


def _make_tool_call(name: str, arguments: dict, id: str = "call-x"):
    return ToolCall(
        id=id,
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


class TestSingleOrEmpty:
    def test_empty_list_returns_unchanged(self):
        logger = MagicMock()
        result = consolidate_memory_agent_tool_calls(
            tool_calls=[],
            agent_type=AgentType.episodic_memory_agent,
            logger=logger,
        )
        assert result == []
        logger.warning.assert_not_called()

    def test_single_tool_call_returns_unchanged(self):
        logger = MagicMock()
        tc = _make_tool_call(
            "episodic_memory_insert", {"items": [{"summary": "e1"}]}, id="call-1"
        )
        result = consolidate_memory_agent_tool_calls(
            tool_calls=[tc],
            agent_type=AgentType.episodic_memory_agent,
            logger=logger,
        )
        assert result == [tc]
        logger.warning.assert_not_called()


class TestCombineSameInsert:
    def test_two_same_insert_calls_are_combined(self):
        logger = MagicMock()
        tc1 = _make_tool_call(
            "episodic_memory_insert",
            {"items": [{"summary": "event1"}]},
            id="call-1",
        )
        tc2 = _make_tool_call(
            "episodic_memory_insert",
            {"items": [{"summary": "event2"}, {"summary": "event3"}]},
            id="call-2",
        )
        result = consolidate_memory_agent_tool_calls(
            tool_calls=[tc1, tc2],
            agent_type=AgentType.episodic_memory_agent,
            logger=logger,
        )
        assert len(result) == 1
        combined = result[0]
        assert combined.function.name == "episodic_memory_insert"
        # Keeps first call's id
        assert combined.id == "call-1"
        args = json.loads(combined.function.arguments)
        assert args["items"] == [
            {"summary": "event1"},
            {"summary": "event2"},
            {"summary": "event3"},
        ]
        logger.info.assert_called()

    def test_three_same_insert_calls_are_combined(self):
        logger = MagicMock()
        tcs = [
            _make_tool_call(
                "semantic_memory_insert",
                {"items": [{"name": f"concept{i}"}]},
                id=f"call-{i}",
            )
            for i in range(3)
        ]
        result = consolidate_memory_agent_tool_calls(
            tool_calls=tcs,
            agent_type=AgentType.semantic_memory_agent,
            logger=logger,
        )
        assert len(result) == 1
        args = json.loads(result[0].function.arguments)
        assert len(args["items"]) == 3
        assert args["items"][0]["name"] == "concept0"
        assert args["items"][2]["name"] == "concept2"

    def test_all_insert_tool_types_supported(self):
        logger = MagicMock()
        insert_tools = [
            ("episodic_memory_insert", AgentType.episodic_memory_agent),
            ("semantic_memory_insert", AgentType.semantic_memory_agent),
            ("resource_memory_insert", AgentType.resource_memory_agent),
            ("procedural_memory_insert", AgentType.procedural_memory_agent),
            ("knowledge_vault_insert", AgentType.knowledge_vault_memory_agent),
        ]
        for tool_name, agent_type in insert_tools:
            tc1 = _make_tool_call(tool_name, {"items": [{"a": 1}]})
            tc2 = _make_tool_call(tool_name, {"items": [{"a": 2}]})
            result = consolidate_memory_agent_tool_calls(
                tool_calls=[tc1, tc2],
                agent_type=agent_type,
                logger=logger,
            )
            assert len(result) == 1, f"combine failed for {tool_name}"
            args = json.loads(result[0].function.arguments)
            assert len(args["items"]) == 2, f"items not merged for {tool_name}"


class TestFallbackTruncate:
    def test_mixed_tool_names_truncates_to_first(self):
        logger = MagicMock()
        tc1 = _make_tool_call("semantic_memory_update", {}, id="call-1")
        tc2 = _make_tool_call("semantic_memory_insert", {"items": [{"a": 1}]}, id="call-2")
        result = consolidate_memory_agent_tool_calls(
            tool_calls=[tc1, tc2],
            agent_type=AgentType.semantic_memory_agent,
            logger=logger,
        )
        assert result == [tc1]
        logger.warning.assert_called_once()
        warning_args = logger.warning.call_args
        # structured fields in warning message or args
        assert "Truncating" in warning_args[0][0]

    def test_non_insert_tool_even_if_same_truncates(self):
        """semantic_memory_update twice is NOT safe to combine (has old_ids)."""
        logger = MagicMock()
        tc1 = _make_tool_call(
            "semantic_memory_update",
            {"old_ids": ["id1"], "new_items": [{"name": "a"}]},
            id="call-1",
        )
        tc2 = _make_tool_call(
            "semantic_memory_update",
            {"old_ids": ["id2"], "new_items": [{"name": "b"}]},
            id="call-2",
        )
        result = consolidate_memory_agent_tool_calls(
            tool_calls=[tc1, tc2],
            agent_type=AgentType.semantic_memory_agent,
            logger=logger,
        )
        assert result == [tc1]
        logger.warning.assert_called_once()

    def test_insert_with_malformed_args_falls_back_to_truncate(self):
        logger = MagicMock()
        tc1 = ToolCall(
            id="call-1",
            function=FunctionCall(
                name="episodic_memory_insert", arguments="not valid json"
            ),
        )
        tc2 = _make_tool_call("episodic_memory_insert", {"items": [{"a": 1}]}, id="call-2")
        result = consolidate_memory_agent_tool_calls(
            tool_calls=[tc1, tc2],
            agent_type=AgentType.episodic_memory_agent,
            logger=logger,
        )
        assert result == [tc1]
        logger.warning.assert_called_once()

    def test_insert_missing_items_key_falls_back_to_truncate(self):
        logger = MagicMock()
        tc1 = _make_tool_call("episodic_memory_insert", {"wrong_key": []}, id="call-1")
        tc2 = _make_tool_call("episodic_memory_insert", {"items": [{"a": 1}]}, id="call-2")
        result = consolidate_memory_agent_tool_calls(
            tool_calls=[tc1, tc2],
            agent_type=AgentType.episodic_memory_agent,
            logger=logger,
        )
        assert result == [tc1]
        logger.warning.assert_called_once()


class TestNonMemoryAgent:
    def test_non_memory_agent_type_returns_unchanged(self):
        logger = MagicMock()
        tc1 = _make_tool_call("some_tool", {})
        tc2 = _make_tool_call("some_tool", {})
        result = consolidate_memory_agent_tool_calls(
            tool_calls=[tc1, tc2],
            agent_type=AgentType.chat_agent,
            logger=logger,
        )
        assert result == [tc1, tc2]
        logger.warning.assert_not_called()
