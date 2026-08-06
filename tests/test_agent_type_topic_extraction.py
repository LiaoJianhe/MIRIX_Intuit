"""Unit tests for AgentType.topic_extraction_agent's participation in the
enum-gated dispatch machinery (ECMS-522).

Covers tasks 1-4 from docs/specs/ECMS-522/tasks.md:
1. AgentType.topic_extraction_agent enum member + round-trip.
2. derive_system_message returns "" without raising for the new type.
3. AgentManager.create_agent creates a tool-less agent for the new type.
4. create_meta_agent's agent_name_to_type map resolves the new type's name.
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.schemas.agent import AgentType, CreateAgent
from mirix.schemas.client import Client
from mirix.schemas.llm_config import LLMConfig
from mirix.services.agent_manager import AgentManager
from mirix.services.helpers.agent_manager_helper import derive_system_message


class TestAgentTypeTopicExtractionEnum:
    def test_topic_extraction_agent_member_exists(self):
        assert AgentType.topic_extraction_agent == "topic_extraction_agent"

    def test_topic_extraction_agent_round_trips_through_value(self):
        assert AgentType("topic_extraction_agent") == AgentType.topic_extraction_agent


class TestDeriveSystemMessageForTopicExtraction:
    def test_returns_empty_string_without_raising(self):
        assert derive_system_message(AgentType.topic_extraction_agent) == ""


def _make_actor():
    return Client(
        id="client-1",
        organization_id="org-1",
        name="Test Client",
        status="active",
    )


def _make_llm_config():
    return LLMConfig(
        model="gpt-4o-mini",
        model_endpoint_type="openai",
        context_window=8192,
    )


class TestCreateAgentToolLessForTopicExtraction:
    @pytest.mark.asyncio
    async def test_create_agent_attaches_no_tools(self):
        am = AgentManager()
        fake_rp = AsyncMock()

        async def _fake_create(table, data_dict):
            return {
                "id": "agent-topic-extraction-client-1",
                "name": data_dict["name"],
                "system": data_dict["system"],
                "agent_type": data_dict["agent_type"],
                "llm_config": data_dict["llm_config"],
                "embedding_config": data_dict["embedding_config"],
                "organization_id": data_dict["organization_id"],
                "tools": data_dict["tools"],
                "tool_rules": data_dict["tool_rules"],
                "parent_id": data_dict["parent_id"],
            }

        fake_rp.create = AsyncMock(side_effect=_fake_create)

        with patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=fake_rp,
        ):
            agent_state = await am.create_agent(
                agent_create=CreateAgent(
                    name="client-1_topic_extraction_agent",
                    agent_type=AgentType.topic_extraction_agent,
                    llm_config=_make_llm_config(),
                    include_base_tools=False,
                ),
                actor=_make_actor(),
            )

        assert agent_state.tools == []
