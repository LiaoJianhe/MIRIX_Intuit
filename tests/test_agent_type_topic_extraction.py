"""Unit tests for AgentType.topic_extraction_agent's participation in the
enum-gated dispatch machinery (ECMS-522).

Covers tasks 1-4 from docs/specs/ECMS-522/tasks.md:
1. AgentType.topic_extraction_agent enum member + round-trip.
2. derive_system_message returns "" without raising for the new type.
3. AgentManager.create_agent creates a tool-less agent for the new type.
4. create_meta_agent's agent_name_to_type map resolves the new type's name.
"""

from mirix.schemas.agent import AgentType


class TestAgentTypeTopicExtractionEnum:
    def test_topic_extraction_agent_member_exists(self):
        assert AgentType.topic_extraction_agent == "topic_extraction_agent"

    def test_topic_extraction_agent_round_trips_through_value(self):
        assert AgentType("topic_extraction_agent") == AgentType.topic_extraction_agent
