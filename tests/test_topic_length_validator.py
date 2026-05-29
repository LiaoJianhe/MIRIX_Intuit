"""Unit tests for VEPAGE-1149 topic length validator.

The meta agent extracts conversation topics via LLM and forwards them
to downstream search as a ``;``-joined string. IPS Search enforces a
200-character cap on text-field values; oversized topics aborted the
entire memory-extraction pipeline. ``Agent._enforce_topic_length``
truncates any individual ``;``-delimited segment that exceeds the cap
so the search still runs (degraded but not crashed). These tests pin
that behavior.
"""

from unittest.mock import MagicMock

from mirix.agent.agent import Agent


def _agent() -> Agent:
    """Bare-minimum Agent instance for testing pure helpers."""
    agent = Agent.__new__(Agent)
    agent.agent_state = MagicMock()
    agent.agent_state.name = "meta_memory_agent"
    return agent


class TestEnforceTopicLength:
    def test_none_passes_through(self):
        assert _agent()._enforce_topic_length(None) is None

    def test_empty_string_passes_through(self):
        # Empty input short-circuits at the top of the validator and
        # passes through unchanged — preserves the existing contract.
        assert _agent()._enforce_topic_length("") == ""

    def test_short_single_topic_unchanged(self):
        assert _agent()._enforce_topic_length("hello") == "hello"

    def test_multiple_short_topics_unchanged(self):
        assert (
            _agent()._enforce_topic_length("alpha; beta ;gamma")
            == "alpha;beta;gamma"
        )

    def test_oversized_single_topic_truncated_to_200(self):
        topic = "x" * 250
        out = _agent()._enforce_topic_length(topic)
        assert out is not None
        assert len(out) == 200

    def test_only_oversized_segment_in_multi_topic_string_truncated(self):
        # Two short topics + one oversize → oversize gets clipped, others
        # are preserved.
        short_a = "alpha"
        short_b = "beta"
        long_c = "c" * 250
        out = _agent()._enforce_topic_length(f"{short_a}; {long_c}; {short_b}")
        parts = out.split(";")
        assert parts[0] == "alpha"
        assert len(parts[1]) == 200
        assert parts[2] == "beta"

    def test_215_char_4_topic_string_from_ticket_repro(self):
        """The literal failing input from the VEPAGE-1149 LangFuse trace.
        Each individual topic is already under 200; the joined string is
        215. Validator should leave each topic untouched and rejoin
        cleanly."""
        topics = (
            "QuickBooks Online payroll tax workflow for restaurant clients; "
            "Reviewing payroll register for accurate tax filing; "
            "QuickBooks subscription renewal date and details; "
            "Switching restaurant clients to monthly payroll runs"
        )
        out = _agent()._enforce_topic_length(topics)
        parts = out.split(";")
        assert len(parts) == 4
        for p in parts:
            assert len(p) <= 200

    def test_empty_segments_dropped(self):
        # Extra ;; and trailing/leading separators should not produce
        # empty topics in the output.
        out = _agent()._enforce_topic_length("alpha; ; beta;;")
        assert out == "alpha;beta"
