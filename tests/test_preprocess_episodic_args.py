"""
Unit tests for the episodic-memory tool-arg preprocessor.

Covers:
- Legacy behavior: when no ``occurred_at`` override is set on the agent,
  LLM-emitted ``occurred_at`` values are converted to UTC before dispatch.
- Override short-circuit: when the API has set ``occurred_at`` on the agent,
  the LLM's value is irrelevant and must not be parsed — callers downstream
  throw it away anyway (see mirix/functions/function_sets/memory_tools.py).
  Previously an ISO-8601 ``T`` from the LLM crashed the whole turn here.
"""

from datetime import datetime, timezone as dt_timezone

from mirix.agent.agent import _preprocess_episodic_tool_args


TZ = "America/Los_Angeles (UTC-08:00)"


class TestOverrideShortCircuit:
    """When an API-provided occurred_at override exists, the LLM's value is ignored."""

    def test_iso_8601_from_llm_is_not_parsed_when_override_set(self):
        # Before the fix, this raised ValueError on the 'T' separator.
        function_args = {
            "items": [
                {
                    "occurred_at": "2026-04-22T08:00:00",
                    "event_type": "user_message",
                    "actor": "user",
                    "summary": "x",
                    "details": "y",
                }
            ]
        }
        override = datetime(2026, 4, 22, 15, 0, 0, tzinfo=dt_timezone.utc)

        _preprocess_episodic_tool_args(
            "episodic_memory_insert", function_args, timezone_str=TZ, occurred_at_override=override
        )

        # Should not raise; LLM value is left untouched (downstream ignores it).
        assert function_args["items"][0]["occurred_at"] == "2026-04-22T08:00:00"

    def test_garbage_from_llm_is_not_parsed_when_override_set(self):
        function_args = {
            "items": [{"occurred_at": "totally bogus", "event_type": "x", "actor": "u", "summary": "", "details": ""}]
        }
        override = datetime(2026, 4, 22, 15, 0, 0, tzinfo=dt_timezone.utc)

        _preprocess_episodic_tool_args(
            "episodic_memory_insert", function_args, timezone_str=TZ, occurred_at_override=override
        )

        assert function_args["items"][0]["occurred_at"] == "totally bogus"


class TestNoOverridePath:
    """Without an override, we still convert LLM values so the downstream
    function gets a UTC datetime. This preserves the legacy behavior."""

    def test_space_separated_llm_value_is_converted_to_utc(self):
        function_args = {
            "items": [
                {
                    "occurred_at": "2026-04-22 08:00:00",
                    "event_type": "user_message",
                    "actor": "user",
                    "summary": "x",
                    "details": "y",
                }
            ]
        }

        _preprocess_episodic_tool_args(
            "episodic_memory_insert", function_args, timezone_str=TZ, occurred_at_override=None
        )

        converted = function_args["items"][0]["occurred_at"]
        # Converted to a UTC-aware datetime: 08:00 LA (DST) -> 15:00 UTC.
        assert converted.hour == 15
        assert converted.utcoffset() == dt_timezone.utc.utcoffset(converted)

    def test_iso_8601_llm_value_is_converted_to_utc(self):
        # With the convert_timezone_to_utc fix, ISO-8601 now works on the
        # non-override path too.
        function_args = {
            "items": [
                {
                    "occurred_at": "2026-04-22T08:00:00",
                    "event_type": "user_message",
                    "actor": "user",
                    "summary": "x",
                    "details": "y",
                }
            ]
        }

        _preprocess_episodic_tool_args(
            "episodic_memory_insert", function_args, timezone_str=TZ, occurred_at_override=None
        )

        converted = function_args["items"][0]["occurred_at"]
        assert converted.hour == 15


class TestReplace:
    """episodic_memory_replace uses ``new_items`` instead of ``items``."""

    def test_replace_short_circuits_on_override(self):
        function_args = {
            "new_items": [
                {
                    "occurred_at": "2026-04-22T08:00:00",
                    "event_type": "user_message",
                    "actor": "user",
                    "summary": "x",
                    "details": "y",
                }
            ]
        }
        override = datetime(2026, 4, 22, 15, 0, 0, tzinfo=dt_timezone.utc)

        _preprocess_episodic_tool_args(
            "episodic_memory_replace", function_args, timezone_str=TZ, occurred_at_override=override
        )

        assert function_args["new_items"][0]["occurred_at"] == "2026-04-22T08:00:00"


class TestUnrelated:
    """Non-episodic functions pass through untouched."""

    def test_unrelated_tool_is_noop(self):
        function_args = {"query": "hello"}
        _preprocess_episodic_tool_args(
            "search_in_memory", function_args, timezone_str=TZ, occurred_at_override=None
        )
        assert function_args == {"query": "hello"}
