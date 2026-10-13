"""Unit tests for the central trace-attribute helper (``update_trace_attributes``).

Trace-level tags in Langfuse are replaced wholesale on every write of the
``langfuse.trace.tags`` span attribute, and the save trace is written from
two independent processes (HTTP entry + worker) plus the finalize emitter —
last writer wins. The helper kills that bug structurally: it accumulates the
per-context tag set in a ContextVar and always writes the FULL set, so
whatever the flush order, the trace converges to the union.

Langfuse v4 removed ``Langfuse.update_current_trace``; trace-level fields are
span attributes, so these tests assert on the attributes written to the
current span. Pure-python: the span is a recording fake; no DB, no network.
"""

from unittest.mock import MagicMock, patch

import pytest

from mirix.observability import context as obs_context
from mirix.observability.trace_attrs import (
    reset_trace_tags,
    update_trace_attributes,
    write_trace_attributes,
)

TAGS = "langfuse.trace.tags"
METADATA = "langfuse.trace.metadata"


class RecordingSpan:
    """Stand-in for the current OTel span that records every attribute write."""

    def __init__(self, recording=True):
        self.writes = []
        self._recording = recording

    def is_recording(self):
        return self._recording

    def set_attribute(self, key, value):
        self.writes.append((key, value))

    def tag_writes(self):
        return [value for key, value in self.writes if key == TAGS]

    def attrs(self):
        """Last value written per key."""
        return dict(self.writes)


@pytest.fixture(autouse=True)
def _clean_context():
    """Each test starts and ends with no trace context and an empty tag set."""
    obs_context.clear_trace_context()
    reset_trace_tags()
    yield
    obs_context.clear_trace_context()
    reset_trace_tags()


def _with_trace(trace_id="trace-1"):
    obs_context.set_trace_context(trace_id=trace_id, observation_id="obs-1")


def _patched(span, langfuse=None):
    """Patch the Langfuse client lookup and the current-span lookup."""
    return (
        patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse if langfuse is not None else MagicMock(),
        ),
        patch("mirix.observability.trace_attrs.get_current_span", return_value=span),
    )


class TestUpdateTraceAttributes:
    def test_tags_merge_across_calls_and_full_set_is_written(self):
        """Successive calls accumulate; every write sends the full sorted set."""
        _with_trace()
        span = RecordingSpan()
        p1, p2 = _patched(span)
        with p1, p2:
            update_trace_attributes(tags=["tid:t-1"])
            update_trace_attributes(tags=["client:aqna-client"])
            update_trace_attributes(tags=["write_kind:extraction"])

        writes = span.tag_writes()
        assert len(writes) == 3
        assert writes[0] == ["tid:t-1"]
        assert writes[1] == sorted(["tid:t-1", "client:aqna-client"])
        assert writes[2] == sorted(["tid:t-1", "client:aqna-client", "write_kind:extraction"])

    def test_duplicate_tags_are_set_deduplicated(self):
        """Re-adding the same tag (e.g. a deduped redelivery re-tagging the same
        outcome) does not duplicate it — set semantics."""
        _with_trace()
        span = RecordingSpan()
        p1, p2 = _patched(span)
        with p1, p2:
            update_trace_attributes(tags=["save_outcome:success"])
            update_trace_attributes(tags=["save_outcome:success"])

        assert span.tag_writes()[-1] == ["save_outcome:success"]

    def test_metadata_written_per_call_as_flat_attributes(self):
        """Metadata is written per call, one attribute per key (Langfuse merges
        metadata by key server-side, so no client-side accumulation is needed)."""
        _with_trace()
        span = RecordingSpan()
        p1, p2 = _patched(span)
        with p1, p2:
            update_trace_attributes(tags=["tid:t-1"], metadata={"tid": "t-1"})
            update_trace_attributes(metadata={"write_kind": "direct"})

        attrs = span.attrs()
        assert attrs[f"{METADATA}.tid"] == "t-1"
        assert attrs[f"{METADATA}.write_kind"] == "direct"
        # A metadata-only call still writes the full accumulated tag set.
        assert span.tag_writes()[-1] == ["tid:t-1"]

    def test_tag_values_used_verbatim_including_colons(self):
        """Values may themselves contain ':' (the tid: precedent embeds arbitrary
        gateway TIDs) — no escaping or splitting."""
        _with_trace()
        span = RecordingSpan()
        p1, p2 = _patched(span)
        with p1, p2:
            update_trace_attributes(tags=["tid:gw:12:34"])

        assert span.tag_writes()[-1] == ["tid:gw:12:34"]

    def test_no_op_when_langfuse_disabled(self):
        _with_trace()
        span = RecordingSpan()
        with (
            patch("mirix.observability.trace_attrs.get_langfuse_client", return_value=None),
            patch("mirix.observability.trace_attrs.get_current_span", return_value=span),
        ):
            update_trace_attributes(tags=["client:x"])  # must not raise
        assert span.writes == []

    def test_no_op_when_no_trace_id_active(self):
        span = RecordingSpan()
        p1, p2 = _patched(span)
        with p1, p2:
            update_trace_attributes(tags=["client:x"])
        assert span.writes == []

    def test_never_raises_when_span_write_fails(self):
        _with_trace()
        span = RecordingSpan()
        span.set_attribute = MagicMock(side_effect=RuntimeError("boom"))
        p1, p2 = _patched(span)
        with p1, p2:
            update_trace_attributes(tags=["client:x"])  # must not raise


class TestWriteTraceAttributes:
    def test_tags_and_metadata_use_langfuse_v4_attribute_names(self):
        span = RecordingSpan()
        with patch("mirix.observability.trace_attrs.get_current_span", return_value=span):
            write_trace_attributes(tags=["t-1"], metadata={"k": "v"})

        assert span.attrs() == {TAGS: ["t-1"], f"{METADATA}.k": "v"}

    def test_metadata_skips_none_and_encodes_non_primitives(self):
        span = RecordingSpan()
        with patch("mirix.observability.trace_attrs.get_current_span", return_value=span):
            write_trace_attributes(metadata={"a": None, "b": 3, "c": {"k": "v"}, "d": True})

        assert span.attrs() == {
            f"{METADATA}.b": 3,
            f"{METADATA}.c": '{"k": "v"}',
            f"{METADATA}.d": True,
        }

    def test_no_op_without_a_recording_span(self):
        span = RecordingSpan(recording=False)
        with patch("mirix.observability.trace_attrs.get_current_span", return_value=span):
            write_trace_attributes(tags=["t"], metadata={"k": "v"})
        assert span.writes == []


class TestReset:
    def test_reset_trace_tags_empties_the_set(self):
        _with_trace()
        span = RecordingSpan()
        p1, p2 = _patched(span)
        with p1, p2:
            update_trace_attributes(tags=["tid:t-1", "client:c-1"])
            reset_trace_tags()
            update_trace_attributes(tags=["tid:t-2"])

        assert span.tag_writes()[-1] == ["tid:t-2"]

    def test_clear_trace_context_resets_the_tag_set(self):
        """dispatch_save's finally calls clear_trace_context(); the tag set must
        reset with it so one save's tags never leak into the next message on a
        reused worker task."""
        _with_trace()
        span = RecordingSpan()
        p1, p2 = _patched(span)
        with p1, p2:
            update_trace_attributes(tags=["save_outcome:success"])
            obs_context.clear_trace_context()
            _with_trace(trace_id="trace-2")
            update_trace_attributes(tags=["tid:t-next"])

        assert span.tag_writes()[-1] == ["tid:t-next"]


class TestAgainstRealLangfuseV4:
    def test_trace_attributes_reach_the_exported_span(self):
        """End to end on the real SDK: the attributes the helper writes are the
        ones Langfuse v4 itself emits for trace-level fields."""
        from langfuse import Langfuse, propagate_attributes
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        client = Langfuse(
            public_key="pk",
            secret_key="sk",
            host="http://localhost:1",
            tracer_provider=provider,
            tracing_enabled=True,
        )

        _with_trace()
        with patch("mirix.observability.trace_attrs.get_langfuse_client", return_value=client):
            with client.start_as_current_observation(name="root", as_type="span"):
                # Identity via the documented propagate_attributes API (as the
                # HTTP entry does); tags/metadata via the helper.
                with propagate_attributes(trace_name="POST /v1/memories", user_id="u-1", session_id="s-1"):
                    update_trace_attributes(tags=["tid:t-1"], metadata={"client": "c-1"})
                    update_trace_attributes(tags=["client:c-1"])

        (span,) = exporter.get_finished_spans()
        attrs = dict(span.attributes)
        assert attrs["langfuse.trace.name"] == "POST /v1/memories"
        assert attrs["user.id"] == "u-1"
        assert attrs["session.id"] == "s-1"
        assert list(attrs[TAGS]) == ["client:c-1", "tid:t-1"]
        assert attrs[f"{METADATA}.client"] == "c-1"
