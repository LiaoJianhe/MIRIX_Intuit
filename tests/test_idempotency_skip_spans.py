"""Unit tests for idempotency skip spans.

Verifies that when L1 source-dedup, L2 processing-complete, or L3 temporal-guard
short-circuits a memory operation, a Langfuse span is emitted under the current
trace context so the trace makes the skip reason visible.
"""

from unittest.mock import MagicMock, patch

import pytest

from mirix.observability.skip_spans import emit_idempotency_skip_span


class TestEmitIdempotencySkipSpan:
    def test_no_op_when_langfuse_disabled(self):
        """Skip span should be a silent no-op when Langfuse isn't initialized."""
        with patch("mirix.observability.skip_spans.get_langfuse_client", return_value=None):
            # Should not raise.
            emit_idempotency_skip_span(name="x", reason="source-deduped")

    def test_no_op_when_no_trace_context(self):
        """Skip span should be a silent no-op when no active trace_id."""
        client = MagicMock()
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch("mirix.observability.skip_spans.get_trace_context", return_value={}),
        ):
            emit_idempotency_skip_span(name="x", reason="processing-complete")
        client.start_as_current_observation.assert_not_called()

    def test_emits_span_with_reason_and_metadata(self):
        """Happy path: emits a span under the current parent with reason/metadata."""
        client = MagicMock()
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=MagicMock())
        span_ctx.__exit__ = MagicMock(return_value=False)
        client.start_as_current_observation.return_value = span_ctx

        trace_ctx = {"trace_id": "trace-123", "observation_id": "obs-abc"}
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch("mirix.observability.skip_spans.get_trace_context", return_value=trace_ctx),
            patch("mirix.observability.skip_spans.mark_observation_as_child"),
        ):
            emit_idempotency_skip_span(
                name="Idempotency Skip: temporal guard (episodic)",
                reason="temporal-guard",
                metadata={"memory_type": "episodic", "memory_id": "m-1"},
            )

        assert client.start_as_current_observation.called
        kwargs = client.start_as_current_observation.call_args.kwargs
        assert kwargs["name"] == "Idempotency Skip: temporal guard (episodic)"
        assert kwargs["as_type"] == "span"
        assert kwargs["trace_context"]["trace_id"] == "trace-123"
        assert kwargs["trace_context"]["parent_span_id"] == "obs-abc"
        md = kwargs["metadata"]
        assert md["skip_reason"] == "temporal-guard"
        assert md["memory_type"] == "episodic"
        assert md["memory_id"] == "m-1"

    def test_swallows_exceptions(self):
        """Observability failures must not break the calling code path."""
        client = MagicMock()
        client.start_as_current_observation.side_effect = RuntimeError("boom")
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch(
                "mirix.observability.skip_spans.get_trace_context",
                return_value={"trace_id": "t1", "observation_id": "o1"},
            ),
        ):
            # Should not raise.
            emit_idempotency_skip_span(name="x", reason="source-deduped")
