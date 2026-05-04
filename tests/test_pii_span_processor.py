"""Verify PIIRedactingSpanProcessor masks only allowlisted attrs."""

from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from mirix.pii import set_redactor
from mirix.pii_span_processor import PIIRedactingSpanProcessor


# -- End-to-end: real OTEL pipeline ----------------------------------------
#
# These tests catch the failure mode where the processor mutates a
# ReadableSpan snapshot (no-op) instead of the live Span. They run a span
# through TracerProvider -> PIIRedactingSpanProcessor -> SimpleSpanProcessor
# -> InMemorySpanExporter and assert on what the exporter actually receives.


def _build_pipeline(allowed_prefixes=("llm.",)):
    exporter = InMemorySpanExporter()
    inner = SimpleSpanProcessor(exporter)
    proc = PIIRedactingSpanProcessor(inner, allowed_prefixes=allowed_prefixes)
    provider = TracerProvider()
    provider.add_span_processor(proc)
    return provider.get_tracer(__name__), exporter, provider


def test_end_to_end_masks_allowlisted_string_attributes():
    tracer, exporter, provider = _build_pipeline()
    set_redactor(lambda s: s.replace("123-45-6789", "[SSN]"))
    try:
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("llm.prompt", "my SSN is 123-45-6789")
            span.set_attribute("http.method", "POST")
        provider.force_flush()

        attrs = exporter.get_finished_spans()[0].attributes
        # Allowlisted string attribute is redacted.
        assert attrs["llm.prompt"] == "my SSN is [SSN]"
        # Out-of-allowlist attribute is untouched.
        assert attrs["http.method"] == "POST"
    finally:
        set_redactor(None)


def test_end_to_end_passthrough_when_no_redactor_registered():
    tracer, exporter, provider = _build_pipeline()
    set_redactor(None)
    with tracer.start_as_current_span("test") as span:
        span.set_attribute("llm.prompt", "raw content")
    provider.force_flush()
    attrs = exporter.get_finished_spans()[0].attributes
    # Default redactor is identity; allowlisted attribute is unchanged.
    assert attrs["llm.prompt"] == "raw content"


def test_end_to_end_skips_non_string_values():
    tracer, exporter, provider = _build_pipeline()
    sentinel = "REDACTED"
    set_redactor(lambda s: sentinel)
    try:
        with tracer.start_as_current_span("test") as span:
            # OTEL doesn't allow arbitrary types; ints are valid.
            span.set_attribute("llm.token_count", 42)
        provider.force_flush()
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["llm.token_count"] == 42
    finally:
        set_redactor(None)


def test_end_to_end_keys_outside_allowlist_are_not_masked():
    tracer, exporter, provider = _build_pipeline(allowed_prefixes=("llm.",))
    set_redactor(lambda s: "[REDACTED]")
    try:
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("http.method", "POST")
            span.set_attribute("http.url", "http://example.test")
        provider.force_flush()
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["http.method"] == "POST"
        assert attrs["http.url"] == "http://example.test"
    finally:
        set_redactor(None)


# -- Unit-level: lifecycle forwarding ---------------------------------------


def test_processor_forwards_lifecycle_methods():
    inner = MagicMock()
    proc = PIIRedactingSpanProcessor(inner, allowed_prefixes=("llm.",))

    proc.shutdown()
    inner.shutdown.assert_called_once()

    inner.force_flush.return_value = True
    assert proc.force_flush(1000) is True
    inner.force_flush.assert_called_once_with(1000)


def test_processor_on_start_forwards_to_inner():
    inner = MagicMock()
    proc = PIIRedactingSpanProcessor(inner, allowed_prefixes=("llm.",))
    fake_span = MagicMock()
    proc.on_start(fake_span, parent_context=None)
    inner.on_start.assert_called_once_with(fake_span, None)


def test_processor_on_end_forwards_to_inner():
    """on_end is a forward-only path now; mutation happens in _on_ending."""
    inner = MagicMock()
    proc = PIIRedactingSpanProcessor(inner, allowed_prefixes=("llm.",))
    fake_span = MagicMock()
    proc.on_end(fake_span)
    inner.on_end.assert_called_once_with(fake_span)
