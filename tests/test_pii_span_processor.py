"""VEPAGE-983: verify PIIRedactingSpanProcessor masks only allowlisted attrs."""

from unittest.mock import MagicMock, patch

from mirix.pii_span_processor import PIIRedactingSpanProcessor


def _make_span(attrs):
    span = MagicMock()
    span.attributes = attrs
    span._set_attribute_calls = []
    span.set_attribute.side_effect = lambda k, v: span._set_attribute_calls.append(
        (k, v)
    )
    return span


def test_processor_masks_only_allowlisted_prefix():
    inner = MagicMock()
    proc = PIIRedactingSpanProcessor(inner, allowed_prefixes=("llm.",))
    span = _make_span({"llm.prompt": "ssn 123-45-6789", "http.method": "POST"})

    with patch("mirix.pii_span_processor.mask", return_value="ssn [REDACTED]") as m:
        proc.on_end(span)
        m.assert_called_once_with("ssn 123-45-6789")

    keys = {k for k, _ in span._set_attribute_calls}
    assert "llm.prompt" in keys
    assert "http.method" not in keys
    inner.on_end.assert_called_once_with(span)


def test_processor_passes_through_when_no_allowed_attrs():
    inner = MagicMock()
    proc = PIIRedactingSpanProcessor(inner, allowed_prefixes=("llm.",))
    span = _make_span({"http.method": "POST"})

    with patch("mirix.pii_span_processor.mask") as m:
        proc.on_end(span)
        m.assert_not_called()

    inner.on_end.assert_called_once_with(span)


def test_processor_skips_non_string_values():
    inner = MagicMock()
    proc = PIIRedactingSpanProcessor(inner, allowed_prefixes=("llm.",))
    span = _make_span({"llm.token_count": 42})

    with patch("mirix.pii_span_processor.mask") as m:
        proc.on_end(span)
        m.assert_not_called()


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
