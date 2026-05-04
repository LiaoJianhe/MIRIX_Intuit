"""OpenTelemetry SpanProcessor that masks allowlisted attribute values.

Runs on the OTEL export thread (off the request critical path). Wraps an
inner processor (typically ``BatchSpanProcessor``) and only redacts
attribute values whose key matches one of the allowlisted prefixes
(``llm.prompt``, ``llm.completion``, ``llm.tool_args`` by default).
Anything outside the allowlist is assumed to be PII-free because the call
sites are written to drop raw content. The processor is defense-in-depth
for the small surface where LLM-call spans deliberately capture content
for debugging.

Redaction goes through :func:`mirix.pii.mask`, which is a passthrough by
default. Register a real redactor at startup with
:func:`mirix.pii.set_redactor` to turn on masking.
"""

from __future__ import annotations

from typing import Iterable, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from mirix.pii import mask


class PIIRedactingSpanProcessor(SpanProcessor):
    def __init__(
        self,
        inner: SpanProcessor,
        allowed_prefixes: Iterable[str] = (
            "llm.prompt",
            "llm.completion",
            "llm.tool_args",
        ),
    ) -> None:
        self._inner = inner
        self._prefixes = tuple(allowed_prefixes)

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self._inner.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        attrs = getattr(span, "attributes", None) or {}
        for key, value in list(attrs.items()):
            if not isinstance(key, str):
                continue
            if not any(key.startswith(p) for p in self._prefixes):
                continue
            if not isinstance(value, str):
                continue
            redacted = mask(value)
            try:
                # ReadableSpan does not expose set_attribute publicly, but the
                # underlying Span subclass that flows through processors does.
                span.set_attribute(key, redacted)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._inner.on_end(span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
