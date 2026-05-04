"""OpenTelemetry SpanProcessor that masks allowlisted attribute values.

Runs on the OTEL export thread (off the request critical path). Wraps an
inner processor (typically ``BatchSpanProcessor``) and only redacts
attribute values whose key matches one of the allowlisted prefixes
(``llm.prompt``, ``llm.completion``, ``llm.tool_args`` by default).
Anything outside the allowlist is assumed to be PII-free because the call
sites are written to drop raw content. The processor is defense-in-depth
for the small surface where LLM-call spans deliberately capture content
for debugging.

Mutation point: ``_on_ending``. The OTEL SDK calls ``_on_ending(span)`` on
all registered processors immediately before snapshotting the span into a
``ReadableSpan`` and calling ``on_end(readable_span)``. ``ReadableSpan`` is
an immutable snapshot, so mutating attributes from ``on_end`` is a no-op
that lets the ``ReadableSpan`` be exported with the original (unredacted)
values.

By the time ``_on_ending`` runs, ``span._end_time`` is already set, which
makes ``span.set_attribute()`` reject the write with a "Setting attribute
on ended span" warning. To mutate post-end we have to assign directly to
the underlying ``BoundedAttributes`` mapping (``span._attributes``), which
``set_attribute``'s ended-span guard does not gate.

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

    def _on_ending(self, span: Span) -> None:
        """Mutate the span's attributes in place before it is snapshotted.

        Walk the underlying attributes mapping; for any key that matches one
        of the allowlisted prefixes and whose value is a string, replace the
        value with the redactor's output. Non-string values and
        out-of-allowlist keys are left untouched. Mutates ``span._attributes``
        directly because ``set_attribute`` is closed for writing once the
        span has been ended.
        """
        # Fail loud if the span shape changes — silently falling back to
        # ``span.attributes`` would land on a read-only mappingproxy and the
        # bare except below would swallow the TypeError, causing every PII
        # value to ship to OTLP unredacted. Better to log once and let the
        # outer broad except keep the export pipeline alive.
        attrs = getattr(span, "_attributes", None)
        if attrs is None:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "PIIRedactingSpanProcessor: span has no mutable _attributes "
                "(type=%s); skipping redaction for this span",
                type(span).__name__,
            )
            return
        for key, value in list(attrs.items()):
            if not isinstance(key, str):
                continue
            if not any(key.startswith(p) for p in self._prefixes):
                continue
            if not isinstance(value, str):
                continue
            redacted = mask(value)
            try:
                attrs[key] = redacted
            except Exception:
                pass
        # Forward to the inner processor so any wrapper that overrides
        # _on_ending (rare) still gets the hook.
        inner_on_ending = getattr(self._inner, "_on_ending", None)
        if inner_on_ending is not None:
            try:
                inner_on_ending(span)
            except Exception:
                pass

    def on_end(self, span: ReadableSpan) -> None:
        # By the time on_end is called the span is a ReadableSpan snapshot;
        # mutation here is a no-op. Just forward to the inner processor for
        # export.
        self._inner.on_end(span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
