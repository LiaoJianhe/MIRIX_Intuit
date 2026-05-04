"""Pluggable text-redaction hook used by Mirix's logging filter and span processor.

Mirix does not ship a redactor — by default :func:`mask` returns its input
unchanged. Callers wire up redaction by registering a callable with
:func:`set_redactor`. The callable must take a string and return a string;
it can implement any policy the caller wants (regex, ML, an external
service, etc.).

Two consumers inside Mirix call :func:`mask`:

- :class:`mirix.pii_filter.PIIRedactionFilter` — a stdlib ``logging.Filter``
  that masks records at WARNING+ before they're emitted.
- :class:`mirix.pii_span_processor.PIIRedactingSpanProcessor` — an OTEL
  ``SpanProcessor`` that masks allowlisted span attributes on the export
  thread before they ship to OTLP.

Example — plug in a simple regex-based redactor::

    import re
    from mirix.pii import set_redactor

    SSN = re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b")
    EMAIL = re.compile(r"\\b[\\w.+-]+@[\\w.-]+\\b")

    def my_redactor(text: str) -> str:
        text = SSN.sub("[SSN]", text)
        text = EMAIL.sub("[EMAIL]", text)
        return text

    set_redactor(my_redactor)

The registry is process-global. Register once at startup, before the first
log emission or span export. If no redactor is registered, :func:`mask` is a
passthrough — useful for tests and for environments that don't need
redaction.
"""

from __future__ import annotations

from typing import Callable, Optional

Redactor = Callable[[str], str]


def _passthrough(text: str) -> str:
    return text


_redactor: Redactor = _passthrough


def set_redactor(redactor: Optional[Redactor]) -> None:
    """Register the process-global redactor.

    Pass ``None`` to reset to the passthrough default.
    """
    global _redactor
    _redactor = redactor if redactor is not None else _passthrough


def get_redactor() -> Redactor:
    """Return the currently registered redactor (or the passthrough default)."""
    return _redactor


def mask(text: str) -> str:
    """Return ``text`` with PII redacted by the registered redactor.

    Empty strings short-circuit. Any exception raised by the redactor is
    caught and the original ``text`` is returned — a buggy redactor must
    never break logging or trace export.
    """
    if not text:
        return text
    try:
        return _redactor(text)
    except Exception:
        return text
