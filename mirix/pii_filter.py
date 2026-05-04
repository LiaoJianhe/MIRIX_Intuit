"""WARNING+ ``logging.Filter`` that masks records via the registered redactor.

Hot-path (info-level) records are NEVER touched — they are expected to be
free of PII because call sites are written to log structural metadata only
(counts, IDs, mode flags), never raw user content. This filter is the
safety net for tracebacks, validation errors, and any warning/error log
lines that may carry user-visible content.

By default the redactor is a passthrough (``mirix.pii.mask`` is identity).
Register a real redactor at startup with :func:`mirix.pii.set_redactor` to
turn on masking; see ``mirix/pii.py`` for the contract and an example.
"""

from __future__ import annotations

import logging

from mirix.pii import mask

_MIN_LEVEL = logging.WARNING


class PIIRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < _MIN_LEVEL:
            return True
        rendered = record.getMessage()
        redacted = mask(rendered)
        if redacted != rendered:
            record.msg = redacted
            record.args = ()
        return True
