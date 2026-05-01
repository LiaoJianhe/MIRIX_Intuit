"""WARNING+ logging.Filter that masks records via ispy-pii.

Hot-path (info-level) records are NEVER touched — they are expected to be
free of PII because the call sites have been changed to drop raw content.
This filter is the safety net for tracebacks, validation errors, and any
warning/error log lines that may carry user-visible content.

When adding a new log line:
- info-level: do NOT log raw conversation content, prompts, completions, or
  full request payloads. Log structural metadata (counts, IDs, mode flags)
  only.
- warning+/error: content is allowed; this filter will mask it before it
  leaves the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from mirix.pii import mask

_MIN_LEVEL = logging.WARNING


@lru_cache(maxsize=512)
def _masked(msg: str) -> str:
    return mask(msg)


class PIIRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < _MIN_LEVEL:
            return True
        rendered = record.getMessage()
        redacted = _masked(rendered)
        if redacted != rendered:
            record.msg = redacted
            record.args = ()
        return True
