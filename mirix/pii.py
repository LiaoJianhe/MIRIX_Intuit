"""ispy-pii REST client.

Redacts PII from text via Intuit's ispy-pii service. Used by the WARNING+ log
filter and by the OTEL span processor — never on the request hot path. On any
failure (network, timeout, non-200), returns a fully-redacted placeholder
rather than raw text. The kill switch MIRIX_ISPY_PII_ENABLED=false disables
the network call entirely (passthrough). Paired with the hot-path "drop
content, don't mask" pattern, that means the service still emits no PII even
when masking is off.

Default disabled. PROD/E2E deployments must explicitly set
MIRIX_ISPY_PII_ENABLED=true.
"""

from __future__ import annotations

import logging
import os
from typing import Final

import httpx

logger = logging.getLogger(__name__)

REDACTED_PLACEHOLDER: Final[str] = "[REDACTED — PII masking unavailable]"

_DEFAULT_ENDPOINT: Final[str] = "https://ispypiis.api.intuit.com/v2/analyze"
_DEFAULT_TIMEOUT_MS: Final[int] = 200


def _enabled() -> bool:
    return os.getenv("MIRIX_ISPY_PII_ENABLED", "false").lower() == "true"


def _endpoint() -> str:
    return os.getenv("MIRIX_ISPY_PII_ENDPOINT", _DEFAULT_ENDPOINT)


def _timeout_seconds() -> float:
    try:
        return (
            int(os.getenv("MIRIX_ISPY_PII_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS)))
            / 1000.0
        )
    except ValueError:
        return _DEFAULT_TIMEOUT_MS / 1000.0


_client: httpx.Client = httpx.Client(timeout=_timeout_seconds())


def mask(text: str, *, fmt: str = "PLAIN_TEXT") -> str:
    """Return ispy-pii-redacted text, or a placeholder on any failure."""
    if not text:
        return text
    if not _enabled():
        return text
    try:
        resp = _client.post(
            _endpoint(),
            json={
                "text": text,
                "format": fmt,
                "sensitivityLevel": "SENSITIVE",
                "confidenceLevel": "LIKELY",
            },
            timeout=_timeout_seconds(),
        )
        if resp.status_code != 200:
            return REDACTED_PLACEHOLDER
        body = resp.json()
        redacted = body.get("redactedText")
        return redacted if isinstance(redacted, str) else REDACTED_PLACEHOLDER
    except (httpx.HTTPError, ValueError):
        return REDACTED_PLACEHOLDER
