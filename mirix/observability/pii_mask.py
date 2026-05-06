"""ispy-pii-backed mask callback for the Langfuse SDK.

The Langfuse Python SDK accepts a ``mask`` callable on the ``Langfuse``
constructor. The SDK invokes it for every observation's input/output/
metadata, walking dicts and lists recursively, and exports whatever the
callable returns. We use that hook to send each string value through
ispy-pii so PII tokens (emails, SSNs, phone numbers, credit cards, etc.)
are replaced with masked equivalents (e.g. ``***-**-6789``) before
Langfuse ships traces upstream.

The mask runs on Langfuse's flush thread, not the request hot path, so a
synchronous network call to ispy-pii is acceptable. Failures must never
raise: a logging/observability path that re-enters its own exception
handler is a deadlock waiting to happen. On any failure we substitute
:data:`REDACTED_PLACEHOLDER` so Langfuse exports the placeholder rather
than dropping the span or leaking unredacted content.

Usage (once the Langfuse client wiring lands)::

    from langfuse import Langfuse
    from mirix.observability import build_langfuse_mask

    langfuse = Langfuse(
        public_key=...,
        secret_key=...,
        host=...,
        mask=build_langfuse_mask(
            endpoint="http://ispypiis-e2e.api.intuit.com/v2/analyze",
            timeout_seconds=0.2,
        ),
    )

The kill switch ``MIRIX_LANGFUSE_MASK_ENABLED=false`` disables the
network call; the mask becomes a passthrough. Defaults to enabled.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Callable, Final, Optional

import httpx

logger = logging.getLogger(__name__)

REDACTED_PLACEHOLDER: Final[str] = "[REDACTED — PII masking unavailable]"

_DEFAULT_TIMEOUT_S: Final[float] = 0.2
# Bound the LRU so a long-running process doesn't accumulate unbounded
# memory if the mask sees high-cardinality strings (e.g. unique queries).
_DEFAULT_CACHE_SIZE: Final[int] = 4096


def _enabled() -> bool:
    return os.getenv("MIRIX_LANGFUSE_MASK_ENABLED", "true").lower() == "true"


def _payload(text: str) -> dict:
    return {
        "text": text,
        "format": "PLAIN_TEXT",
        "sensitivityLevel": "SENSITIVE",
        "confidenceLevel": "LIKELY",
    }


def _extract_redacted(body: object) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    redacted = body.get("redactedText")
    return redacted if isinstance(redacted, str) else None


def build_langfuse_mask(
    endpoint: str,
    timeout_seconds: float = _DEFAULT_TIMEOUT_S,
    cache_size: int = _DEFAULT_CACHE_SIZE,
) -> Callable[..., Any]:
    """Construct a Langfuse-compatible mask callable.

    Returns a function with signature ``mask(data, **kwargs) -> Any``
    that the Langfuse SDK will invoke for every value it would export.
    Strings are forwarded to ispy-pii; dicts/lists are walked
    recursively; other types pass through unchanged.

    The returned callable is closed over a long-lived
    :class:`httpx.Client` so connection pooling holds across the
    process. The client is intentionally synchronous: the mask fires on
    Langfuse's flush thread and never on a request handler's event
    loop.
    """
    client = httpx.Client(timeout=timeout_seconds)

    @lru_cache(maxsize=cache_size)
    def _mask_one(text: str) -> str:
        if not text:
            return text
        try:
            resp = client.post(endpoint, json=_payload(text))
            if resp.status_code != 200:
                return REDACTED_PLACEHOLDER
            redacted = _extract_redacted(resp.json())
            return redacted if redacted is not None else REDACTED_PLACEHOLDER
        except Exception:
            return REDACTED_PLACEHOLDER

    def mask(data: Any = None, **_: Any) -> Any:
        # Langfuse calls mask(data=...) on every observation field. The
        # SDK walks nested structures itself in some configurations and
        # delivers leaf strings; in others it delivers the whole object.
        # Handle both shapes by walking ourselves.
        if not _enabled():
            return data
        if isinstance(data, str):
            return _mask_one(data)
        if isinstance(data, dict):
            return {k: mask(data=v) for k, v in data.items()}
        if isinstance(data, list):
            return [mask(data=item) for item in data]
        if isinstance(data, tuple):
            return tuple(mask(data=item) for item in data)
        return data

    return mask


# Convenience alias for callers that want a default-configured mask
# without specifying the endpoint at construction time. The endpoint is
# read from the ``MIRIX_ISPY_PII_ENDPOINT`` env var so deployment can
# point preprod and prod at the right ispy-pii instance.
def ispy_pii_mask(data: Any = None, **kwargs: Any) -> Any:
    """Default mask using env-configured ispy-pii endpoint.

    Reads ``MIRIX_ISPY_PII_ENDPOINT`` (defaults to the e2e mesh
    endpoint) and ``MIRIX_ISPY_PII_TIMEOUT_MS`` (defaults to 200ms).
    Suitable for direct use as ``Langfuse(mask=ispy_pii_mask)`` once the
    upstream client wiring accepts the callback.
    """
    return _default_mask()(data=data, **kwargs)


@lru_cache(maxsize=1)
def _default_mask() -> Callable[..., Any]:
    endpoint = os.getenv(
        "MIRIX_ISPY_PII_ENDPOINT",
        "http://ispypiis-e2e.api.intuit.com/v2/analyze",
    )
    try:
        timeout_ms = int(os.getenv("MIRIX_ISPY_PII_TIMEOUT_MS", "200"))
    except ValueError:
        timeout_ms = 200
    return build_langfuse_mask(
        endpoint=endpoint,
        timeout_seconds=timeout_ms / 1000.0,
    )
