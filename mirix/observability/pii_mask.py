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

The Langfuse client wiring (see ``langfuse_client.py``) reads the mask
via :func:`get_langfuse_mask` at construction time, defaulting to the
env-configured ispy-pii masker if no override has been registered. Tests
and downstream consumers (e.g. ECMS) can register a different callable
via :func:`set_langfuse_mask` before calling
``initialize_langfuse()``.

The kill switch ``MIRIX_LANGFUSE_MASK_ENABLED=false`` disables the
network call; the mask becomes a passthrough. Defaults to enabled.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Final, Optional

import httpx

# Re-export from mirix.pii so the placeholder string and the
# ispy-pii payload/extract helpers are defined once. Splunk-bound
# error logs (mirix.pii) and Langfuse-bound trace masking (this
# module) hit the same ispy-pii endpoint; sharing the wire-level
# helpers keeps tier/format changes from drifting between paths.
from mirix.pii import (
    REDACTED_PLACEHOLDER,
    build_ispy_payload,
    extract_redacted,
    get_ispy_pii_endpoint,
    get_ispy_pii_timeout_seconds,
)

_DEFAULT_TIMEOUT_S: Final[float] = 0.2
# Bound the LRU so a long-running process doesn't accumulate unbounded
# memory if the mask sees high-cardinality strings (e.g. unique queries).
_DEFAULT_CACHE_SIZE: Final[int] = 4096


def _enabled() -> bool:
    return os.getenv("MIRIX_LANGFUSE_MASK_ENABLED", "true").lower() == "true"


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

    # Bounded LRU that only stores successful results. functools.lru_cache
    # cannot distinguish "ispy-pii returned masked text" from "ispy-pii
    # failed and we substituted the placeholder", so a single transient
    # failure (429, 5xx, timeout, malformed body) would poison the cache
    # for that string until process restart. After ispy-pii recovers,
    # traces would still show the placeholder for every poisoned string.
    # Use an OrderedDict + lock so insert/evict is thread-safe (the mask
    # is invoked on the calling thread, which may be a request handler).
    _cache: "OrderedDict[str, str]" = OrderedDict()
    _cache_lock = threading.Lock()

    def _cache_get(key: str) -> Optional[str]:
        with _cache_lock:
            value = _cache.get(key)
            if value is not None:
                _cache.move_to_end(key)
            return value

    def _cache_put(key: str, value: str) -> None:
        with _cache_lock:
            _cache[key] = value
            _cache.move_to_end(key)
            while len(_cache) > cache_size:
                _cache.popitem(last=False)

    def _mask_one(text: str) -> str:
        if not text:
            return text
        cached = _cache_get(text)
        if cached is not None:
            return cached
        try:
            resp = client.post(endpoint, json=build_ispy_payload(text))
            if resp.status_code != 200:
                return REDACTED_PLACEHOLDER
            result = extract_redacted(resp.json())
        except Exception:
            return REDACTED_PLACEHOLDER
        # extract_redacted falls back to REDACTED_PLACEHOLDER on a
        # malformed 200 response; treat that as a failure too and do not
        # cache it, so the next call gets a fresh attempt.
        if result != REDACTED_PLACEHOLDER:
            _cache_put(text, result)
        return result

    def mask(data: Any = None, **_: Any) -> Any:
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


_default_mask_singleton: Optional[Callable[..., Any]] = None


def _default_mask() -> Callable[..., Any]:
    """Build the default env-configured masker on first use, then cache.

    Intentionally caches per-process: the masker holds a long-lived
    httpx.Client and we don't want runtime env-var changes to silently
    rebuild it. Tests that need a different endpoint should call
    ``build_langfuse_mask`` directly rather than relying on the
    default.
    """
    global _default_mask_singleton
    if _default_mask_singleton is not None:
        return _default_mask_singleton
    _default_mask_singleton = build_langfuse_mask(
        endpoint=get_ispy_pii_endpoint(),
        timeout_seconds=get_ispy_pii_timeout_seconds(),
    )
    return _default_mask_singleton


def ispy_pii_mask(data: Any = None, **kwargs: Any) -> Any:
    """Default mask using env-configured ispy-pii endpoint.

    Reads ``MIRIX_ISPY_PII_ENDPOINT`` (defaults to the e2e mesh
    endpoint) and ``MIRIX_ISPY_PII_TIMEOUT_MS`` (defaults to 200ms).
    Suitable for direct registration via ``set_langfuse_mask`` or as
    the default consulted by ``get_langfuse_mask`` when nothing has
    been registered.
    """
    return _default_mask()(data=data, **kwargs)


# Module-level singleton holding the active mask callable. Exposed via
# set_langfuse_mask / get_langfuse_mask so downstream consumers (notably
# ECMS) can register their own callable before initialize_langfuse()
# constructs the Langfuse client. Module-level (not pydantic-settings)
# because Callable isn't an env-driven setting type.
_active_mask: Optional[Callable[..., Any]] = None


def set_langfuse_mask(fn: Optional[Callable[..., Any]]) -> None:
    """Register the mask callable that initialize_langfuse() will use.

    Pass ``None`` to clear the registration; the next call to
    :func:`get_langfuse_mask` will return the env-configured default
    masker (:func:`ispy_pii_mask`).
    """
    global _active_mask
    _active_mask = fn


def get_langfuse_mask() -> Callable[..., Any]:
    """Return the active mask callable.

    Falls back to the env-configured default
    (:func:`ispy_pii_mask`) when nothing has been registered. The
    ``Langfuse(...)`` constructor reads from this function, so
    ``set_langfuse_mask`` must be called before
    ``initialize_langfuse()`` to take effect.
    """
    return _active_mask or ispy_pii_mask
