"""ispy-pii client + safe error logging helper for MIRIX.

Use :func:`log_error_strip_pii` at any async catch site where the
exception's ``str(e)`` could echo user input (LLM provider 4xx response
bodies, MIRIX-internal errors that wrap user content, ``inner_step()``
failures whose wrapped ``LLMError`` carries the upstream provider
message). The helper sends ``str(e)`` to ispy-pii and emits the
redacted version, so the error reason is preserved in Splunk with PII
tokens (emails, SSNs, phone numbers, credit cards, etc.) scrubbed::

    try:
        ...
    except Exception as e:
        await pii.log_error_strip_pii(
            logger, "inner_step() failed: num_messages=%d", len(msgs), exc=e
        )
        raise

Why async (and not the sync helper this replaces)?

MIRIX is an async-native codebase (see CLAUDE.md: "All I/O is async —
never introduce sync blocking calls"). Every catch site we mask from
runs inside an async event loop — ``inner_step()``, the LLM clients'
``handle_llm_error`` chain, the batch path. A synchronous
``httpx.Client.post()`` here would block the loop for up to the
ispy-pii timeout, mirroring the VEPAGE-983 anti-pattern where a sync
``httpx`` call inside a logging filter stalled the FastAPI request
thread. The mask call therefore runs through ``httpx.AsyncClient`` and
yields cooperatively while waiting.

The Langfuse trace path uses a separate seam (the
``mirix.observability.pii_mask`` masker registered as the Langfuse
``mask=`` callback). That callback fires on the SDK's flush thread, not
on a request handler's event loop, so synchronous ``httpx.Client`` is
still correct there. The two paths are intentionally separate: error
logs go to Splunk via stdlib logging; trace attributes go to Langfuse
via the SDK.

The kill switch ``MIRIX_ISPY_PII_ENABLED=false`` disables the network
call (passthrough). Defaults to enabled. If the network call fails we
substitute :data:`REDACTED_PLACEHOLDER` so the helper never raises —
logging paths must not re-enter their own exception handlers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Final, Optional

import httpx

from mirix.log import safe_traceback

REDACTED_PLACEHOLDER: Final[str] = "[REDACTED — PII masking unavailable]"

_DEFAULT_ENDPOINT: Final[str] = "http://ispypiis-e2e.api.intuit.com/v2/analyze"
_DEFAULT_TIMEOUT_MS: Final[int] = 200


def _enabled() -> bool:
    return os.getenv("MIRIX_ISPY_PII_ENABLED", "true").lower() == "true"


def get_ispy_pii_endpoint() -> str:
    """Resolve the ispy-pii v2/analyze endpoint from env.

    Shared between the async log helper here and the Langfuse mask
    callback in ``mirix.observability.pii_mask`` so deployment-time
    URL switches (preprod vs prod) land in one place.
    """
    return os.getenv("MIRIX_ISPY_PII_ENDPOINT", _DEFAULT_ENDPOINT)


def get_ispy_pii_timeout_seconds() -> float:
    """Resolve the ispy-pii request timeout (seconds) from env.

    Shared with ``mirix.observability.pii_mask``. Defaults to 200ms;
    the env var is the millisecond value (``MIRIX_ISPY_PII_TIMEOUT_MS``).
    """
    try:
        return (
            int(os.getenv("MIRIX_ISPY_PII_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS)))
            / 1000.0
        )
    except ValueError:
        return _DEFAULT_TIMEOUT_MS / 1000.0


def build_ispy_payload(text: str) -> dict:
    """Construct the ispy-pii v2/analyze request body.

    Shared between the async log helper here and the Langfuse mask
    callback in ``mirix.observability.pii_mask`` so a config change
    (sensitivity tier, etc.) lands in exactly one place.

    SENSITIVE is the documented default tier per sdm-docs/ispypii: it
    covers common PII (names, emails, phone numbers, addresses, SSNs,
    credit cards). HIGHLY_SENSITIVE — which sounds stricter — is
    actually a narrower subset (only passwords, API keys, SSNs, credit
    cards, driver's licenses), so it would miss the long tail of names
    and contact details that error logs and trace attributes most
    often echo.
    """
    return {
        "text": text,
        "format": "PLAIN_TEXT",
        "sensitivityLevel": "SENSITIVE",
        "confidenceLevel": "LIKELY",
    }


def extract_redacted(body: object) -> str:
    """Pull ``redactedText`` from an ispy-pii response, or fall back.

    Shared with ``mirix.observability.pii_mask``; centralized so the
    fallback semantics (return the placeholder on any malformed
    response) are identical across both call paths.
    """
    if not isinstance(body, dict):
        return REDACTED_PLACEHOLDER
    redacted = body.get("redactedText")
    return redacted if isinstance(redacted, str) else REDACTED_PLACEHOLDER


_async_client: Optional[httpx.AsyncClient] = None


def _get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(timeout=get_ispy_pii_timeout_seconds())
    return _async_client


async def _mask_async(text: str) -> str:
    """Async ispy-pii redaction. Returns the placeholder on any failure.

    Must never raise — async catch sites rely on this to log without
    re-entering their own exception handler. The outer
    ``except Exception`` is deliberate.
    """
    if not text or not _enabled():
        return text
    try:
        resp = await _get_async_client().post(
            get_ispy_pii_endpoint(),
            json=build_ispy_payload(text),
            timeout=get_ispy_pii_timeout_seconds(),
        )
        if resp.status_code != 200:
            return REDACTED_PLACEHOLDER
        return extract_redacted(resp.json())
    except Exception:
        return REDACTED_PLACEHOLDER


async def log_error_strip_pii(
    log: logging.Logger,
    fmt: str,
    *args: Any,
    exc: BaseException,
    level: int = logging.ERROR,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an async catch-site exception with PII stripped from the message.

    Use at any async catch site where ``str(exc)`` could echo user input
    (LLM provider 4xx, ``inner_step()`` errors wrapping LLM responses,
    etc.). The helper sends ``str(exc)`` to ispy-pii and emits the
    redacted version — the error reason stays in Splunk for
    debuggability, with PII tokens scrubbed.

    The emitted log line is::

        <fmt % args> error_type=<type> msg=<masked str(exc)>
        <frames-only traceback ending in ExceptionType>

    The ispy-pii network call happens inside this coroutine, so the
    request thread cooperates with the event loop via ``await`` — the
    masking never blocks other concurrent requests on the same worker.
    This matches the rule in CLAUDE.md (only 5 sanctioned sync
    touchpoints exist; ispy-pii is not one of them).

    ``extra``: optional dict of structured fields to attach to the log
    record (Splunk indexes these as searchable fields directly, without
    relying on key=value auto-extraction from the message body). Use
    for IDs, structural shape, counts, etc. Do NOT include raw
    ``str(exc)``-derived values here — that's the leak this helper
    exists to prevent.

    The helper injects two structured fields automatically alongside
    the caller's ``extra``: ``error_type`` (the exception class name)
    and ``error`` (the masked exception message). This gives Splunk
    dashboards a structured field to key on — symmetric with the
    ECMS helper in ``common.pii.log_error_strip_pii``. Caller-supplied
    fields take precedence on name collision.

    Yields the event loop for up to ``MIRIX_ISPY_PII_TIMEOUT_MS``
    waiting on ispy-pii. Do not call from a hot path.
    """
    # Belt-and-suspenders: every operation here is theoretically capable
    # of raising (custom __str__, malformed exception chain, broken log
    # handler), and a logging path that re-enters its own caller's
    # exception handler is a deadlock waiting to happen. Wrap the whole
    # body and degrade silently on any failure — the alternative is
    # swallowing the original exception entirely.
    try:
        masked = await _mask_async(str(exc))
        tb = safe_traceback(exc)
        combined_extra: Dict[str, Any] = {
            "error_type": type(exc).__name__,
            "error": masked,
        }
        if extra:
            combined_extra.update(extra)
        log.log(
            level,
            fmt + " error_type=%s msg=%s\n%s",
            *args,
            type(exc).__name__,
            masked,
            tb,
            extra=combined_extra,
        )
    except Exception:
        # Last-ditch fallback. Use the stdlib logger directly with the
        # safest possible format string. If even THIS raises, there's
        # nothing more we can do without re-entering.
        try:
            log.log(level, fmt + " (mask helper failed)", *args)
        except Exception:
            pass
