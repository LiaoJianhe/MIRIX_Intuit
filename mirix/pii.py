"""ispy-pii client + safe error logging helper for MIRIX.

Use :func:`log_error_strip_pii_sync` at any sync catch site where the
exception's ``str(e)`` could echo user input (LLM provider 4xx response
bodies, MIRIX-internal errors that wrap user content, ``inner_step()``
failures whose wrapped ``LLMError`` carries the upstream provider
message). The helper sends ``str(e)`` to ispy-pii and emits the
redacted version, so the error reason is preserved in Splunk with PII
tokens (emails, SSNs, phone numbers, credit cards, etc.) scrubbed::

    try:
        ...
    except Exception as e:
        pii.log_error_strip_pii_sync(
            logger, "inner_step() failed: num_messages=%d", len(msgs), exc=e
        )
        raise

Why a sync helper (and not the async one ECMS uses)?

MIRIX catch sites — ``agent.py:inner_step``, the LLM clients'
``handle_llm_error`` methods — are sync (not coroutines). We can't
``await`` from there. The mask call happens via ``httpx.Client``
synchronously; the catch site blocks for ≤200ms (timeout). Acceptable
for an error path — these fire rarely. Do not call this helper from a
hot path.

The Langfuse trace path uses a separate, async-friendly seam (the
``mirix.observability.pii_mask`` masker registered as the Langfuse
``mask=`` callback). The two paths are intentionally separate: error
logs go to Splunk via stdlib logging; trace attributes go to Langfuse
via the SDK's flush thread.

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


def build_ispy_payload(text: str) -> dict:
    """Construct the ispy-pii v2/analyze request body.

    Shared between the sync log helper here and the Langfuse mask
    callback in ``mirix.observability.pii_mask`` so a config change
    (sensitivity tier, etc.) lands in exactly one place.

    HIGHLY_SENSITIVE catches the strictest tier of detectors (SSN,
    credit-card, IBAN, financial IDs, PHI) — what we actually need on
    error logs and trace attributes that may echo financial / PHI
    user content.
    """
    return {
        "text": text,
        "format": "PLAIN_TEXT",
        "sensitivityLevel": "HIGHLY_SENSITIVE",
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


_sync_client: Optional[httpx.Client] = None


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(timeout=_timeout_seconds())
    return _sync_client


def _mask_sync(text: str) -> str:
    """Synchronous ispy-pii redaction. Returns the placeholder on any failure.

    Must never raise — sync catch sites rely on this to log without
    re-entering their own exception handler. The outer
    ``except Exception`` is deliberate.
    """
    if not text or not _enabled():
        return text
    try:
        resp = _get_sync_client().post(
            _endpoint(), json=build_ispy_payload(text), timeout=_timeout_seconds()
        )
        if resp.status_code != 200:
            return REDACTED_PLACEHOLDER
        return extract_redacted(resp.json())
    except Exception:
        return REDACTED_PLACEHOLDER


def log_error_strip_pii_sync(
    log: logging.Logger,
    fmt: str,
    *args: Any,
    exc: BaseException,
    level: int = logging.ERROR,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a sync catch-site exception with PII stripped from the message.

    Use at sync catch sites where ``str(exc)`` could echo user input
    (LLM provider 4xx, ``inner_step()`` errors wrapping LLM responses,
    etc.). The helper sends ``str(exc)`` to ispy-pii synchronously and
    emits the redacted version — the error reason stays in Splunk for
    debuggability, with PII tokens scrubbed.

    The emitted log line is::

        <fmt % args> error_type=<type> msg=<masked str(exc)>
        <frames-only traceback ending in ExceptionType>

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
    async helper in ECMS (``common.pii.log_error_strip_pii``).
    Caller-supplied fields take precedence on name collision.

    Blocks the calling thread for up to ``MIRIX_ISPY_PII_TIMEOUT_MS``
    waiting on ispy-pii. Do not call from a hot path.
    """
    # Belt-and-suspenders: every operation here is theoretically capable
    # of raising (custom __str__, malformed exception chain, broken log
    # handler), and a logging path that re-enters its own caller's
    # exception handler is a deadlock waiting to happen. Wrap the whole
    # body and degrade silently on any failure — the alternative is
    # swallowing the original exception entirely.
    try:
        masked = _mask_sync(str(exc))
        tb = safe_traceback(exc)
        # Inject error_type / error (masked) as structured fields so
        # Splunk dashboards can key on them. Caller-supplied keys win.
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
