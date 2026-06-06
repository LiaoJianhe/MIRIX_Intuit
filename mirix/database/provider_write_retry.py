"""Retry + classification helpers for relational-provider writes.

The provider boundary in ECMS translates each SDK exception into one of three
MIRIX-owned types (`ProviderTransientError`, `ProviderPermanentError`,
`ProviderConflictError`) — see VEPAGE-1251 §5.6. With that translation in
place, classification here is plain `isinstance`.

For transition while the boundary is being wired up, this module ALSO retains
the legacy structural detection (status_code / error_code / class name) so
provider calls that have not yet been wrapped still work. New translations
should always go through the boundary; the structural fallback is a safety
net, not a target.

Three failure classes:

* **Conflict** — unique-constraint violation. Caller treats as no-op
  (idempotent re-write). Detected via `isinstance(ProviderConflictError)` or
  legacy structural hints.
* **Transient** — 5xx, 429, connection/request timeout. Retried with
  exponential backoff. On exhausted budget the propagated exception is
  tagged with the `__mirix_inner_exhausted__` marker so the whole-step
  policy does not re-retry it (kills the inner × whole-step multiplication).
* **Permanent** — anything else (auth, unknown entity, malformed request).
  Raised immediately, no retry.
"""

import asyncio
from typing import Awaitable, Callable, TypeVar

from mirix.errors import (
    ProviderConflictError,
    ProviderPermanentError,
    ProviderTransientError,
)
from mirix.log import get_logger
from mirix.queue.error_policy import mark_inner_exhausted

logger = get_logger(__name__)

T = TypeVar("T")

# Legacy structural detection — kept ONLY for code paths that have not yet
# been migrated to translate SDK exceptions at the boundary. Once every
# provider call site re-raises Provider*Error, these constants and helpers
# can be removed.
_TRANSIENT_STATUS_CODES = frozenset({429, 502, 503, 504})
_TRANSIENT_CLASS_HINTS = ("Timeout", "ServerError", "Throttle")
_CONFLICT_HINTS = ("CONFLICT", "DUPLICATE", "UNIQUE")


def _exc_status_code(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)


def is_conflict(exc: BaseException) -> bool:
    """True if the exception is a duplicate-key / unique-constraint violation.

    Four call sites depend on this:
    `source_message_manager.py:217` (L1 dedup), `user_manager.py`,
    `client_manager.py`, `memory_citation_manager.py` (L3 dedup). Each
    treats a conflict as an idempotent no-op (`continue` / `if not is_conflict(exc): raise`).

    Preferred shape (post-translation): the provider boundary raises
    `ProviderConflictError`; this returns True by isinstance.

    Legacy shape (during transition): structural detection on
    status_code 409, error_code/message hints, or the "violates a database
    constraint" + `uq_`/`_pkey` parse path some IPSR backends emit.
    """
    if isinstance(exc, ProviderConflictError):
        return True

    status = _exc_status_code(exc)
    if status == 409:
        return True
    code = (getattr(exc, "error_code", "") or "").upper()
    msg = (str(exc) or "").upper()
    if any(hint in code or hint in msg for hint in _CONFLICT_HINTS):
        return True
    return "VIOLATES A DATABASE CONSTRAINT" in msg and ("UQ_" in msg or "_PKEY" in msg)


def is_transient(exc: BaseException) -> bool:
    """True if the exception is worth retrying.

    Preferred: `isinstance(exc, ProviderTransientError)`.
    Legacy: structural matching on status_code / class-name hints / asyncio
    TimeoutError until every provider call is wrapped.
    """
    if isinstance(exc, ProviderTransientError):
        return True
    if isinstance(exc, ProviderPermanentError) or isinstance(exc, ProviderConflictError):
        return False
    if isinstance(exc, asyncio.TimeoutError):
        return True
    status = _exc_status_code(exc)
    if status in _TRANSIENT_STATUS_CODES:
        return True
    name = type(exc).__name__
    return any(hint in name for hint in _TRANSIENT_CLASS_HINTS)


async def retry_transient(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    op: str,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
) -> T:
    """Run ``coro_factory()`` with retry on transient errors.

    Conflicts and permanent errors are raised immediately. Caller is
    responsible for catching ``is_conflict``-classified exceptions.

    On exhausted budget, the propagated exception is tagged with the
    inner-exhausted marker so `process_with_policy` does NOT add another
    whole-step retry cycle on top (VEPAGE-1251 §5.6).
    """
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last = exc
            if is_conflict(exc) or not is_transient(exc):
                raise
            if attempt == max_attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "%s transient failure (attempt %d/%d): %s — retrying in %.1fs",
                op,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    assert last is not None
    # The inner tier exhausted its budget — tag so the whole-step loop
    # doesn't re-retry the same provider call.
    raise mark_inner_exhausted(last)
