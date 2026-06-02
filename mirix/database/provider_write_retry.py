"""Retry + classification helpers for relational-provider writes.

The provider interface is duck-typed (no IPSR import in MIRIX), so exception
classification is structural — we look at attributes (status_code, error_code,
class name) rather than isinstance checks against IPSR-specific classes.

Three failure classes:

* **Conflict** — unique-constraint violation. Caller treats as no-op (idempotent
  re-write). Detected via 4xx status code or "conflict"/"duplicate" hint in
  error_code/message. Returned, not raised.
* **Transient** — 5xx, 429, connection/request timeout. Retried with exponential
  backoff. After max_attempts, raised to the caller (which logs WARNING + emits
  a skip-span and swallows).
* **Permanent** — anything else (auth, unknown entity, malformed request).
  Raised immediately, no retry.
"""

import asyncio
from typing import Awaitable, Callable, TypeVar

from mirix.log import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_TRANSIENT_STATUS_CODES = frozenset({429, 502, 503, 504})
_TRANSIENT_CLASS_HINTS = ("Timeout", "ServerError", "Throttle")
_CONFLICT_HINTS = ("CONFLICT", "DUPLICATE", "UNIQUE")


def _exc_status_code(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)


def is_conflict(exc: BaseException) -> bool:
    """True if the exception looks like a unique-constraint violation.

    We don't have a single canonical IPSR error code for unique conflicts yet;
    treat 409 plus any error_code/message containing CONFLICT/DUPLICATE/UNIQUE
    as a conflict. Conservative — false positives just skip a retry, not data loss.

    IPS-Relational does not surface unique-index violations as a 409 nor with a
    CONFLICT/DUPLICATE/UNIQUE token. It raises ``BadRequestError`` (no
    ``status_code``) with the message
    ``"Client data violates a database constraint:  uq_<index>"`` and an ambiguous
    ``DATABASE_CONSTRAINT_VIOLATION`` error_code that is *also* used for
    column-shape mismatches. We therefore key off the unique-index name prefix
    (``uq_``) which only appears for genuine uniqueness conflicts — the
    column-shape variant of the same error carries no constraint name.

    Primary-key collisions (``<table>_pkey``) are the same class of duplicate-key
    conflict; PG/asyncpg surfaces them as ``UniqueConstraintViolationError`` with
    a message containing ``"users_pkey"`` etc. Match ``_PKEY`` too. The
    column-shape variant still carries no constraint name, so this does not
    widen the false-positive surface.
    """
    status = _exc_status_code(exc)
    if status == 409:
        return True
    code = (getattr(exc, "error_code", "") or "").upper()
    msg = (str(exc) or "").upper()
    if any(hint in code or hint in msg for hint in _CONFLICT_HINTS):
        return True
    return "VIOLATES A DATABASE CONSTRAINT" in msg and ("UQ_" in msg or "_PKEY" in msg)


def is_transient(exc: BaseException) -> bool:
    """True if the exception is worth retrying."""
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
    raise last
