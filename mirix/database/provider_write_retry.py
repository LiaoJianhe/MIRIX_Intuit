"""Retry + classification helpers for relational-provider writes.

Each registered provider catches its SDK exceptions at the boundary and
re-raises one of the MIRIX-owned types (`ProviderTransientError`,
`ProviderPermanentError`, `ProviderConflictError`, `ProviderNotFoundError`).
With translation in place, classification here is **pure isinstance** —
no structural detection, no string-matching, no SDK class imports in
core.

Three failure classes the manager call sites care about:

* **Conflict** — unique-constraint violation. Caller treats as no-op
  (idempotent re-write). `isinstance(exc, ProviderConflictError)`.
* **Transient** — 5xx, 429, timeout. Retried with exponential backoff.
  On exhausted budget the propagated exception is tagged with the
  `__mirix_inner_exhausted__` marker so the whole-step policy does not
  re-retry it (kills the inner × whole-step multiplication).
* **Permanent** — anything else (auth, unknown entity, malformed request).
  Raised immediately, no retry.

KNOWN STACKING (follow-up): A registered provider may have its own
inner-retry tier (e.g. one that retries 5xx/429 before exceptions reach
this helper), so callers that wrap a provider call in `retry_transient`
can get N×M attempts. The inner-exhausted marker short-circuits the
second tier, but the cleaner long-term fix is to delete this helper and
let the provider's own retry tier be the only one. Tracked separately.
"""

import asyncio
from typing import Awaitable, Callable, TypeVar

from mirix.errors import (
    ProviderConflictError,
    ProviderPermanentError,
    ProviderTransientError,
)
from mirix.log import get_logger
from mirix.queue.error_policy import is_inner_exhausted, mark_inner_exhausted

logger = get_logger(__name__)

T = TypeVar("T")


def is_conflict(exc: BaseException) -> bool:
    """True if the exception is a duplicate-key / unique-constraint violation.

    Four call sites depend on this: `source_message_manager.py` (L1 dedup),
    `user_manager.py`, `client_manager.py`, `memory_citation_manager.py`
    (L3 dedup). Each treats a conflict as an idempotent no-op
    (`continue` / `if not is_conflict(exc): raise`).

    Post-translation: pure isinstance against the MIRIX-owned type.
    """
    return isinstance(exc, ProviderConflictError)


def is_transient(exc: BaseException) -> bool:
    """True if the exception is worth retrying.

    Post-translation: pure isinstance against the MIRIX-owned type. Conflict
    and Permanent shapes always short-circuit to False; only
    ProviderTransientError is retryable.
    """
    if isinstance(exc, (ProviderPermanentError, ProviderConflictError)):
        return False
    return isinstance(exc, ProviderTransientError)


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
    whole-step retry cycle on top (kills the inner × whole-step
    multiplication).

    If the exception arrives already inner-exhausted from a deeper tier
    (e.g. the provider's own `_retry_transient` already retried 3×), this
    helper skips its own retry loop and propagates immediately — avoids
    the stacking N×M multiplication.
    """
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last = exc
            if is_conflict(exc) or not is_transient(exc):
                raise
            if is_inner_exhausted(exc):
                # Deeper tier already exhausted its budget on this exact
                # failure. Don't retry; just propagate the already-tagged
                # exception so process_with_policy sees the verdict too.
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
