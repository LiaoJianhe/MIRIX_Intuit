"""Centralized error classification and whole-step retry policy.

Single source of truth for what counts as a Transient error (worth retrying)
vs a Permanent error (don't retry). The same classify() function is used by
the inline LLM retry in _get_ai_reply and the whole-step retry around
agent.step, so the contract is consistent.

Callers run an agent step under process_with_policy; the returned Outcome
tells the consumer loop whether to ack the message or let it be redelivered.

CONTRACT (VEPAGE-1251):

* classify(exc) is PURE-ISINSTANCE against MIRIX-owned types — no
  string-matching. The three Provider*Error types are how ECMS providers
  speak to this module across the ECMS -> MIRIX dependency direction:
  the boundary translates SDK exceptions; classify maps the result.

* mark_inner_exhausted(exc) tags a propagated exception so
  process_with_policy returns TRANSIENT_EXHAUSTED after exactly one
  attempt instead of running the full backoff cycle. Inner-retry tiers
  (_get_ai_reply, provider_write_retry.retry_transient) use this to kill
  the inner x whole-step multiplication.

* Origin-split for pure-Python bug shapes: AttributeError / KeyError /
  TypeError / IndexError / NameError with no provider frame in the
  traceback classify PERMANENT (the VEPAGE-1228 shape). With a provider
  frame on the chain, fall back to the historical Transient default.
"""

from __future__ import annotations

import asyncio
import logging
import random
import types
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

from sqlalchemy.exc import DataError, DBAPIError, IntegrityError, OperationalError

from mirix.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMChainingExhaustedError,
    LLMConnectionError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMServerError,
    LLMUnprocessableEntityError,
    ProviderPermanentError,
    ProviderTransientError,
)
from mirix.settings import settings

logger = logging.getLogger(__name__)


class Bucket(str, Enum):
    """Two-state error model: retry or don't."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"


class SaveOutcome(str, Enum):
    """Terminal verdict for one save attempt.

    Same enum used by `process_with_policy` (the policy's verdict) AND by
    `MemorySourceManager.finalize_source` (what gets recorded on the source).
    One vocabulary, one set of values — the seam that VEPAGE-1250 will use
    to diversify column writes per outcome.
    """

    SUCCESS = "success"
    PERMANENT_FAILURE = "permanent_failure"
    TRANSIENT_EXHAUSTED = "transient_exhausted"


@dataclass(frozen=True)
class Outcome:
    """Result of process_with_policy. Consumers translate this into ack/raise."""

    kind: SaveOutcome
    cause: Optional[BaseException] = None
    bucket: Optional[Bucket] = None


# Subclass matches use isinstance — order is not significant for correctness,
# but kept stable for readability.
#
# Provider* types are MIRIX-owned and re-raised by the ECMS provider boundary
# from SDK-specific exceptions. This keeps `classify()` pure isinstance and
# respects the dependency direction (ECMS -> MIRIX): MIRIX cannot import the
# IPS-R / IPS-Search / SDK exception classes, so the seam that *can* (the
# provider) is responsible for the translation. See VEPAGE-1251 design §5.6.
_PERMANENT_TYPES: tuple[type[BaseException], ...] = (
    LLMUnprocessableEntityError,  # 422
    LLMBadRequestError,  # 400
    LLMAuthenticationError,  # 401
    LLMPermissionDeniedError,  # 403
    LLMChainingExhaustedError,  # meta-agent LLM emitted malformed calls past budget
    ProviderPermanentError,  # provider boundary: auth, bad request, etc.
    IntegrityError,  # SQLAlchemy: constraint/unique/foreign-key violations
    DataError,  # SQLAlchemy: bad input shape (invalid syntax, range)
)

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    LLMRateLimitError,  # 429
    LLMServerError,  # 5xx
    LLMConnectionError,
    ProviderTransientError,  # provider boundary: 429, 5xx, timeout
    OperationalError,  # SQLAlchemy: connection / serialization / lock
    DBAPIError,  # SQLAlchemy generic DBAPI failure
    asyncio.TimeoutError,  # I/O stall worth one retry
)

# Pure-Python bug shapes. When an exception of one of these types reaches
# `classify()` without a provider/SDK frame in its traceback, treat as
# PERMANENT (it'll never succeed on retry). This is the origin-split for
# the VEPAGE-1228 AttributeError shape — see design §5.4. With a provider
# frame in the traceback, the exception is treated as transient (the
# provider might have wrapped a real infra failure).
_PYTHON_BUG_TYPES: tuple[type[BaseException], ...] = (
    AttributeError,
    KeyError,
    TypeError,
    IndexError,
    NameError,
)

# Frame module-name fragments that signal "this exception came from
# provider / database / SDK code, not pure-Python agent logic." If any
# frame on the traceback matches, an otherwise-bug-shaped exception is
# treated as transient (the provider might have crashed mid-call).
_PROVIDER_FRAME_HINTS: tuple[str, ...] = (
    "ipsr",
    "ipss",
    "ipsrclient",
    "ipssearchclient",
    "sqlalchemy",
    "asyncpg",
    "psycopg",
    "httpx",
    "openai",
    "anthropic",
)


def _traceback_has_provider_frame(tb: Optional[types.TracebackType]) -> bool:
    """Walk the traceback looking for a frame in provider/SDK code. If any
    such frame is on the chain, the exception is not purely an agent-logic
    bug and should keep the default Transient classification."""
    cur = tb
    depth = 0
    while cur is not None and depth < 64:
        module_path = (cur.tb_frame.f_globals.get("__name__") or "").lower()
        if any(hint in module_path for hint in _PROVIDER_FRAME_HINTS):
            return True
        cur = cur.tb_next
        depth += 1
    return False


# Inner-exhausted marker. The marker is an attribute on the propagated
# exception — NOT a new exception type — so callers downstream can still
# `isinstance` against the original LLM*/Provider*/SQLAlchemy class.
#
# When `_get_ai_reply` (or any inner-retry tier) exhausts its budget on a
# transient, it tags the propagated exception so `process_with_policy` knows
# the dependency was already retried in place and the whole-step loop must
# not retry it again. Removes the LLM inner (3) × whole-step (3) = 9
# multiplication; the same shape applies to ORM / provider tiers.
_INNER_EXHAUSTED_ATTR = "__mirix_inner_exhausted__"


def mark_inner_exhausted(exc: BaseException) -> BaseException:
    """Tag exc so process_with_policy treats it as already-retried.

    Returns the exception so call sites can chain:
        raise mark_inner_exhausted(exc)

    Idempotent — re-tagging the same exception is a no-op.
    """
    try:
        setattr(exc, _INNER_EXHAUSTED_ATTR, True)
    except Exception:
        # Some C-extension exceptions reject arbitrary attrs. Treat as a
        # missed optimization rather than a failure — the whole-step loop
        # will just run its normal budget.
        pass
    return exc


def is_inner_exhausted(exc: BaseException) -> bool:
    """Read the inner-exhausted marker. False if the attribute is missing or
    the exception doesn't accept attrs."""
    return bool(getattr(exc, _INNER_EXHAUSTED_ATTR, False))

# One-shot warning per unknown exception class so the log doesn't flood when
# a new error type starts appearing in production.
_unknown_warned: set[type[BaseException]] = set()


def format_exc_chain(exc: BaseException, max_depth: int = 8) -> str:
    """Render an exception's __cause__ chain as a compact one-line string.

    VEPAGE-1157 instrumentation. Used to capture wrapped-exception cascades
    in log lines so we can see the original type even when intermediate code
    re-raised with `raise X(...) from y`.

    Example output: "RuntimeError: foo <- LLMUnprocessableEntityError: 422"
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    depth = 0
    while current is not None and depth < max_depth and id(current) not in seen:
        parts.append(f"{type(current).__name__}: {current}")
        seen.add(id(current))
        current = current.__cause__
        depth += 1
    return " <- ".join(parts)


def classify(exc: BaseException) -> Bucket:
    """Map an exception to a Bucket.

    Walks exc.__cause__ when the outer type is unmapped so that a Permanent
    exception wrapped in another type (e.g. `raise RuntimeError(...) from
    LLMUnprocessableEntityError`) still classifies as Permanent. Without this,
    any code on the path between the LLM client and process_with_policy that
    rewraps exceptions silently turns Permanent into Transient and re-creates
    the 422 cascade.

    Unknown exception classes default to TRANSIENT: a wasted redelivery is
    preferable to silently swallowing a bug. The first occurrence of each
    unmapped class emits a warning so the explicit lists can be updated.
    """
    # Walk the cause chain so wrapped permanent/transient exceptions are
    # classified by their original type. Bounded to avoid pathological chains.
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    depth = 0
    while current is not None and depth < 8 and id(current) not in seen:
        if isinstance(current, _PERMANENT_TYPES):
            logger.info(
                "error_policy.classify: PERMANENT (matched %s at depth=%d) — chain: %s",
                type(current).__name__,
                depth,
                format_exc_chain(exc),
            )
            return Bucket.PERMANENT
        if isinstance(current, _TRANSIENT_TYPES):
            logger.info(
                "error_policy.classify: TRANSIENT (matched %s at depth=%d) — chain: %s",
                type(current).__name__,
                depth,
                format_exc_chain(exc),
            )
            return Bucket.TRANSIENT
        seen.add(id(current))
        current = current.__cause__
        depth += 1

    # Origin-split for pure-Python bug shapes (VEPAGE-1251 §5.4). If the
    # exception is a recognized bug class AND its traceback has no
    # provider/SDK frame, treat as permanent — retrying a deterministic
    # AttributeError just burns redeliveries. With a provider frame on
    # the chain, fall through to the historical Transient default (the
    # provider might have wrapped real infra trouble).
    if isinstance(exc, _PYTHON_BUG_TYPES) and not _traceback_has_provider_frame(
        exc.__traceback__
    ):
        logger.info(
            "error_policy.classify: PERMANENT (pure-Python bug shape %s, no provider "
            "frame in traceback) — chain: %s",
            type(exc).__name__,
            format_exc_chain(exc),
        )
        return Bucket.PERMANENT

    exc_type = type(exc)
    if exc_type not in _unknown_warned:
        _unknown_warned.add(exc_type)
        logger.warning(
            "error_policy: unmapped exception class %s defaulted to Transient — "
            "add to _PERMANENT_TYPES or _TRANSIENT_TYPES",
            exc_type.__name__,
        )
    return Bucket.TRANSIENT


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter, capped. attempt is 0-indexed."""
    raw = min(cap, base * (2**attempt))
    return raw * (0.5 + 0.5 * random.random())


async def process_with_policy(
    run_step: Callable[[], Awaitable[object]],
    *,
    memory_source_id: Optional[str] = None,
    on_permanent: Optional[Callable[[str, str, BaseException], Awaitable[None]]] = None,
) -> Outcome:
    """Run an agent step under the whole-step retry policy.

    run_step is a zero-argument async callable that invokes agent.step(...)
    with caller-specific arguments. Keeping it opaque means this module does
    not depend on Agent.

    on_permanent, when provided, is invoked with (memory_source_id, error_message,
    exc) the first time a Permanent classification is reached. It is purely
    advisory — its exceptions are caught and logged. Callers can use it to
    emit additional telemetry. No retry decision depends on it.

    Returns an Outcome; the caller's consumer loop decides ack vs raise.
    """
    max_attempts = max(1, settings.whole_step_retry_max_attempts)
    base = settings.whole_step_retry_base_seconds
    cap = settings.whole_step_retry_max_delay

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts + 1):  # initial attempt + max_attempts retries
        try:
            await run_step()
            return Outcome(kind=SaveOutcome.SUCCESS)
        except Exception as exc:
            bucket = classify(exc)
            last_exc = exc
            if bucket is Bucket.PERMANENT:
                error_message = f"{type(exc).__name__}: {exc}"
                logger.info(
                    "error_policy: permanent failure (%s) on memory_source=%s — %s " "— chain: %s",
                    type(exc).__name__,
                    memory_source_id,
                    exc,
                    format_exc_chain(exc),
                )
                if on_permanent is not None and memory_source_id is not None:
                    try:
                        await on_permanent(memory_source_id, error_message, exc)
                    except Exception:
                        logger.exception(
                            "error_policy: on_permanent callback failed for memory_source=%s",
                            memory_source_id,
                        )
                return Outcome(kind=SaveOutcome.PERMANENT_FAILURE, cause=exc, bucket=bucket)
            # An inner-retry tier (LLM, ORM, provider boundary) has already
            # exhausted its budget on this exact failure — repeating the whole
            # step is wasted work that just re-runs persist/extract/sub-agent
            # spawns. Verdict as TRANSIENT_EXHAUSTED immediately. This kills
            # the inner × whole-step multiplication (VEPAGE-1251 §5.6).
            if is_inner_exhausted(exc):
                logger.info(
                    "error_policy: inner-exhausted transient (%s) on memory_source=%s — "
                    "skipping whole-step retry — chain: %s",
                    type(exc).__name__,
                    memory_source_id,
                    format_exc_chain(exc),
                )
                return Outcome(
                    kind=SaveOutcome.TRANSIENT_EXHAUSTED, cause=exc, bucket=Bucket.TRANSIENT
                )
            if attempt < max_attempts:
                sleep_s = _backoff_seconds(attempt, base, cap)
                logger.info(
                    "error_policy: transient failure (%s) on memory_source=%s, attempt %d/%d, sleeping %.1fs",
                    type(exc).__name__,
                    memory_source_id,
                    attempt + 1,
                    max_attempts + 1,
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
                continue
            logger.warning(
                "error_policy: transient retries exhausted on memory_source=%s after %d attempts: %s " "— chain: %s",
                memory_source_id,
                max_attempts + 1,
                exc,
                format_exc_chain(exc),
            )
            return Outcome(kind=SaveOutcome.TRANSIENT_EXHAUSTED, cause=exc, bucket=Bucket.TRANSIENT)

    # Defensive — the loop above either returns or continues. This line satisfies
    # the type checker for the case where max_attempts is somehow exceeded
    # without a return path being taken.
    return Outcome(kind=SaveOutcome.TRANSIENT_EXHAUSTED, cause=last_exc, bucket=Bucket.TRANSIENT)


async def dispatch_save(
    run_step: Callable[[], Awaitable[object]],
    *,
    memory_source_id: Optional[str],
) -> Outcome:
    """One save attempt, classified and finalized — the shared post-policy
    handler for ALL three run modes (numaflow, kafka, in-memory).

    Flow:
      1. Run the save under process_with_policy (classify + bounded retry).
      2. Route the resulting Outcome through the single finalize chokepoint
         (`MemorySourceManager.finalize_source`).

    No conscious redelivery. TRANSIENT_EXHAUSTED is dead-letter behavior in
    all modes — the in-process retry budget already covered the transient
    case, and consuming-side redelivery is reserved for *process-death*
    cases (broker semantics on un-ack'd messages), not for failures we
    classified.

    Today (boolean schema) finalize_source records the SaveOutcome in a log
    line and writes `processing_complete=True` regardless of outcome.
    VEPAGE-1250 will diversify the actual column writes per outcome by
    touching only `finalize_source` — this dispatcher doesn't change.

    Step() no longer finalizes internally; ALL finalize calls flow through
    this function. (VEPAGE-1251 Option B.)
    """
    outcome = await process_with_policy(run_step, memory_source_id=memory_source_id)

    if memory_source_id is not None:
        # Late import to avoid cycle (memory_source_manager doesn't import
        # error_policy at runtime).
        from mirix.services.memory_source_manager import MemorySourceManager

        await MemorySourceManager().finalize_source(memory_source_id, outcome.kind)

    return outcome
