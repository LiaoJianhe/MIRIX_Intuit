"""Centralized error classification and whole-step retry policy.

VEPAGE-1091 / Phase A. This module is the single source of truth for what
counts as a Transient error (worth retrying) vs a Permanent error (don't
retry, write status='failed' and ack). The same classify() function is
used by Layer 1 (inline LLM retry in _get_ai_reply) and Layer 2 (whole-step
retry around agent.step), so the contract is consistent across both.

Runtimes (in-memory, vanilla Kafka, Numaflow) all call process_with_policy()
and translate the returned Outcome to their own ack/redeliver mechanism.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

from mirix.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMServerError,
    LLMUnprocessableEntityError,
)
from mirix.settings import settings

logger = logging.getLogger(__name__)


class Bucket(str, Enum):
    """Two-state error model. See VEPAGE-1091 design doc."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"


class OutcomeKind(str, Enum):
    COMPLETED = "completed"
    PERMANENT_FAILURE = "permanent_failure"
    TRANSIENT_EXHAUSTED = "transient_exhausted"


@dataclass(frozen=True)
class Outcome:
    """Result of process_with_policy. Runtimes translate this into ack/raise."""

    kind: OutcomeKind
    cause: Optional[BaseException] = None
    bucket: Optional[Bucket] = None


# Exception class → Bucket. Subclass matches use isinstance, so more
# specific entries should still come first by convention even though the
# classifier walks the tuple in order.
_PERMANENT_TYPES: tuple[type[BaseException], ...] = (
    LLMUnprocessableEntityError,  # 422 — LXS Risk Screening (the bug)
    LLMBadRequestError,           # 400
    LLMAuthenticationError,       # 401
    LLMPermissionDeniedError,     # 403
)

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    LLMRateLimitError,    # 429
    LLMServerError,       # 5xx (incl. 424 dependency timeout)
    LLMConnectionError,   # network blip
)

# One-shot warning per unknown exception class so we triage and classify it
# later without flooding logs.
_unknown_warned: set[type[BaseException]] = set()


def classify(exc: BaseException) -> Bucket:
    """Map an exception to a Bucket.

    Unknown exception classes default to TRANSIENT — we'd rather waste a
    redelivery than silently swallow a bug. First sight of an unknown class
    logs a warning so it can be added to one of the explicit lists.
    """
    if isinstance(exc, _PERMANENT_TYPES):
        return Bucket.PERMANENT
    if isinstance(exc, _TRANSIENT_TYPES):
        return Bucket.TRANSIENT
    exc_type = type(exc)
    if exc_type not in _unknown_warned:
        _unknown_warned.add(exc_type)
        logger.warning(
            "error_policy: unmapped exception class %s defaulted to Transient — add to error_policy._PERMANENT_TYPES or _TRANSIENT_TYPES",
            exc_type.__name__,
        )
    return Bucket.TRANSIENT


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with jitter, capped. attempt is 0-indexed."""
    raw = min(cap, base * (2**attempt))
    # Full jitter halves the worst case but keeps the cap as the ceiling.
    return raw * (0.5 + 0.5 * random.random())


async def process_with_policy(
    run_step: Callable[[], Awaitable[object]],
    *,
    memory_source_id: Optional[str] = None,
    on_permanent: Optional[Callable[[str, str, BaseException], Awaitable[None]]] = None,
) -> Outcome:
    """Run agent.step() under Layer 2 policy.

    run_step is a zero-argument async callable that invokes agent.step(...)
    with whatever caller-specific arguments it needs. Keeping it opaque here
    means this module doesn't depend on Agent.

    on_permanent (optional) is called with (memory_source_id, error_message, exc)
    when a Permanent error is seen, so the caller can write status='failed' +
    error_message to the memory_sources row. Decoupled to avoid a circular
    import on MemorySourceManager.

    Returns Outcome describing what happened. The caller's runtime layer
    decides what ack/raise mechanism to use.
    """
    max_attempts = max(1, settings.whole_step_retry_max_attempts)
    base = settings.whole_step_retry_base_seconds
    cap = settings.whole_step_retry_max_delay

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts + 1):  # initial try + max_attempts retries
        try:
            await run_step()
            return Outcome(kind=OutcomeKind.COMPLETED)
        except Exception as exc:
            bucket = classify(exc)
            last_exc = exc
            if bucket is Bucket.PERMANENT:
                error_message = f"{type(exc).__name__}: {exc}"
                logger.info(
                    "error_policy: permanent failure (%s) on memory_source=%s — %s",
                    type(exc).__name__,
                    memory_source_id,
                    exc,
                )
                if on_permanent is not None and memory_source_id is not None:
                    try:
                        await on_permanent(memory_source_id, error_message, exc)
                    except Exception:
                        logger.exception(
                            "error_policy: on_permanent callback failed for memory_source=%s",
                            memory_source_id,
                        )
                return Outcome(kind=OutcomeKind.PERMANENT_FAILURE, cause=exc, bucket=bucket)
            # Transient — sleep and retry if budget remains.
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
            # Budget exhausted.
            logger.warning(
                "error_policy: transient retries exhausted on memory_source=%s after %d attempts: %s",
                memory_source_id,
                max_attempts + 1,
                exc,
            )
            return Outcome(kind=OutcomeKind.TRANSIENT_EXHAUSTED, cause=exc, bucket=Bucket.TRANSIENT)

    # Unreachable but keeps the type checker happy.
    return Outcome(kind=OutcomeKind.TRANSIENT_EXHAUSTED, cause=last_exc, bucket=Bucket.TRANSIENT)
