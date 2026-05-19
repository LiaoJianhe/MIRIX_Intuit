"""Centralized error classification and whole-step retry policy.

Single source of truth for what counts as a Transient error (worth retrying)
vs a Permanent error (don't retry). The same classify() function is used by
the inline LLM retry in _get_ai_reply and the whole-step retry around
agent.step, so the contract is consistent.

Callers run an agent step under process_with_policy; the returned Outcome
tells the consumer loop whether to ack the message or let it be redelivered.
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
    """Two-state error model: retry or don't."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"


class OutcomeKind(str, Enum):
    COMPLETED = "completed"
    PERMANENT_FAILURE = "permanent_failure"
    TRANSIENT_EXHAUSTED = "transient_exhausted"


@dataclass(frozen=True)
class Outcome:
    """Result of process_with_policy. Consumers translate this into ack/raise."""

    kind: OutcomeKind
    cause: Optional[BaseException] = None
    bucket: Optional[Bucket] = None


# Subclass matches use isinstance — order is not significant for correctness,
# but kept stable for readability.
_PERMANENT_TYPES: tuple[type[BaseException], ...] = (
    LLMUnprocessableEntityError,  # 422
    LLMBadRequestError,           # 400
    LLMAuthenticationError,       # 401
    LLMPermissionDeniedError,     # 403
)

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    LLMRateLimitError,    # 429
    LLMServerError,       # 5xx
    LLMConnectionError,
)

# One-shot warning per unknown exception class so the log doesn't flood when
# a new error type starts appearing in production.
_unknown_warned: set[type[BaseException]] = set()


def classify(exc: BaseException) -> Bucket:
    """Map an exception to a Bucket.

    Unknown exception classes default to TRANSIENT: a wasted redelivery is
    preferable to silently swallowing a bug. The first occurrence of each
    unmapped class emits a warning so the explicit lists can be updated.
    """
    if isinstance(exc, _PERMANENT_TYPES):
        return Bucket.PERMANENT
    if isinstance(exc, _TRANSIENT_TYPES):
        return Bucket.TRANSIENT
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
                "error_policy: transient retries exhausted on memory_source=%s after %d attempts: %s",
                memory_source_id,
                max_attempts + 1,
                exc,
            )
            return Outcome(kind=OutcomeKind.TRANSIENT_EXHAUSTED, cause=exc, bucket=Bucket.TRANSIENT)

    # Defensive — the loop above either returns or continues. This line satisfies
    # the type checker for the case where max_attempts is somehow exceeded
    # without a return path being taken.
    return Outcome(kind=OutcomeKind.TRANSIENT_EXHAUSTED, cause=last_exc, bucket=Bucket.TRANSIENT)
