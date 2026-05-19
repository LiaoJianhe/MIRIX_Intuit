"""Tests for mirix.queue.error_policy.

Covers:
- classify() across every Permanent and Transient class in the mapping table
- Unknown exceptions default to Transient and emit a one-shot warning
- process_with_policy() success / Permanent / Transient-retry / Transient-exhausted paths
- on_permanent callback invocation and error-message shape
- on_permanent failure does not mask the underlying Permanent outcome
"""

from __future__ import annotations

import pytest

from mirix.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMServerError,
    LLMUnprocessableEntityError,
)
from mirix.queue import error_policy as ep
from mirix.queue.error_policy import Bucket, Outcome, OutcomeKind, classify, process_with_policy


# ---------- classify() ----------


@pytest.mark.parametrize(
    "exc_cls",
    [
        LLMUnprocessableEntityError,  # 422
        LLMBadRequestError,           # 400
        LLMAuthenticationError,       # 401
        LLMPermissionDeniedError,     # 403
    ],
)
def test_classify_permanent(exc_cls):
    assert classify(exc_cls("boom")) is Bucket.PERMANENT


@pytest.mark.parametrize(
    "exc_cls",
    [
        LLMRateLimitError,    # 429
        LLMServerError,       # 5xx
        LLMConnectionError,   # network
    ],
)
def test_classify_transient(exc_cls):
    assert classify(exc_cls("boom")) is Bucket.TRANSIENT


def test_classify_unknown_defaults_to_transient(caplog):
    # Reset the one-shot dedup set so we can assert the warning fires.
    ep._unknown_warned.discard(RuntimeError)
    with caplog.at_level("WARNING", logger="mirix.queue.error_policy"):
        assert classify(RuntimeError("something weird")) is Bucket.TRANSIENT
    assert any("RuntimeError" in r.message for r in caplog.records)


def test_classify_unknown_warns_once_per_class(caplog):
    ep._unknown_warned.discard(KeyError)
    with caplog.at_level("WARNING", logger="mirix.queue.error_policy"):
        classify(KeyError("a"))
        classify(KeyError("b"))
    warnings = [r for r in caplog.records if "KeyError" in r.message]
    assert len(warnings) == 1, "expected one-shot warning per class"


# ---------- process_with_policy() ----------


async def test_process_with_policy_completed_path():
    calls = {"n": 0}

    async def run_step():
        calls["n"] += 1

    out = await process_with_policy(run_step, memory_source_id="src-ok")
    assert out.kind is OutcomeKind.COMPLETED
    assert out.cause is None
    assert calls["n"] == 1


async def test_process_with_policy_permanent_path_invokes_callback_and_short_circuits():
    perm_calls: list[tuple[str, str, str]] = []

    async def on_perm(source_id, error_message, exc):
        perm_calls.append((source_id, error_message, type(exc).__name__))

    async def run_step():
        raise LLMUnprocessableEntityError("content rejected")

    out = await process_with_policy(
        run_step,
        memory_source_id="src-422",
        on_permanent=on_perm,
    )
    assert out.kind is OutcomeKind.PERMANENT_FAILURE
    assert out.bucket is Bucket.PERMANENT
    assert isinstance(out.cause, LLMUnprocessableEntityError)
    # Single attempt — Permanent does not retry.
    assert perm_calls == [
        ("src-422", "LLMUnprocessableEntityError: content rejected", "LLMUnprocessableEntityError")
    ]


async def test_process_with_policy_permanent_callback_failure_is_swallowed():
    """If on_permanent raises, the Outcome is still PERMANENT_FAILURE.

    We must not let a logging/DB hiccup convert a permanent failure into a
    transient retry — that would re-create the cascade.
    """

    async def on_perm(source_id, error_message, exc):
        raise RuntimeError("DB hiccup writing status")

    async def run_step():
        raise LLMBadRequestError("400 schema")

    out = await process_with_policy(run_step, memory_source_id="src-cb", on_permanent=on_perm)
    assert out.kind is OutcomeKind.PERMANENT_FAILURE
    assert out.bucket is Bucket.PERMANENT


async def test_process_with_policy_transient_retry_then_success(monkeypatch):
    # Make sleep instant.
    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)

    attempts = {"n": 0}

    async def run_step():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise LLMRateLimitError("429 try again")

    out = await process_with_policy(run_step, memory_source_id="src-flaky")
    assert out.kind is OutcomeKind.COMPLETED
    assert attempts["n"] == 2


async def test_process_with_policy_transient_exhausted(monkeypatch):
    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)
    # Settings default: whole_step_retry_max_attempts=2 → 3 total attempts
    from mirix.settings import settings as svc_settings

    expected_total = svc_settings.whole_step_retry_max_attempts + 1

    attempts = {"n": 0}

    async def run_step():
        attempts["n"] += 1
        raise LLMRateLimitError("still 429")

    out = await process_with_policy(run_step, memory_source_id="src-tx")
    assert out.kind is OutcomeKind.TRANSIENT_EXHAUSTED
    assert out.bucket is Bucket.TRANSIENT
    assert isinstance(out.cause, LLMRateLimitError)
    assert attempts["n"] == expected_total


async def test_process_with_policy_unknown_exception_treated_as_transient(monkeypatch):
    """Unknown exceptions walk the Transient path, not the Permanent one.

    Guards against accidentally hiding bugs by classifying them as Permanent
    (which would silently ack the message and never retry).
    """
    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)

    perm_calls: list[str] = []

    async def on_perm(source_id, error_message, exc):
        perm_calls.append(source_id)

    async def run_step():
        raise ValueError("oops, a bug")

    out = await process_with_policy(run_step, memory_source_id="src-bug", on_permanent=on_perm)
    assert out.kind is OutcomeKind.TRANSIENT_EXHAUSTED
    assert perm_calls == [], "on_permanent must not fire for Transient outcomes"


# ---------- Outcome dataclass ----------


def test_outcome_is_frozen():
    out = Outcome(kind=OutcomeKind.COMPLETED)
    with pytest.raises(Exception):
        out.kind = OutcomeKind.PERMANENT_FAILURE  # type: ignore[misc]
