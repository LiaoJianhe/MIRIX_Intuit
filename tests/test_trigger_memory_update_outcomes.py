"""Tests for the sub-agent fan-out outcome reducer in memory_tools.

These guard the 422-cascade fix for the sub-agent case: when trigger_memory_update
fans out to multiple sub-agents in parallel and any of them hits a Permanent
LLM error, the original exception type must propagate up unwrapped so the
error_policy classifier can mark the whole step Permanent and ack.

Test strategy: drive the pure helper directly. No async, no LLM, no agent
infrastructure needed — the helper is the entire decision boundary.
"""

from __future__ import annotations

import logging

import pytest

from mirix.errors import LLMRateLimitError, LLMUnprocessableEntityError
from mirix.functions.function_sets.memory_tools import (
    _decide_step_outcome_from_sub_agent_results,
)
from mirix.queue.error_policy import Bucket, classify


def test_all_success_returns_responses():
    """Baseline: all sub-agents succeeded → return their responses, no exception."""
    out = _decide_step_outcome_from_sub_agent_results(
        memory_types=["semantic", "episodic", "resource"],
        results=["sem ok\n", "ep ok\n", "res ok\n"],
    )
    assert out == ["sem ok\n", "ep ok\n", "res ok\n"]


def test_all_permanent_raises_permanent_classifiable_exception():
    """The 422 cascade case. If every failure is Permanent, the propagated
    exception must classify as PERMANENT so the policy layer acks the message
    instead of retrying."""
    perm1 = LLMUnprocessableEntityError("rejected by risk screening")
    perm2 = LLMUnprocessableEntityError("rejected by risk screening")
    perm3 = LLMUnprocessableEntityError("rejected by risk screening")
    with pytest.raises(BaseException) as exc_info:
        _decide_step_outcome_from_sub_agent_results(
            memory_types=["semantic", "episodic", "resource"],
            results=[perm1, perm2, perm3],
        )
    assert classify(exc_info.value) is Bucket.PERMANENT, (
        "all-permanent failures must propagate a Permanent-classifiable exception "
        "so the step acks instead of retrying — this is the 422 cascade fix"
    )


def test_mixed_transient_and_permanent_raises_permanent():
    """Any Permanent failure makes the whole step Permanent. The transient
    sibling does NOT win — retrying the whole step would just hit the same
    permanent failure again. The transient gets a fresh chance on the next
    message's natural lifecycle."""
    transient = LLMRateLimitError("429 backoff")
    permanent = LLMUnprocessableEntityError("rejected")
    with pytest.raises(BaseException) as exc_info:
        _decide_step_outcome_from_sub_agent_results(
            memory_types=["semantic", "episodic", "resource"],
            results=[permanent, transient, "res ok\n"],
        )
    assert classify(exc_info.value) is Bucket.PERMANENT


def test_all_success_plus_one_permanent_raises_permanent():
    """Two sub-agents succeeded (their memory writes are committed). One failed
    permanently. No transient anywhere → retrying won't help → ack."""
    permanent = LLMUnprocessableEntityError("rejected")
    with pytest.raises(BaseException) as exc_info:
        _decide_step_outcome_from_sub_agent_results(
            memory_types=["semantic", "episodic", "resource"],
            results=["sem ok\n", permanent, "res ok\n"],
        )
    assert classify(exc_info.value) is Bucket.PERMANENT


def test_permanent_wins_regardless_of_iteration_order():
    """The decision must not depend on which exception appears first in the list.
    A 422 at position 2 should still produce a Permanent outcome even when a
    429 at position 0 is encountered first."""
    perm = LLMUnprocessableEntityError("rejected")
    trans = LLMRateLimitError("429")
    with pytest.raises(BaseException) as exc_info:
        _decide_step_outcome_from_sub_agent_results(
            memory_types=["a", "b", "c"],
            results=[trans, "ok\n", perm],
        )
    assert classify(exc_info.value) is Bucket.PERMANENT


def test_all_transient_raises_transient():
    """With no permanent failures, an all-transient step should propagate
    Transient so the policy layer retries."""
    trans1 = LLMRateLimitError("429")
    trans2 = LLMRateLimitError("429")
    with pytest.raises(BaseException) as exc_info:
        _decide_step_outcome_from_sub_agent_results(
            memory_types=["a", "b", "c"],
            results=[trans1, "ok\n", trans2],
        )
    assert classify(exc_info.value) is Bucket.TRANSIENT


def test_logs_every_sub_agent_failure(caplog):
    """Diagnostic regression guard: the old wrapping `RuntimeError(..., 'semantic')`
    captured the sub-agent name in the message. The new design uses logs instead.
    Every failure must be logged with its sub-agent name and exception type.

    Mirix's get_logger() returns a non-propagating "Mirix" logger, so we attach
    caplog's handler directly to that logger for the duration of the test
    rather than relying on caplog.at_level which only patches propagation.
    """
    perm = LLMUnprocessableEntityError("rejected")
    trans = LLMRateLimitError("429")

    mirix_logger = logging.getLogger("Mirix")
    mirix_logger.addHandler(caplog.handler)
    previous_level = mirix_logger.level
    mirix_logger.setLevel(logging.WARNING)
    try:
        with pytest.raises(BaseException):
            _decide_step_outcome_from_sub_agent_results(
                memory_types=["semantic", "episodic", "resource"],
                results=[perm, trans, "ok\n"],
            )
    finally:
        mirix_logger.removeHandler(caplog.handler)
        mirix_logger.setLevel(previous_level)

    log_messages = "\n".join(r.message for r in caplog.records)
    assert "semantic" in log_messages
    assert "episodic" in log_messages
    assert "LLMUnprocessableEntityError" in log_messages
    assert "LLMRateLimitError" in log_messages
