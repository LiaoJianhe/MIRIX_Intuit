"""S5 of VEPAGE-1251: internal consume loop routes through process_with_policy.

Before this story, `_consume_loop` swallowed every failure and finalized
regardless of bucket. A transient DB blip on the internal path was locked
in as complete-but-empty.

After this story, the internal loop runs `_process_message_async` under
`process_with_policy`, getting the same classification + in-process retry
the numaflow path already has:

- COMPLETED → noop (mark success via the in-step finalize chokepoint).
- PERMANENT_FAILURE → finalize_source(PERMANENT_FAILURE). Poison input;
  no future processing.
- TRANSIENT_EXHAUSTED → finalize_source(TRANSIENT_EXHAUSTED). The
  internal path has no redelivery, so we still mark done — but the
  outcome value is now distinguishable from PERMANENT for VEPAGE-1250.

This is the design's "#1 concrete win": same classifier, same policy,
all three run modes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from mirix.errors import LLMUnprocessableEntityError, LLMServerError
from mirix.queue.worker import QueueWorker


def _message(source_id):
    """Build a minimal QueueMessage with optional memory_source_id."""
    from mirix.queue.message_pb2 import QueueMessage

    msg = QueueMessage()
    msg.agent_id = "agent-loop"
    if source_id is not None:
        msg.memory_source_id = source_id
    return msg


@pytest.mark.asyncio
async def test_consume_loop_permanent_calls_finalize_with_permanent(monkeypatch):
    """A Permanent classification (422 from the LLM) finalizes with
    PERMANENT_FAILURE — not generic 'mark complete'."""
    from mirix.services.memory_source_manager import FinalizeOutcome

    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )

    source_id = "src-permanent"
    msg = _message(source_id)

    worker = QueueWorker(queue=Mock(), server=Mock())
    worker._running = True

    async def fake_get(timeout=None):
        worker._running = False
        return msg

    worker.queue.get = fake_get
    worker._process_message_async = AsyncMock(
        side_effect=LLMUnprocessableEntityError("422 rejected")
    )

    await worker._consume_loop()

    finalize.assert_awaited()
    args = finalize.call_args
    assert args.args[0] == source_id
    assert args.args[1] == FinalizeOutcome.PERMANENT_FAILURE


@pytest.mark.asyncio
async def test_consume_loop_transient_retried_then_finalized_exhausted(monkeypatch):
    """A sustained transient (LLM 5xx) runs through process_with_policy's
    retry loop. After exhaustion the internal-loop finalizes with
    TRANSIENT_EXHAUSTED (no redelivery on this path)."""
    from mirix.services.memory_source_manager import FinalizeOutcome

    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )
    # Make backoff sleeps zero-time so the test runs fast.
    monkeypatch.setattr(
        "mirix.queue.error_policy.asyncio.sleep", AsyncMock()
    )

    source_id = "src-transient"
    msg = _message(source_id)

    worker = QueueWorker(queue=Mock(), server=Mock())
    worker._running = True

    async def fake_get(timeout=None):
        worker._running = False
        return msg

    worker.queue.get = fake_get
    proc = AsyncMock(side_effect=LLMServerError("503 still"))
    worker._process_message_async = proc

    await worker._consume_loop()

    # Sustained transient: the policy retries to budget (more than 1
    # attempt). The exact count depends on whole_step_retry_max_attempts;
    # asserting >= 2 is enough to prove the retry loop ran.
    assert proc.await_count >= 2

    # And then the loop finalizes with the right outcome.
    finalize.assert_awaited()
    args = finalize.call_args
    assert args.args[0] == source_id
    assert args.args[1] == FinalizeOutcome.TRANSIENT_EXHAUSTED


@pytest.mark.asyncio
async def test_consume_loop_completed_does_not_finalize(monkeypatch):
    """Successful completion: the in-step finalize chokepoint already
    marked the source SUCCESS; the loop must NOT call finalize again."""
    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )

    msg = _message("src-ok")

    worker = QueueWorker(queue=Mock(), server=Mock())
    worker._running = True

    async def fake_get(timeout=None):
        worker._running = False
        return msg

    worker.queue.get = fake_get
    worker._process_message_async = AsyncMock(return_value=None)

    await worker._consume_loop()

    finalize.assert_not_awaited()
