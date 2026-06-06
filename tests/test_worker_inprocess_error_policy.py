"""Worker error-handling contract across the two consumption paths.

Design (after consolidating the policy):

  * ``_process_message_async`` (the shared core) does NOT classify or mark the
    source complete — it runs the agent step and RE-RAISES on any failure. This
    is what lets the external consumer's ``process_with_policy`` actually see the
    exception (previously the core swallowed it, so the external policy was dead).

  * The internal ``_consume_loop`` has no redelivery (``queue.get`` already
    consumed the message), so on a propagated failure it marks the source
    ``processing_complete`` and swallows — otherwise the source hangs and the
    SDK's ``wait_for_save`` polls to its timeout.

These tests pin both halves without a DB/LLM.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.worker import QueueWorker


def _message(source_id: str | None) -> QueueMessage:
    m = QueueMessage()
    m.agent_id = "agent-test"
    m.client_id = "client-test"
    m.user_id = "user-test"
    if source_id is not None:
        m.memory_source_id = source_id
    return m


# ---------------------------------------------------------------------------
# Core re-raises (so external process_with_policy can classify)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_reraises_on_failure(monkeypatch):
    """_process_message_async must propagate, not swallow, processing failures."""
    actor = SimpleNamespace(id="client-test", organization_id="org-test")
    server = Mock()
    server.client_manager = Mock(get_client_by_id=AsyncMock(return_value=actor))
    server.send_messages = AsyncMock(side_effect=RuntimeError("boom"))

    user = SimpleNamespace(id="user-test", organization_id="org-test")
    monkeypatch.setattr(
        "mirix.queue.worker.UserManager",
        Mock(return_value=Mock(get_user_by_id=AsyncMock(return_value=user))),
    )

    worker = QueueWorker(queue=Mock(), server=server)

    with pytest.raises(RuntimeError, match="boom"):
        await worker._process_message_async(_message("src-x"))


# ---------------------------------------------------------------------------
# Internal finalize-on-failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_marks_source_complete(monkeypatch):
    """Worker finalize routes through the single chokepoint
    (MemorySourceManager.finalize_source); the chokepoint then writes
    processing_complete=True via mark_processing_complete."""
    from mirix.services.memory_source_manager import FinalizeOutcome

    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )
    worker = QueueWorker(queue=Mock(), server=Mock())
    await worker._finalize_source_on_failure(_message("src-finalize"))
    finalize.assert_awaited_once_with("src-finalize", FinalizeOutcome.PERMANENT_FAILURE)


@pytest.mark.asyncio
async def test_finalize_is_noop_without_source_id(monkeypatch):
    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )
    worker = QueueWorker(queue=Mock(), server=Mock())
    # No memory_source_id on the message, and a None message: both no-op.
    await worker._finalize_source_on_failure(_message(None))
    await worker._finalize_source_on_failure(None)
    finalize.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_never_raises_on_mark_error(monkeypatch):
    """The chokepoint itself catches DB failures and logs — same contract
    as the legacy _finalize_source_on_failure: must not raise."""
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(
            return_value=Mock(
                finalize_source=AsyncMock(side_effect=RuntimeError("should never escape")),
                mark_processing_complete=AsyncMock(side_effect=RuntimeError("db down")),
            )
        ),
    )
    worker = QueueWorker(queue=Mock(), server=Mock())
    # The chokepoint must swallow internal failures so the loop keeps running.
    # If the mock's finalize_source side_effect leaked, this test would raise.
    # (The real finalize_source wraps mark_processing_complete in try/except.)
    try:
        await worker._finalize_source_on_failure(_message("src-err"))
    except RuntimeError:
        pytest.fail("finalize must not propagate DB errors out of the consume loop")


# ---------------------------------------------------------------------------
# Consume loop: a processing failure → finalize + swallow + keep looping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consume_loop_finalizes_and_swallows_on_failure(monkeypatch):
    source_id = "src-loop"
    worker = QueueWorker(queue=Mock(), server=Mock())

    # One message, then stop the loop so it doesn't spin.
    msg = _message(source_id)

    async def fake_get(timeout=None):
        worker._running = False  # stop after this iteration
        return msg

    worker.queue.get = fake_get

    # Core raises (simulating a processing failure).
    monkeypatch.setattr(worker, "_process_message_async", AsyncMock(side_effect=RuntimeError("boom")))
    finalize = AsyncMock()
    monkeypatch.setattr(worker, "_finalize_source_on_failure", finalize)

    worker._running = True
    await worker._consume_loop()  # must not raise

    finalize.assert_awaited_once()
    # The finalize was called with the message that failed.
    assert finalize.await_args.args[0] is msg
