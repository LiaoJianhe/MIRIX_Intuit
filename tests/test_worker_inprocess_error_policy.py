"""Worker error-handling contract for the internal _consume_loop path.

After VEPAGE-1251 consolidation:

  * ``_process_message_async`` (the shared core) does NOT classify or mark the
    source complete — it runs the agent step and RE-RAISES on any failure.
  * Both the external (numaflow) path and the internal _consume_loop use
    the shared `error_policy.dispatch_save` helper, which runs
    `process_with_policy` then routes the verdict to the single finalize
    chokepoint.
  * step() does not finalize internally — `dispatch_save` calls
    finalize_source for all outcomes including SUCCESS (Option B).

These tests pin the core's re-raise contract and the consume-loop's
delegation to dispatch_save.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
# Core re-raises (so dispatch_save's process_with_policy can classify)
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
# Consume loop: delegates to dispatch_save
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consume_loop_delegates_to_dispatch_save(monkeypatch):
    """The consume loop pulls a message, then calls dispatch_save with a
    run_step closure that wraps _process_message_async. dispatch_save
    handles classification, retry, and finalize."""
    source_id = "src-loop"
    worker = QueueWorker(queue=Mock(), server=Mock())

    msg = _message(source_id)

    async def fake_get(timeout=None):
        worker._running = False
        return msg

    worker.queue.get = fake_get

    monkeypatch.setattr(
        worker, "_process_message_async", AsyncMock(return_value=None)
    )

    dispatch_mock = AsyncMock()
    with patch("mirix.queue.error_policy.dispatch_save", dispatch_mock):
        worker._running = True
        await worker._consume_loop()

    dispatch_mock.assert_awaited_once()
    # Called with run_step closure and the source_id.
    kwargs = dispatch_mock.await_args.kwargs
    assert kwargs["memory_source_id"] == source_id


@pytest.mark.asyncio
async def test_consume_loop_continues_on_unexpected_loop_body_error(monkeypatch):
    """If something other than dispatch_save raises in the loop body
    (e.g. a queue protocol error), the loop swallows and keeps running.
    No finalize call happens because we don't know the save's outcome."""
    worker = QueueWorker(queue=Mock(), server=Mock())

    call_count = {"n": 0}

    async def fake_get(timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: deliberately raise something unrecognized to
            # exercise the bare-except safety net.
            raise RuntimeError("queue protocol blew up")
        # Second call: stop the loop.
        worker._running = False
        raise __import__("asyncio").TimeoutError("loop drain")

    worker.queue.get = fake_get

    worker._running = True
    # Must not raise — the safety-net except swallows and logs.
    await worker._consume_loop()
    assert call_count["n"] >= 1
