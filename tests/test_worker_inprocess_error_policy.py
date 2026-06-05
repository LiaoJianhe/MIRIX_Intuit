"""In-process queue worker runs the same error policy as the Kafka path.

Regression guard for the divergence where ``QueueWorker._process_message_async``
(the in-process / ``kafka_enabled=false`` path the full-stack tests use) merely
logged-and-swallowed a save failure and never marked the memory source
``processing_complete`` — so the source hung "in progress" forever and the SDK's
``wait_for_save`` polled to its 300s timeout. The Kafka path
(``process_external_message``) already classified errors and marked the source
complete on permanent failure; these tests pin that the in-process worker now:

  * routes the agent step through ``process_with_policy`` with an ``on_permanent``
    callback that marks the source complete, and
  * also marks the source complete on TRANSIENT_EXHAUSTED (the in-process worker
    has no redelivery behind it, unlike the Kafka consumer).

We patch ``process_with_policy`` to drive each Outcome deterministically and stub
the actor/user resolution, so no DB/LLM is needed — the unit under test is the
worker's *handling* of outcomes, not the policy classifier (covered by
test_error_policy.py).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mirix.queue.error_policy import Bucket, Outcome, OutcomeKind
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.worker import QueueWorker


def _message(source_id: str) -> QueueMessage:
    m = QueueMessage()
    m.agent_id = "agent-test"
    m.client_id = "client-test"
    m.user_id = "user-test"
    m.memory_source_id = source_id
    return m


def _worker_with_stubbed_resolution() -> QueueWorker:
    """A worker whose actor/user resolution is stubbed (no DB)."""
    actor = SimpleNamespace(id="client-test", organization_id="org-test")
    server = Mock()
    server.client_manager = Mock(get_client_by_id=AsyncMock(return_value=actor))
    server.send_messages = AsyncMock(return_value=None)
    return QueueWorker(queue=Mock(), server=server)


@pytest.fixture
def patched(monkeypatch):
    """Patch UserManager (DB) and capture the mark_processing_complete mock."""
    user = SimpleNamespace(id="user-test", organization_id="org-test")
    fake_user_mgr = Mock(
        get_user_by_id=AsyncMock(return_value=user),
        get_admin_user=AsyncMock(return_value=user),
        DEFAULT_TIME_ZONE="UTC",
    )
    monkeypatch.setattr("mirix.queue.worker.UserManager", Mock(return_value=fake_user_mgr))

    mark_complete = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(mark_processing_complete=mark_complete)),
    )
    return SimpleNamespace(mark_complete=mark_complete, monkeypatch=monkeypatch)


@pytest.mark.asyncio
async def test_permanent_failure_marks_source_complete(patched):
    source_id = "src-permanent"

    # Drive the policy to a PERMANENT_FAILURE and invoke the on_permanent the
    # worker passed in (that's what marks the source complete).
    async def fake_policy(run_step, *, memory_source_id=None, on_permanent=None):
        exc = RuntimeError("boom-permanent")
        if on_permanent and memory_source_id:
            await on_permanent(memory_source_id, str(exc), exc)
        return Outcome(kind=OutcomeKind.PERMANENT_FAILURE, cause=exc, bucket=Bucket.PERMANENT)

    patched.monkeypatch.setattr("mirix.queue.error_policy.process_with_policy", fake_policy)

    await _worker_with_stubbed_resolution()._process_message_async(_message(source_id))
    patched.mark_complete.assert_awaited_once_with(source_id)


@pytest.mark.asyncio
async def test_transient_exhausted_marks_source_complete(patched):
    source_id = "src-transient"

    async def fake_policy(run_step, *, memory_source_id=None, on_permanent=None):
        # Transient path: policy does NOT call on_permanent; the worker must
        # finalize the source itself because there's no redelivery.
        return Outcome(
            kind=OutcomeKind.TRANSIENT_EXHAUSTED,
            cause=RuntimeError("boom-transient"),
            bucket=Bucket.TRANSIENT,
        )

    patched.monkeypatch.setattr("mirix.queue.error_policy.process_with_policy", fake_policy)

    await _worker_with_stubbed_resolution()._process_message_async(_message(source_id))
    patched.mark_complete.assert_awaited_once_with(source_id)


@pytest.mark.asyncio
async def test_completed_does_not_mark_via_failure_path(patched):
    source_id = "src-ok"

    async def fake_policy(run_step, *, memory_source_id=None, on_permanent=None):
        await run_step()  # healthy
        return Outcome(kind=OutcomeKind.COMPLETED)

    patched.monkeypatch.setattr("mirix.queue.error_policy.process_with_policy", fake_policy)

    await _worker_with_stubbed_resolution()._process_message_async(_message(source_id))
    # The failure-path finalizer must NOT fire on success (the normal pipeline
    # owns completion marking).
    patched.mark_complete.assert_not_awaited()
