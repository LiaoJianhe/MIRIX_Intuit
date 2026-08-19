"""Direct-producer saves (EventBus publishers bypassing the HTTP save API)
carry no LangFuse trace context on the queue message. Before ECMS-548 this
meant the worker silently skipped all span/trace emission for these saves —
they succeeded but were invisible in LangFuse. The worker now mints its own
worker-rooted trace when none was propagated, so these saves still get
tid/client/write_kind tagging like any other save.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.worker import QueueWorker


def _build_message(client_id="client-1"):
    msg = QueueMessage()
    msg.client_id = client_id
    msg.agent_id = "agent-1"
    msg.user_id = "user-1"
    return msg


def _make_worker(write_scope="client-real-scope"):
    actor = SimpleNamespace(id="client-1", organization_id="org-1", write_scope=write_scope, name="direct-producer")
    user = SimpleNamespace(id="user-1", organization_id="org-1")

    server = MagicMock()
    server.client_manager.get_client_by_id = AsyncMock(return_value=actor)
    send_spy = AsyncMock(return_value=MagicMock(step_count=1))
    server.send_messages = send_spy

    worker = QueueWorker(queue=MagicMock(), server=server)
    return worker, send_spy, user


@pytest.mark.asyncio
async def test_worker_mints_trace_when_message_has_no_trace_context():
    """No trace context on the message + LangFuse enabled -> the worker mints
    its own trace_id and still opens the 'Meta Agent' span/tags, instead of
    silently no-op'ing."""
    worker, send_spy, user = _make_worker()
    message = _build_message()

    mock_langfuse = MagicMock()
    mock_span = MagicMock()
    mock_langfuse.start_as_current_observation.return_value.__enter__.return_value = mock_span

    with (
        patch("mirix.queue.worker.UserManager") as mock_um,
        patch("mirix.queue.worker.reconcile_user_org_to_actor", side_effect=lambda u, a: u),
        patch("mirix.queue.worker.restore_trace_from_queue_message", return_value=False),
        patch("mirix.queue.worker.get_langfuse_client", return_value=mock_langfuse),
        patch("mirix.queue.worker.get_trace_context", return_value=None),
        patch("mirix.queue.worker.mark_observation_as_child") as mock_mark_child,
        patch("mirix.observability.trace_attrs.update_trace_attributes") as mock_update_attrs,
    ):
        mock_um.return_value.get_user_by_id = AsyncMock(return_value=user)
        mock_um.return_value.get_admin_user = AsyncMock(return_value=user)

        await worker._process_message_async(message)

    send_spy.assert_awaited_once()

    # A trace was opened even though the message carried no trace context.
    mock_langfuse.start_as_current_observation.assert_called_once()
    trace_context = mock_langfuse.start_as_current_observation.call_args.kwargs["trace_context"]
    assert trace_context["trace_id"]

    # A worker-minted trace has no HTTP-entry parent — it must stay root, so
    # mark_observation_as_child must NOT be called (unlike a propagated trace).
    mock_mark_child.assert_not_called()

    # Existing tid/client/write_kind tagging still happens for minted traces.
    mock_update_attrs.assert_called_once()
    tags = mock_update_attrs.call_args.kwargs["tags"]
    assert any(t.startswith("client:") for t in tags)
    assert any(t.startswith("write_kind:") for t in tags)


@pytest.mark.asyncio
async def test_worker_does_not_mint_trace_when_langfuse_disabled():
    """No active LangFuse client -> no trace minting, no span emission
    attempted (existing no-op-when-disabled behavior is preserved)."""
    worker, send_spy, user = _make_worker()
    message = _build_message()

    with (
        patch("mirix.queue.worker.UserManager") as mock_um,
        patch("mirix.queue.worker.reconcile_user_org_to_actor", side_effect=lambda u, a: u),
        patch("mirix.queue.worker.restore_trace_from_queue_message", return_value=False),
        patch("mirix.queue.worker.get_langfuse_client", return_value=None),
        patch("mirix.queue.worker.get_trace_context", return_value=None),
    ):
        mock_um.return_value.get_user_by_id = AsyncMock(return_value=user)
        mock_um.return_value.get_admin_user = AsyncMock(return_value=user)

        await worker._process_message_async(message)

    send_spy.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_preserves_propagated_trace_as_child():
    """When the message DOES carry a propagated trace_id (HTTP-entry save),
    the worker must still mark its span as a child, not root — minting must
    only kick in when trace_id is absent."""
    worker, send_spy, user = _make_worker()
    message = _build_message()

    mock_langfuse = MagicMock()
    mock_span = MagicMock()
    mock_langfuse.start_as_current_observation.return_value.__enter__.return_value = mock_span

    with (
        patch("mirix.queue.worker.UserManager") as mock_um,
        patch("mirix.queue.worker.reconcile_user_org_to_actor", side_effect=lambda u, a: u),
        patch("mirix.queue.worker.restore_trace_from_queue_message", return_value=True),
        patch("mirix.queue.worker.get_langfuse_client", return_value=mock_langfuse),
        patch(
            "mirix.queue.worker.get_trace_context",
            return_value={
                "trace_id": "propagated-trace-id",
                "observation_id": None,
                "user_id": None,
                "session_id": None,
            },
        ),
        patch("mirix.queue.worker.mark_observation_as_child") as mock_mark_child,
        patch("mirix.observability.trace_attrs.update_trace_attributes"),
    ):
        mock_um.return_value.get_user_by_id = AsyncMock(return_value=user)
        mock_um.return_value.get_admin_user = AsyncMock(return_value=user)

        await worker._process_message_async(message)

    send_spy.assert_awaited_once()
    trace_context = mock_langfuse.start_as_current_observation.call_args.kwargs["trace_context"]
    assert trace_context["trace_id"] == "propagated-trace-id"
    mock_mark_child.assert_called_once()
