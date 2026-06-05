"""Tests for LangFuse integration."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import pytest_asyncio

from mirix.observability.langfuse_client import (
    _reset_for_testing,
    flush_langfuse,
    get_langfuse_client,
    initialize_langfuse,
    is_langfuse_enabled,
    shutdown_langfuse,
)
from mirix.observability.trace_propagation import (
    TRACE_METADATA_KEY,
    add_trace_to_message,
    deserialize_trace_context,
    serialize_trace_context,
)


@pytest_asyncio.fixture(autouse=True)
async def reset_singleton():
    """Reset singleton before each test. Must be pytest_asyncio.fixture so async tests await it."""
    await _reset_for_testing()
    yield
    await _reset_for_testing()


@pytest.mark.asyncio
async def test_langfuse_disabled_by_default():
    """Test LangFuse is disabled without configuration."""
    with patch("mirix.settings.settings") as mock_settings:
        mock_settings.langfuse_enabled = False

        client = await initialize_langfuse()
        assert client is None
        assert not is_langfuse_enabled()


@pytest.mark.asyncio
async def test_langfuse_initialization_with_credentials():
    """Test LangFuse initializes with valid credentials."""
    with patch("mirix.settings.settings") as mock_settings:
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host = "https://cloud.langfuse.com"
        mock_settings.langfuse_debug = False
        mock_settings.langfuse_flush_interval = 10

        with patch("langfuse.Langfuse") as MockLangfuse, patch("opentelemetry.sdk.trace.TracerProvider"):
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            client = await initialize_langfuse()
            assert client is not None
            assert is_langfuse_enabled()

            mock_client.flush.assert_called_once()


@pytest.mark.asyncio
async def test_langfuse_missing_credentials():
    """Test LangFuse handles missing credentials gracefully."""
    with patch("mirix.settings.settings") as mock_settings:
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = None
        mock_settings.langfuse_secret_key = None

        client = await initialize_langfuse()
        assert client is None
        assert not is_langfuse_enabled()


def test_trace_context_serialization():
    """Test trace context can be serialized for Kafka."""
    from mirix.observability.context import set_trace_context

    set_trace_context(trace_id="trace-123", user_id="user-456", session_id="session-789")

    serialized = serialize_trace_context()

    assert serialized is not None
    assert serialized["trace_id"] == "trace-123"
    assert serialized["user_id"] == "user-456"
    assert serialized["session_id"] == "session-789"


def test_trace_context_serialization_no_trace():
    """Test serialization returns None when no trace context."""
    from mirix.observability.context import clear_trace_context

    clear_trace_context()
    serialized = serialize_trace_context()

    assert serialized is None


def test_trace_context_deserialization():
    """Test trace context can be restored from Kafka message."""
    from mirix.observability.context import get_trace_context

    message = {
        TRACE_METADATA_KEY: {
            "trace_id": "trace-abc",
            "user_id": "user-xyz",
        }
    }

    result = deserialize_trace_context(message)

    assert result is True

    context = get_trace_context()
    assert context["trace_id"] == "trace-abc"
    assert context["user_id"] == "user-xyz"


def test_trace_context_deserialization_no_metadata():
    """Test deserialization handles missing metadata."""
    message = {"some_key": "some_value"}

    result = deserialize_trace_context(message)

    assert result is False


def test_add_trace_to_message():
    """Test adding trace context to Kafka message."""
    from mirix.observability.context import set_trace_context

    set_trace_context(trace_id="trace-test-123")

    message = {"data": "test"}
    result = add_trace_to_message(message)

    assert TRACE_METADATA_KEY in result
    assert result[TRACE_METADATA_KEY]["trace_id"] == "trace-test-123"
    assert result["data"] == "test"


@pytest.mark.asyncio
async def test_graceful_degradation():
    """Test that operations work without LangFuse."""
    with patch("mirix.observability.langfuse_client.get_langfuse_client") as mock_get:
        mock_get.return_value = None

        assert not is_langfuse_enabled()
        assert await flush_langfuse() is True
        await shutdown_langfuse()


@pytest.mark.asyncio
async def test_flush_langfuse_with_timeout():
    """Test flush respects timeout parameter."""
    with patch("mirix.settings.settings") as mock_settings:
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host = "https://cloud.langfuse.com"
        mock_settings.langfuse_debug = False
        mock_settings.langfuse_flush_interval = 10
        mock_settings.langfuse_flush_timeout = 5.0

        with patch("langfuse.Langfuse") as MockLangfuse, patch("opentelemetry.sdk.trace.TracerProvider"):
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            await initialize_langfuse()
            result = await flush_langfuse(timeout=15.0)

            assert result is True
            mock_client.flush.assert_called()


def test_context_isolation():
    """Test that trace contexts don't leak between operations."""
    from mirix.observability.context import clear_trace_context, get_trace_context, set_trace_context

    set_trace_context(trace_id="trace-1")
    assert get_trace_context()["trace_id"] == "trace-1"

    clear_trace_context()
    assert get_trace_context()["trace_id"] is None

    set_trace_context(trace_id="trace-2")
    assert get_trace_context()["trace_id"] == "trace-2"


# ============================================================================
# Intuit TID (request correlation) propagation
#
# The TID is propagated independently of LangFuse trace context: it must reach
# the worker even when tracing is disabled (no active trace_id). These tests
# pin that behavior on the protobuf path used by the real Kafka queue.
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_tid():
    """Keep TID state clean around each test in this module."""
    from mirix.observability.context import clear_intuit_tid

    clear_intuit_tid()
    yield
    clear_intuit_tid()


def test_tid_propagates_to_queue_message_without_active_trace():
    """TID is copied onto the protobuf message even when there is NO trace."""
    from mirix.observability.context import clear_trace_context, set_intuit_tid
    from mirix.observability.trace_propagation import add_trace_to_queue_message
    from mirix.queue.message_pb2 import QueueMessage

    clear_trace_context()  # ensure no active trace -> trace fields skip
    set_intuit_tid("tid-no-trace")

    msg = add_trace_to_queue_message(QueueMessage())

    assert msg.HasField("intuit_tid")
    assert msg.intuit_tid == "tid-no-trace"
    # And the trace early-return held: no trace fields were populated.
    assert not msg.HasField("langfuse_trace_id")


def test_tid_restored_from_queue_message_without_trace_fields():
    """TID is restored into context independent of trace fields presence."""
    from mirix.observability.context import get_intuit_tid
    from mirix.observability.trace_propagation import restore_trace_from_queue_message
    from mirix.queue.message_pb2 import QueueMessage

    msg = QueueMessage()
    msg.intuit_tid = "tid-restore"

    # No trace fields set -> returns False for trace, but TID still restored.
    trace_restored = restore_trace_from_queue_message(msg)

    assert trace_restored is False
    assert get_intuit_tid() == "tid-restore"


def test_tid_and_trace_propagate_together():
    """When both are present, both make the round trip onto the message."""
    from mirix.observability.context import clear_trace_context, get_intuit_tid, set_intuit_tid, set_trace_context
    from mirix.observability.trace_propagation import add_trace_to_queue_message, restore_trace_from_queue_message
    from mirix.queue.message_pb2 import QueueMessage

    clear_trace_context()
    set_trace_context(trace_id="trace-xyz", user_id="user-1")
    set_intuit_tid("tid-both")

    msg = add_trace_to_queue_message(QueueMessage())
    assert msg.intuit_tid == "tid-both"
    assert msg.langfuse_trace_id == "trace-xyz"

    # Fresh worker side: clear, then restore.
    clear_trace_context()
    from mirix.observability.context import clear_intuit_tid

    clear_intuit_tid()
    restore_trace_from_queue_message(msg)
    assert get_intuit_tid() == "tid-both"


def test_tid_isolation():
    """TID does not leak between operations once cleared."""
    from mirix.observability.context import clear_intuit_tid, get_intuit_tid, set_intuit_tid

    set_intuit_tid("tid-1")
    assert get_intuit_tid() == "tid-1"

    clear_intuit_tid()
    assert get_intuit_tid() is None

    set_intuit_tid("tid-2")
    assert get_intuit_tid() == "tid-2"


def test_log_filter_injects_tid():
    """The logging filter stamps the current TID onto records ('-' when unset)."""
    import logging

    from mirix.log import _tid_log_filter
    from mirix.observability.context import clear_intuit_tid, set_intuit_tid

    set_intuit_tid("tid-log")
    rec = logging.LogRecord("Mirix", logging.INFO, "/tmp/x.py", 1, "msg", None, None)
    _tid_log_filter.filter(rec)
    assert rec.intuit_tid == "tid-log"

    clear_intuit_tid()
    rec2 = logging.LogRecord("Mirix", logging.INFO, "/tmp/x.py", 1, "msg", None, None)
    _tid_log_filter.filter(rec2)
    assert rec2.intuit_tid == "-"
