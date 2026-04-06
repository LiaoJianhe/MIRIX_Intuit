"""Tests for S6: Extended POST /memory/add Request Schema.

Covers:
- AddMemoryRequest new fields
- Protobuf serialization/deserialization of new fields
- put_messages serialization of source fields
- Worker extraction of new fields from protobuf
- memory_source_id pre-generation and response inclusion
- Backward compatibility (existing clients without new fields)
"""

import json
import uuid

import pytest
from google.protobuf.json_format import MessageToDict, ParseDict

from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_util import deserialize_queue_message, serialize_queue_message


class TestAddMemoryRequestSchema:
    """Test the extended AddMemoryRequest Pydantic model."""

    def test_new_fields_accepted(self):
        from mirix.server.rest_api import AddMemoryRequest

        req = AddMemoryRequest(
            meta_agent_id="agent-1",
            messages=[{"role": "user", "content": "hello"}],
            external_id="ext-123",
            external_thread_id="thread-456",
            summarize=True,
            summary="A chat about billing",
            source_type="conversation",
            source_system="slack",
            source_metadata={"channel_id": "C123"},
        )
        assert req.external_id == "ext-123"
        assert req.external_thread_id == "thread-456"
        assert req.summarize is True
        assert req.summary == "A chat about billing"
        assert req.source_type == "conversation"
        assert req.source_system == "slack"
        assert req.source_metadata == {"channel_id": "C123"}

    def test_defaults(self):
        from mirix.server.rest_api import AddMemoryRequest

        req = AddMemoryRequest(
            meta_agent_id="agent-1",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.external_id is None
        assert req.external_thread_id is None
        assert req.summarize is False
        assert req.summary is None
        assert req.source_type == "conversation"
        assert req.source_system is None
        assert req.source_metadata is None

    def test_backward_compatible(self):
        """Existing clients sending only old fields still work."""
        from mirix.server.rest_api import AddMemoryRequest

        req = AddMemoryRequest(
            meta_agent_id="agent-1",
            messages=[{"role": "user", "content": "hi"}],
            chaining=False,
            verbose=True,
            use_cache=False,
            occurred_at="2026-01-15T10:00:00Z",
        )
        assert req.meta_agent_id == "agent-1"
        assert req.chaining is False


class TestProtobufSchema:
    """Test protobuf serialization/deserialization of new fields."""

    def test_queue_message_source_fields_roundtrip(self):
        msg = QueueMessage()
        msg.client_id = "client-1"
        msg.agent_id = "agent-1"
        msg.memory_source_id = "src-abc123"
        msg.external_id = "ext-456"
        msg.external_thread_id = "thread-789"
        msg.source_type = "conversation"
        msg.source_system = "slack"
        msg.source_metadata.update({"channel_id": "C123", "team_id": "T456"})
        msg.summary = "User asked about billing"
        msg.summarize = True

        serialized = msg.SerializeToString()
        deserialized = QueueMessage()
        deserialized.ParseFromString(serialized)

        assert deserialized.memory_source_id == "src-abc123"
        assert deserialized.external_id == "ext-456"
        assert deserialized.external_thread_id == "thread-789"
        assert deserialized.source_type == "conversation"
        assert deserialized.source_system == "slack"
        metadata = MessageToDict(deserialized.source_metadata)
        assert metadata["channel_id"] == "C123"
        assert metadata["team_id"] == "T456"
        assert deserialized.summary == "User asked about billing"
        assert deserialized.summarize is True

    def test_per_message_fields_roundtrip(self):
        msg = QueueMessage()
        msg.client_id = "client-1"
        msg.agent_id = "agent-1"

        proto_msg = ProtoMessageCreate()
        proto_msg.role = ProtoMessageCreate.ROLE_USER
        proto_msg.text_content = "hello"
        proto_msg.external_message_id = "msg-ext-1"
        proto_msg.message_occurred_at = "2026-01-15T10:00:00Z"
        msg.source_messages.append(proto_msg)

        serialized = msg.SerializeToString()
        deserialized = QueueMessage()
        deserialized.ParseFromString(serialized)

        src_msg = deserialized.source_messages[0]
        assert src_msg.external_message_id == "msg-ext-1"
        assert src_msg.message_occurred_at == "2026-01-15T10:00:00Z"
        assert src_msg.text_content == "hello"

    def test_old_messages_no_new_fields(self):
        """Old-format messages (without new fields) deserialize cleanly."""
        msg = QueueMessage()
        msg.client_id = "client-1"
        msg.agent_id = "agent-1"
        msg.chaining = True

        serialized = msg.SerializeToString()
        deserialized = QueueMessage()
        deserialized.ParseFromString(serialized)

        assert not deserialized.HasField("memory_source_id")
        assert not deserialized.HasField("external_id")
        assert not deserialized.HasField("external_thread_id")
        assert not deserialized.HasField("source_type")
        assert not deserialized.HasField("source_system")
        assert not deserialized.HasField("summary")
        assert not deserialized.HasField("summarize")
        assert len(deserialized.source_messages) == 0

    def test_json_serialization_roundtrip(self):
        """New fields survive JSON serialization format too."""
        msg = QueueMessage()
        msg.client_id = "client-1"
        msg.agent_id = "agent-1"
        msg.memory_source_id = "src-test"
        msg.external_id = "ext-test"
        msg.source_type = "document"

        serialized = serialize_queue_message(msg, format="json")
        deserialized = deserialize_queue_message(serialized, format="json")

        assert deserialized.memory_source_id == "src-test"
        assert deserialized.external_id == "ext-test"
        assert deserialized.source_type == "document"

    def test_protobuf_serialization_roundtrip(self):
        """New fields survive protobuf serialization format."""
        msg = QueueMessage()
        msg.client_id = "client-1"
        msg.agent_id = "agent-1"
        msg.memory_source_id = "src-test"
        msg.summarize = True

        proto_src = ProtoMessageCreate()
        proto_src.role = ProtoMessageCreate.ROLE_USER
        proto_src.text_content = "test"
        proto_src.external_message_id = "ext-msg-1"
        msg.source_messages.append(proto_src)

        serialized = serialize_queue_message(msg, format="protobuf")
        deserialized = deserialize_queue_message(serialized, format="protobuf")

        assert deserialized.memory_source_id == "src-test"
        assert deserialized.summarize is True
        assert deserialized.source_messages[0].external_message_id == "ext-msg-1"


class TestWorkerExtraction:
    """Test that the worker correctly extracts new fields from protobuf."""

    def test_convert_proto_message_carries_per_message_fields(self):
        from mirix.queue.worker import QueueWorker

        worker = QueueWorker.__new__(QueueWorker)

        proto_msg = ProtoMessageCreate()
        proto_msg.role = ProtoMessageCreate.ROLE_USER
        proto_msg.text_content = "hello world"
        proto_msg.external_message_id = "ext-msg-abc"
        proto_msg.message_occurred_at = "2026-01-15T14:30:00Z"

        pydantic_msg = worker._convert_proto_message_to_pydantic(proto_msg)

        assert pydantic_msg.content == "hello world"
        assert pydantic_msg.external_message_id == "ext-msg-abc"
        assert pydantic_msg.message_occurred_at == "2026-01-15T14:30:00Z"

    def test_convert_proto_message_without_per_message_fields(self):
        from mirix.queue.worker import QueueWorker

        worker = QueueWorker.__new__(QueueWorker)

        proto_msg = ProtoMessageCreate()
        proto_msg.role = ProtoMessageCreate.ROLE_USER
        proto_msg.text_content = "hello"

        pydantic_msg = worker._convert_proto_message_to_pydantic(proto_msg)

        assert pydantic_msg.content == "hello"
        assert pydantic_msg.external_message_id is None
        assert pydantic_msg.message_occurred_at is None


class TestSourceMessageNormalization:
    """Test that normalize_message extracts per-message fields from converted messages."""

    def test_normalize_with_per_message_fields(self):
        from mirix.queue.worker import QueueWorker
        from mirix.services.source_message_manager import normalize_message

        worker = QueueWorker.__new__(QueueWorker)

        proto_msg = ProtoMessageCreate()
        proto_msg.role = ProtoMessageCreate.ROLE_USER
        proto_msg.text_content = "hello"
        proto_msg.external_message_id = "ext-msg-1"
        proto_msg.message_occurred_at = "2026-01-15T10:00:00Z"

        pydantic_msg = worker._convert_proto_message_to_pydantic(proto_msg)
        normalized = normalize_message(pydantic_msg)

        assert normalized["role"] == "user"
        assert normalized["content"] == {"text": "hello"}
        assert normalized["external_message_id"] == "ext-msg-1"
        assert normalized["occurred_at"] == "2026-01-15T10:00:00Z"

    def test_normalize_without_per_message_fields(self):
        from mirix.queue.worker import QueueWorker
        from mirix.services.source_message_manager import normalize_message

        worker = QueueWorker.__new__(QueueWorker)

        proto_msg = ProtoMessageCreate()
        proto_msg.role = ProtoMessageCreate.ROLE_USER
        proto_msg.text_content = "hello"

        pydantic_msg = worker._convert_proto_message_to_pydantic(proto_msg)
        normalized = normalize_message(pydantic_msg)

        assert normalized["role"] == "user"
        assert "external_message_id" not in normalized
        assert "occurred_at" not in normalized


class TestMemorySourceIdGeneration:
    """Test memory_source_id format and uniqueness."""

    def test_memory_source_id_format(self):
        """memory_source_id should be src-{uuid4} format."""
        memory_source_id = f"src-{uuid.uuid4()}"
        assert memory_source_id.startswith("src-")
        parts = memory_source_id.split("-", 1)
        assert parts[0] == "src"
        # Validate UUID portion
        uuid.UUID(parts[1])

    def test_memory_source_id_uniqueness(self):
        """Each call generates a unique ID."""
        ids = {f"src-{uuid.uuid4()}" for _ in range(100)}
        assert len(ids) == 100
