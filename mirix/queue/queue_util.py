import json
import logging
from typing import List, Optional

from google.protobuf.json_format import MessageToDict, ParseDict

import mirix.queue as queue
from mirix.observability import add_trace_to_queue_message
from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
from mirix.schemas.client import Client
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import MessageCreate
from mirix.schemas.mirix_message_content import TextContent

logger = logging.getLogger(__name__)


# Queue message serialization utilities


def serialize_queue_message(message: QueueMessage, format: str = "protobuf") -> bytes:
    """
    Serialize QueueMessage to bytes in the specified format.

    Args:
        message: QueueMessage protobuf to serialize
        format: Serialization format - 'protobuf' or 'json'

    Returns:
        Serialized message bytes

    Raises:
        ValueError: If format is not supported
    """
    if format == "json":
        message_dict = MessageToDict(message, preserving_proto_field_name=True)
        return json.dumps(message_dict).encode("utf-8")
    elif format == "protobuf":
        return message.SerializeToString()
    else:
        raise ValueError(f"Unsupported serialization format: {format}")


def deserialize_queue_message(serialized_msg: bytes, format: str = "protobuf") -> QueueMessage:
    """
    Deserialize bytes to QueueMessage in the specified format.

    Args:
        serialized_msg: Serialized message bytes
        format: Serialization format - 'protobuf' or 'json'

    Returns:
        QueueMessage protobuf object

    Raises:
        ValueError: If format is not supported or deserialization fails
    """
    queue_message = QueueMessage()

    try:
        if format == "json":
            message_dict = json.loads(serialized_msg.decode("utf-8"))
            return ParseDict(message_dict, queue_message)
        elif format == "protobuf":
            queue_message.ParseFromString(serialized_msg)
            return queue_message
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    except Exception as e:
        raise ValueError(f"Failed to deserialize message ({format} format): {e}") from e


async def put_messages(
    actor: Client,
    agent_id: str,
    input_messages: List[MessageCreate],
    chaining: Optional[bool] = True,
    user_id: Optional[str] = None,
    verbose: Optional[bool] = None,
    filter_tags: Optional[dict] = None,
    block_filter_tags: Optional[dict] = None,
    block_filter_tags_update_mode: Optional[str] = "merge",
    use_cache: bool = True,
    occurred_at: Optional[str] = None,
    memory_source_id: Optional[str] = None,
    external_id: Optional[str] = None,
    external_thread_id: Optional[str] = None,
    source_type: Optional[str] = None,
    source_system: Optional[str] = None,
    source_metadata: Optional[dict] = None,
    summary: Optional[str] = None,
    summarize: bool = False,
    source_messages: Optional[List[dict]] = None,
):
    """
    Create QueueMessage protobuf and send to queue.

    Args:
        actor: The Client performing the action (for auth/write operations)
               Client ID is derived from actor.id
        agent_id: ID of the agent to send message to
        input_messages: List of messages to send
        chaining: Enable/disable chaining
        user_id: Optional user ID (end-user ID)
        verbose: Enable verbose logging
        filter_tags: Filter tags dictionary
        block_filter_tags: Optional dict; applied only when blocks are created (e.g. from default template)
        block_filter_tags_update_mode: "merge" (default) or "replace" for existing block filter_tags
        use_cache: Control Redis cache behavior
        occurred_at: Optional ISO 8601 timestamp string for episodic memory
    """
    logger.debug("Creating queue message for agent_id=%s, client_id=%s", agent_id, actor.id)

    if not actor or not actor.id:
        raise ValueError(
            f"Cannot queue message: actor is None or has no ID. "
            f"actor={actor}, actor.id={actor.id if actor else 'N/A'}"
        )

    # Convert Pydantic MessageCreate list to protobuf MessageCreate list
    proto_input_messages = []
    for msg in input_messages:
        proto_msg = ProtoMessageCreate()
        # Map role
        if msg.role == MessageRole.user:
            proto_msg.role = ProtoMessageCreate.ROLE_USER
        elif msg.role == MessageRole.system:
            proto_msg.role = ProtoMessageCreate.ROLE_SYSTEM
        else:
            proto_msg.role = ProtoMessageCreate.ROLE_UNSPECIFIED

        # Handle content (can be string or list)
        if isinstance(msg.content, str):
            proto_msg.text_content = msg.content
        # For list content, we'd need to convert to structured_content
        # but for now, just convert to string representation
        elif isinstance(msg.content, list):
            # Convert list of content to string for now
            text_parts = []
            for content_part in msg.content:
                if isinstance(content_part, TextContent):
                    text_parts.append(content_part.text)
            proto_msg.text_content = "\n".join(text_parts)

        # Optional fields
        if msg.name:
            proto_msg.name = msg.name
        if msg.otid:
            proto_msg.otid = msg.otid
        if msg.sender_id:
            proto_msg.sender_id = msg.sender_id
        if msg.group_id:
            proto_msg.group_id = msg.group_id
        if msg.external_message_id:
            proto_msg.external_message_id = msg.external_message_id
        if msg.message_occurred_at:
            proto_msg.message_occurred_at = msg.message_occurred_at

        proto_input_messages.append(proto_msg)

    # Build the QueueMessage
    queue_msg = QueueMessage()

    queue_msg.client_id = actor.id

    queue_msg.agent_id = agent_id
    queue_msg.input_messages.extend(proto_input_messages)

    # Optional fields
    if chaining is not None:
        queue_msg.chaining = chaining
    if user_id:
        queue_msg.user_id = user_id
    if verbose is not None:
        queue_msg.verbose = verbose

    # Convert dict to Struct for filter_tags
    if filter_tags:
        queue_msg.filter_tags.update(filter_tags)

    # Optional block_filter_tags (applied only when blocks are created)
    if block_filter_tags:
        queue_msg.block_filter_tags.update(block_filter_tags)

    if block_filter_tags_update_mode:
        queue_msg.block_filter_tags_update_mode = block_filter_tags_update_mode

    # Set use_cache
    queue_msg.use_cache = use_cache

    # Set occurred_at if provided
    if occurred_at is not None:
        queue_msg.occurred_at = occurred_at

    # Memory source fields
    if memory_source_id is not None:
        queue_msg.memory_source_id = memory_source_id
    if external_id is not None:
        queue_msg.external_id = external_id
    if external_thread_id is not None:
        queue_msg.external_thread_id = external_thread_id
    if source_type is not None:
        queue_msg.source_type = source_type
    if source_system is not None:
        queue_msg.source_system = source_system
    if source_metadata:
        queue_msg.source_metadata.update(source_metadata)
    if summary is not None:
        queue_msg.summary = summary
    if summarize:
        queue_msg.summarize = summarize

    # WHY TWO MESSAGE ARRAYS (source_messages vs input_messages)?
    #
    # input_messages (field 3) carries the PACKED message for agent processing.
    # The add_memory handler flattens all conversation turns into a single
    # MessageCreate with [USER]/[ASSISTANT] text markers — this is what the
    # MetaAgent and sub-agents consume. Per-message identity (role, external_message_id,
    # occurred_at) is lost during that flattening.
    #
    # source_messages (field 24) carries the ORIGINAL per-turn messages so that
    # _persist_memory_source() can store them individually in the source_messages
    # DB table with their per-message metadata intact. These are never used for
    # agent processing.
    #
    # This duplication exists because the flattening happens before enqueue and
    # we can't change that without reworking how agents consume messages. A future
    # refactor could move the flattening into the MetaAgent itself, eliminating
    # the need for two copies.
    if source_messages:
        for msg_dict in source_messages:
            proto_src_msg = ProtoMessageCreate()
            role = msg_dict.get("role", "user")
            if role == "user":
                proto_src_msg.role = ProtoMessageCreate.ROLE_USER
            elif role == "system":
                proto_src_msg.role = ProtoMessageCreate.ROLE_SYSTEM
            elif role == "assistant":
                proto_src_msg.role = ProtoMessageCreate.ROLE_ASSISTANT
            else:
                proto_src_msg.role = ProtoMessageCreate.ROLE_UNSPECIFIED

            content = msg_dict.get("content", "")
            if isinstance(content, str):
                proto_src_msg.text_content = content
            elif isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                proto_src_msg.text_content = "\n".join(text_parts)

            if msg_dict.get("external_message_id"):
                proto_src_msg.external_message_id = msg_dict["external_message_id"]
            if msg_dict.get("occurred_at"):
                proto_src_msg.message_occurred_at = msg_dict["occurred_at"]

            queue_msg.source_messages.append(proto_src_msg)

    # Add LangFuse trace context for distributed tracing
    queue_msg = add_trace_to_queue_message(queue_msg)

    # Send to queue
    logger.debug(
        "Sending message to queue: agent_id=%s, input_messages_count=%s, occurred_at=%s",
        agent_id,
        len(input_messages),
        occurred_at,
    )
    await queue.save(queue_msg)
    logger.debug("Message successfully sent to queue")
