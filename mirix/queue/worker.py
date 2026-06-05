"""
Background worker that consumes messages from the queue.
Runs as an asyncio.Task in the main event loop (async-native).
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from google.protobuf.json_format import MessageToDict

from mirix.log import get_logger
from mirix.observability import get_langfuse_client, mark_observation_as_child, restore_trace_from_queue_message
from mirix.observability.context import (
    clear_intuit_tid,
    clear_trace_context,
    get_intuit_tid,
    get_trace_context,
)
from mirix.queue.message_pb2 import QueueMessage
from mirix.services.user_manager import UserManager

if TYPE_CHECKING:
    from mirix.schemas.client import Client
    from mirix.schemas.message import MessageCreate

    from .queue_interface import QueueInterface


logger = get_logger(__name__)


def reconcile_user_org_to_actor(user, actor):
    """Return ``user`` with its org corrected to the actor's (client's) org.

    The ``users`` row is a global, id-only-PK shared stub (VEPAGE-1155): its
    ``organization_id`` records whichever org first created the user, and
    per-org isolation lives on the child tables (blocks, memories). On the save
    path the resolved user therefore carries its *first-seen* org, not the org
    this save is for. Block scoping (``block_manager.get_blocks`` keys off
    ``user.organization_id``) and the temporal guard then operate against the
    stale org. The save's authoritative org is the client/actor's org, so
    correct it here, at the single point where the stub becomes the live user
    for this save.

    Returns a copy (never mutates the input). No-op when ``user`` is None, the
    actor has no org, or the orgs already match.
    """
    if user is None:
        return None
    actor_org = getattr(actor, "organization_id", None)
    if not actor_org or user.organization_id == actor_org:
        return user
    return user.model_copy(update={"organization_id": actor_org})


class QueueWorker:
    """Background worker that processes messages from the queue as an asyncio.Task"""

    def __init__(
        self,
        queue: "QueueInterface",
        server: Optional[Any] = None,
        partition_id: Optional[int] = None,
    ):
        """
        Initialize the queue worker.

        Args:
            queue: Async queue implementation to consume from
            server: Optional server instance to invoke APIs on
            partition_id: Optional partition ID for partitioned queues.
                         If set, worker will only consume from this partition.
        """
        logger.debug(
            "Initializing queue worker: server=%s, partition_id=%s",
            "provided" if server else "None",
            partition_id,
        )

        self.queue = queue
        self._server = server
        self._partition_id = partition_id
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def _convert_proto_user_to_pydantic(self, proto_user) -> "Client":
        """
        Convert protobuf User to Pydantic Client.

        The protobuf schema still uses "User" for historical reasons,
        but it represents a Client in the new architecture.
        """
        from mirix.schemas.client import Client

        return Client(
            id=proto_user.id,
            organization_id=(proto_user.organization_id if proto_user.organization_id else None),
            name=proto_user.name,
            status=proto_user.status,
            write_scope=None,
            read_scopes=[],
            created_at=(proto_user.created_at.ToDatetime() if proto_user.HasField("created_at") else datetime.now()),
            updated_at=(proto_user.updated_at.ToDatetime() if proto_user.HasField("updated_at") else datetime.now()),
            is_deleted=proto_user.is_deleted,
        )

    def _convert_proto_message_to_pydantic(self, proto_msg) -> "MessageCreate":
        """Convert protobuf MessageCreate to Pydantic MessageCreate."""
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import MessageCreate

        if proto_msg.role == proto_msg.ROLE_USER:
            role = MessageRole.user
        elif proto_msg.role == proto_msg.ROLE_SYSTEM:
            role = MessageRole.system
        else:
            role = MessageRole.user

        content = proto_msg.text_content if proto_msg.HasField("text_content") else ""

        kwargs = {}
        if hasattr(proto_msg, "external_message_id") and proto_msg.HasField("external_message_id"):
            kwargs["external_message_id"] = proto_msg.external_message_id
        if hasattr(proto_msg, "message_occurred_at") and proto_msg.HasField("message_occurred_at"):
            kwargs["message_occurred_at"] = proto_msg.message_occurred_at

        return MessageCreate(
            role=role,
            content=content,
            name=proto_msg.name if proto_msg.HasField("name") else None,
            otid=proto_msg.otid if proto_msg.HasField("otid") else None,
            sender_id=proto_msg.sender_id if proto_msg.HasField("sender_id") else None,
            group_id=proto_msg.group_id if proto_msg.HasField("group_id") else None,
            filter_tags=None,
            **kwargs,
        )

    @staticmethod
    def _convert_proto_source_message_to_dict(proto_msg) -> dict:
        """Convert a protobuf MessageCreate from source_messages to a plain dict.

        Unlike _convert_proto_message_to_pydantic (which produces Pydantic
        MessageCreate objects for agent processing), this returns a plain dict
        because source_messages need to preserve the "assistant" role which
        Pydantic MessageCreate doesn't allow (its role field is
        Literal["user", "system"]).

        The returned dict is consumed by _persist_memory_source() →
        normalize_message() which accepts dicts with any string role.
        """
        _ROLE_MAP = {
            proto_msg.ROLE_USER: "user",
            proto_msg.ROLE_SYSTEM: "system",
            proto_msg.ROLE_ASSISTANT: "assistant",
        }
        role = _ROLE_MAP.get(proto_msg.role, "user")
        content = proto_msg.text_content if proto_msg.HasField("text_content") else ""

        result = {"role": role, "content": content}

        if hasattr(proto_msg, "external_message_id") and proto_msg.HasField("external_message_id"):
            result["external_message_id"] = proto_msg.external_message_id
        if hasattr(proto_msg, "message_occurred_at") and proto_msg.HasField("message_occurred_at"):
            result["message_occurred_at"] = proto_msg.message_occurred_at
        if hasattr(proto_msg, "message_metadata") and proto_msg.message_metadata:
            result["metadata"] = MessageToDict(proto_msg.message_metadata)

        return result

    def set_server(self, server: Any) -> None:
        """Set or update the server instance."""
        self._server = server
        logger.info("Updated worker server instance")

    async def process_external_message(self, message: QueueMessage) -> None:
        """
        Process a message that was consumed by an external Kafka consumer.

        Args:
            message: QueueMessage protobuf already consumed from Kafka
        """
        logger.debug(
            "Processing externally consumed message: agent_id=%s, user_id=%s",
            message.agent_id,
            message.user_id if message.HasField("user_id") else "None",
        )
        await self._process_message_async(message)

    async def _process_message_async(self, message: QueueMessage) -> None:
        """Process a queue message by calling server.send_messages()."""
        try:
            trace_restored = restore_trace_from_queue_message(message)
            if trace_restored:
                logger.debug("Restored trace context from queue message for processing")

            server = self._server

            if server is None:
                logger.warning(
                    "No server available - skipping message: agent_id=%s, input_messages_count=%s",
                    message.agent_id,
                    len(message.input_messages),
                )
                return

            langfuse = get_langfuse_client()
            trace_context = get_trace_context()
            trace_id = trace_context.get("trace_id") if trace_context else None
            parent_span_id = trace_context.get("observation_id") if trace_context else None
            logger.debug(f"Queue worker trace context: trace_id={trace_id}, parent_span_id={parent_span_id}")

            client_id = message.client_id if message.client_id else None
            if not client_id:
                raise ValueError(f"Queue message for agent {message.agent_id} missing required client_id")

            input_messages = [self._convert_proto_message_to_pydantic(msg) for msg in message.input_messages]
            chaining = message.chaining if message.HasField("chaining") else True
            user_id = message.user_id if message.HasField("user_id") else None

            async def _resolve_actor_and_user():
                actor = await server.client_manager.get_client_by_id(client_id)
                if not actor:
                    raise ValueError(f"Client with id={client_id} not found in database")

                user_manager = UserManager()
                if user_id:
                    try:
                        user = await user_manager.get_user_by_id(user_id)
                    except Exception:
                        logger.info(
                            "User with id=%s not found, auto-creating with organization_id=%s",
                            user_id,
                            actor.organization_id,
                        )
                        from mirix.schemas.user import User as PydanticUser

                        try:
                            user = await user_manager.create_user(
                                pydantic_user=PydanticUser(
                                    id=user_id,
                                    name=user_id,
                                    organization_id=actor.organization_id,
                                    timezone=user_manager.DEFAULT_TIME_ZONE,
                                    status="active",
                                    is_deleted=False,
                                    is_admin=False,
                                )
                            )
                            logger.info(
                                "Auto-created user: %s in organization: %s",
                                user_id,
                                actor.organization_id,
                            )
                        except Exception as create_error:
                            logger.error(
                                "Failed to auto-create user with id=%s: %s. Falling back to admin user.",
                                user_id,
                                create_error,
                            )
                            user = await user_manager.get_admin_user()
                    return actor, reconcile_user_org_to_actor(user, actor)
                user = await user_manager.get_admin_user()
                return actor, reconcile_user_org_to_actor(user, actor)

            actor, user = await _resolve_actor_and_user()

            # Extract filter_tags from protobuf Struct (deep conversion to avoid ListValue/Value remnants)
            filter_tags = None
            if message.HasField("filter_tags") and message.filter_tags:
                filter_tags = MessageToDict(message.filter_tags)

            use_cache = message.use_cache if message.HasField("use_cache") else True
            occurred_at = message.occurred_at if message.HasField("occurred_at") else None

            # Extract block_filter_tags (deep conversion to native Python types)
            block_filter_tags = None
            if hasattr(message, "block_filter_tags") and message.block_filter_tags:
                try:
                    block_filter_tags = MessageToDict(message.block_filter_tags)
                except Exception as e:
                    raise ValueError("block_filter_tags was provided but could not be parsed as a dict") from e

            block_filter_tags_update_mode = (
                message.block_filter_tags_update_mode if message.HasField("block_filter_tags_update_mode") else "merge"
            )

            # Extract memory source fields if present on the protobuf message
            memory_source_id = (
                message.memory_source_id
                if hasattr(message, "memory_source_id") and message.HasField("memory_source_id")
                else None
            )
            external_id = (
                message.external_id if hasattr(message, "external_id") and message.HasField("external_id") else None
            )
            external_thread_id = (
                message.external_thread_id
                if hasattr(message, "external_thread_id") and message.HasField("external_thread_id")
                else None
            )
            source_type = (
                message.source_type if hasattr(message, "source_type") and message.HasField("source_type") else None
            )
            source_system = (
                message.source_system
                if hasattr(message, "source_system") and message.HasField("source_system")
                else None
            )
            source_metadata = None
            if hasattr(message, "source_metadata") and message.source_metadata:
                try:
                    source_metadata = MessageToDict(message.source_metadata)
                except Exception:
                    pass
            summary = message.summary if hasattr(message, "summary") and message.HasField("summary") else None
            summarize = message.summarize if hasattr(message, "summarize") and message.HasField("summarize") else False

            # Extract original per-turn messages for source_message persistence.
            #
            # These are converted to plain dicts (not Pydantic MessageCreate) because
            # MessageCreate.role is Literal["user", "system"] and can't hold "assistant".
            # The dicts go straight to _persist_memory_source() → normalize_message()
            # which accepts dicts with string roles.
            source_messages = None
            if hasattr(message, "source_messages") and message.source_messages:
                source_messages = [self._convert_proto_source_message_to_dict(msg) for msg in message.source_messages]

            # Extract direct_writes — each entry tells the meta-agent to call
            # the registered handler for memory_type instead of dispatching
            # sub-agents via the LLM. Payloads are opaque JSON blobs here;
            # the handler's signature validates their shape at call time.
            direct_writes = None
            if hasattr(message, "direct_writes") and message.direct_writes:
                import json as _json

                direct_writes = [
                    {"memory_type": w.memory_type, "payload": _json.loads(w.payload_json)}
                    for w in message.direct_writes
                ]

            # Log the processing
            logger.info(
                "Processing message via server: agent_id=%s, client_id=%s (from actor), user_id=%s, input_messages_count=%s, use_cache=%s, filter_tags=%s, occurred_at=%s",
                message.agent_id,
                actor.id,
                user_id,
                len(input_messages),
                use_cache,
                filter_tags,
                occurred_at,
            )

            async def _do_send_messages():
                return await server.send_messages(
                    actor=actor,
                    agent_id=message.agent_id,
                    input_messages=input_messages,
                    chaining=chaining,
                    user=user,
                    filter_tags=filter_tags,
                    block_filter_tags=block_filter_tags,
                    block_filter_tags_update_mode=block_filter_tags_update_mode,
                    use_cache=use_cache,
                    occurred_at=occurred_at,
                    memory_source_id=memory_source_id,
                    external_id=external_id,
                    external_thread_id=external_thread_id,
                    source_type=source_type,
                    source_system=source_system,
                    source_metadata=source_metadata,
                    summary=summary,
                    summarize=summarize,
                    source_messages=source_messages,
                    direct_writes=direct_writes,
                )

            if langfuse and trace_id:
                from typing import cast

                from langfuse.types import TraceContext

                from mirix.observability.context import set_trace_context

                trace_context_dict: dict = {"trace_id": trace_id}

                with langfuse.start_as_current_observation(
                    name="Meta Agent",
                    as_type="agent",
                    trace_context=cast(TraceContext, trace_context_dict),
                    metadata={
                        "agent_id": message.agent_id,
                        "message_count": len(input_messages),
                        "source": "queue_worker",
                        # TID on the root worker span so every span in this
                        # save's tree can be correlated back to the request.
                        "intuit_tid": get_intuit_tid(),
                    },
                ) as span:
                    mark_observation_as_child(span)

                    # Surface the TID at the TRACE level (tag = filterable in the
                    # Langfuse dashboard, metadata = visible) so a failed save's
                    # trace shows its TID for a log pivot. This is the worker
                    # trace, decoupled from the HTTP-entry trace, so it needs its
                    # own trace-level TID independent of the span metadata above.
                    _worker_tid = get_intuit_tid()
                    if _worker_tid:
                        try:
                            langfuse.update_current_trace(
                                tags=[f"tid:{_worker_tid}"],
                                metadata={"intuit_tid": _worker_tid},
                            )
                        except Exception as e:
                            logger.debug("Failed to set TID on worker trace: %s", e)

                    span_observation_id = getattr(span, "id", None)
                    if span_observation_id:
                        set_trace_context(
                            trace_id=trace_id,
                            observation_id=span_observation_id,
                            user_id=trace_context.get("user_id"),
                            session_id=trace_context.get("session_id"),
                        )
                    usage = await _do_send_messages()
            else:
                usage = await _do_send_messages()

            logger.debug(
                "Successfully processed message: agent_id=%s, usage=%s",
                message.agent_id,
                usage.model_dump() if usage else "None",
            )

        except Exception as e:
            logger.error(
                "Error processing message for agent_id=%s: %s",
                message.agent_id,
                e,
                exc_info=True,
            )
        finally:
            clear_trace_context()
            # TID is tracked independently of trace context; clear it too so it
            # never leaks from one message into the next on a reused worker task.
            clear_intuit_tid()

    async def _consume_loop(self) -> None:
        """Async consume loop running as an asyncio.Task in the main event loop."""
        partition_info = f", partition={self._partition_id}" if self._partition_id is not None else ""
        logger.info("Queue worker task started%s", partition_info)

        while self._running:
            try:
                if self._partition_id is not None and hasattr(self.queue, "get_from_partition"):
                    message = await self.queue.get_from_partition(self._partition_id, timeout=1.0)
                else:
                    message = await self.queue.get(timeout=1.0)

                logger.debug(
                    "Received message%s: agent_id=%s, user_id=%s, input_messages_count=%s",
                    partition_info,
                    message.agent_id,
                    message.user_id if message.HasField("user_id") else "None",
                    len(message.input_messages),
                )

                await self._process_message_async(message)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("Queue worker task cancelled%s", partition_info)
                break
            except Exception as e:
                if type(e).__name__ in ["Empty", "StopIteration"]:
                    continue
                logger.error("Error in message consumption loop: %s", e, exc_info=True)

    async def start(self) -> None:
        """Start the background worker as an asyncio.Task."""
        if self._running:
            logger.warning("Queue worker already running")
            return

        partition_info = f" (partition {self._partition_id})" if self._partition_id is not None else ""
        logger.info("Starting queue worker task%s...", partition_info)
        self._running = True

        task_name = f"QueueWorker-{self._partition_id}" if self._partition_id is not None else "QueueWorker"
        self._task = asyncio.create_task(self._consume_loop(), name=task_name)

        logger.info("Queue worker task%s started successfully", partition_info)

    async def stop(self, close_queue: bool = True) -> None:
        """
        Stop the background worker task.

        Args:
            close_queue: Whether to close the queue resources. Set to False
                        when multiple workers share the same queue.
        """
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if close_queue:
            await self.queue.close()
