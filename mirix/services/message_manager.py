import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from mirix.orm.errors import NoResultFound
from mirix.orm.message import Message as MessageModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.message import MessageUpdate
from mirix.services.utils import update_timezone
from mirix.utils import enforce_types


class MessageManager:
    """Manager class to handle business logic related to Messages."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @update_timezone
    @enforce_types
    def get_message_by_id(
        self, message_id: str, actor: PydanticClient, use_cache: bool = True
    ) -> Optional[PydanticMessage]:
        """Fetch a message by ID (with cache - Redis or IPS Cache)."""
        from mirix.log import get_logger

        logger = get_logger(__name__)
        cache_provider = None
        try:
            from mirix.database.cache_provider import (
                get_cache_provider,
                sync_cache_get_hash,
                sync_cache_set_hash,
            )

            cache_provider = get_cache_provider()

            if use_cache and cache_provider:
                cache_key = f"{cache_provider.MESSAGE_PREFIX}{message_id}"
                cached_data = sync_cache_get_hash(cache_key)
                if cached_data:
                    return PydanticMessage(**cached_data)
        except Exception as e:
            logger.warning("Cache read failed for message %s: %s", message_id, e)

        # Cache MISS or no cache - fetch from PostgreSQL
        with self.session_maker() as session:
            try:
                message = MessageModel.read(db_session=session, identifier=message_id, actor=actor)
                pydantic_message = message.to_pydantic()
            except NoResultFound:
                return None

        # Cache after session is closed (no PG connection held during cache I/O)
        try:
            if cache_provider:
                from mirix.settings import settings

                cache_key = f"{cache_provider.MESSAGE_PREFIX}{message_id}"
                data = pydantic_message.model_dump(mode="json")
                sync_cache_set_hash(
                    cache_key,
                    data,
                    ttl=settings.redis_ttl_messages,
                )
        except Exception as e:
            logger.warning("Failed to populate cache for message %s: %s", message_id, e)

        return pydantic_message

    @update_timezone
    @enforce_types
    def get_messages_by_ids(
        self, message_ids: List[str], actor: PydanticClient, use_cache: bool = True
    ) -> List[PydanticMessage]:
        """Fetch messages by ID and return them in the requested order."""
        # TODO: Add Redis pipeline support for batch retrieval when use_cache=True
        with self.session_maker() as session:
            results = MessageModel.list(
                db_session=session,
                id=message_ids,
                organization_id=actor.organization_id,
                limit=len(message_ids),
            )

            # Return messages in requested order, skipping any missing IDs
            # (messages may be missing due to concurrent summarization/cleanup by other workers)
            result_dict = {msg.id: msg.to_pydantic() for msg in results}
            return [result_dict[msg_id] for msg_id in message_ids if msg_id in result_dict]

    @enforce_types
    def create_message(
        self,
        pydantic_msg: PydanticMessage,
        actor: PydanticClient,
        use_cache: bool = True,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> PydanticMessage:
        """Create a new message (with Redis Hash caching).

        Args:
            pydantic_msg: The message data to create
            actor: Client performing the operation (for audit trail)
            use_cache: If True, cache in Redis. If False, skip caching.
            client_id: Optional client_id for data scoping (defaults to actor.id)
            user_id: Optional user_id for data scoping (defaults to None)
        """
        with self.session_maker() as session:
            # Set the organization id of the Pydantic message
            pydantic_msg.organization_id = actor.organization_id

            # Set client_id: use provided value, or from message, or from actor
            if client_id:
                pydantic_msg.client_id = client_id
            elif not pydantic_msg.client_id:
                pydantic_msg.client_id = actor.id

            # Set user_id: use provided value, or from message, or fallback to default user
            if user_id is not None:
                pydantic_msg.user_id = user_id
            elif not pydantic_msg.user_id:
                # Fallback: use default system user for client-level operations
                from mirix.services.user_manager import UserManager

                pydantic_msg.user_id = UserManager.ADMIN_USER_ID

            msg_data = pydantic_msg.model_dump()
            msg = MessageModel(**msg_data)
            msg.create_with_redis(session, actor=actor, use_cache=use_cache)
            return msg.to_pydantic()

    @enforce_types
    def create_many_messages(
        self,
        pydantic_msgs: List[PydanticMessage],
        actor: PydanticClient,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[PydanticMessage]:
        """Create multiple messages."""
        return [self.create_message(m, actor=actor, client_id=client_id, user_id=user_id) for m in pydantic_msgs]

    @enforce_types
    def update_message_by_id(
        self, message_id: str, message_update: MessageUpdate, actor: PydanticClient
    ) -> PydanticMessage:
        """
        Updates an existing record in the database with values from the provided record object.
        """
        with self.session_maker() as session:
            # Fetch existing message from database
            message = MessageModel.read(
                db_session=session,
                identifier=message_id,
                actor=actor,
            )

            # Some safety checks specific to messages
            if message_update.tool_calls and message.role != MessageRole.assistant:
                raise ValueError(
                    f"Tool calls {message_update.tool_calls} can only be added to assistant messages. Message {message_id} has role {message.role}."
                )
            if message_update.tool_call_id and message.role != MessageRole.tool:
                raise ValueError(
                    f"Tool call IDs {message_update.tool_call_id} can only be added to tool messages. Message {message_id} has role {message.role}."
                )

            # get update dictionary
            update_data = message_update.model_dump(exclude_unset=True, exclude_none=True)
            # Remove redundant update fields
            update_data = {key: value for key, value in update_data.items() if getattr(message, key) != value}

            for key, value in update_data.items():
                setattr(message, key, value)
            message.update_with_redis(db_session=session, actor=actor)  # Update PostgreSQL + Redis

            return message.to_pydantic()

    @enforce_types
    def delete_message_by_id(self, message_id: str, actor: PydanticClient) -> bool:
        """Delete a message (removes from cache)."""
        from mirix.database.cache_provider import (
            get_cache_provider,
            sync_cache_delete,
        )

        with self.session_maker() as session:
            try:
                msg = MessageModel.read(
                    db_session=session,
                    identifier=message_id,
                    actor=actor,
                )
                msg.hard_delete(session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Message with id {message_id} not found.")

        # Cache delete after session is closed (no PG connection held during cache I/O)
        cache_provider = get_cache_provider()
        if cache_provider:
            cache_key = f"{cache_provider.MESSAGE_PREFIX}{message_id}"
            sync_cache_delete(cache_key)

    @enforce_types
    async def delete_by_client_id(self, actor: PydanticClient) -> int:
        """
        Bulk delete all NON-SYSTEM messages for a client (removes from Redis cache).
        System messages are preserved as they are essential for agent functionality.
        Optimized with single DB query and batch Redis deletion.

        Args:
            actor: Client whose messages to delete (uses actor.id as client_id)

        Returns:
            Number of records deleted
        """
        from mirix.database.cache_provider import acache_delete_many, get_cache_provider
        from mirix.schemas.message import MessageRole

        def _db_delete():
            with self.session_maker() as session:
                message_ids = [
                    row[0]
                    for row in session.query(MessageModel.id)
                    .filter(
                        MessageModel.client_id == actor.id,
                        MessageModel.role != MessageRole.system,
                    )
                    .all()
                ]
                count = len(message_ids)
                if count == 0:
                    return 0, []
                session.query(MessageModel).filter(
                    MessageModel.client_id == actor.id, MessageModel.role != MessageRole.system
                ).delete(synchronize_session=False)
                session.commit()
                return count, message_ids

        count, message_ids = await asyncio.to_thread(_db_delete)
        if count == 0:
            return 0

        cache_provider = get_cache_provider()
        if cache_provider and message_ids:
            redis_keys = [f"{cache_provider.MESSAGE_PREFIX}{msg_id}" for msg_id in message_ids]
            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                await acache_delete_many(batch)

        return count

    async def soft_delete_by_client_id(self, actor: PydanticClient) -> int:
        """
        Bulk soft delete all messages for a client (updates Redis cache).

        Args:
            actor: Client whose messages to soft delete (uses actor.id as client_id)

        Returns:
            Number of records soft deleted
        """
        from mirix.database.cache_provider import acache_delete, acache_update_hash_field, get_cache_provider

        def _db_soft_delete():
            with self.session_maker() as session:
                messages = (
                    session.query(MessageModel)
                    .filter(MessageModel.client_id == actor.id, MessageModel.is_deleted == False)
                    .all()
                )
                count = len(messages)
                if count == 0:
                    return 0, []
                message_ids = [msg.id for msg in messages]
                for msg in messages:
                    msg.is_deleted = True
                    msg.set_updated_at()
                session.commit()
                return count, message_ids

        count, message_ids = await asyncio.to_thread(_db_soft_delete)
        if count == 0:
            return 0

        cache_provider = get_cache_provider()
        if cache_provider:
            for msg_id in message_ids:
                redis_key = f"{cache_provider.MESSAGE_PREFIX}{msg_id}"
                ok = await acache_update_hash_field(redis_key, "is_deleted", "true")
                if not ok:
                    await acache_delete(redis_key)

        return count

    async def soft_delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk soft delete all messages for a user (updates Redis cache).

        Args:
            user_id: ID of the user whose messages to soft delete

        Returns:
            Number of records soft deleted
        """
        from mirix.database.cache_provider import acache_delete, acache_update_hash_field, get_cache_provider

        def _db_soft_delete():
            with self.session_maker() as session:
                messages = (
                    session.query(MessageModel)
                    .filter(MessageModel.user_id == user_id, MessageModel.is_deleted == False)
                    .all()
                )
                count = len(messages)
                if count == 0:
                    return 0, []
                message_ids = [msg.id for msg in messages]
                for msg in messages:
                    msg.is_deleted = True
                    msg.set_updated_at()
                session.commit()
                return count, message_ids

        count, message_ids = await asyncio.to_thread(_db_soft_delete)
        if count == 0:
            return 0

        cache_provider = get_cache_provider()
        if cache_provider:
            for msg_id in message_ids:
                redis_key = f"{cache_provider.MESSAGE_PREFIX}{msg_id}"
                ok = await acache_update_hash_field(redis_key, "is_deleted", "true")
                if not ok:
                    await acache_delete(redis_key)

        return count

    async def delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk hard delete all NON-SYSTEM messages for a user (removes from Redis cache).
        System messages are preserved as they are essential for agent functionality.
        Optimized with single DB query and batch Redis deletion.

        Args:
            user_id: ID of the user whose messages to delete

        Returns:
            Number of records deleted
        """
        from mirix.database.cache_provider import acache_delete_many, get_cache_provider
        from mirix.schemas.message import MessageRole

        def _db_delete():
            with self.session_maker() as session:
                message_ids = [
                    row[0]
                    for row in session.query(MessageModel.id)
                    .filter(
                        MessageModel.user_id == user_id,
                        MessageModel.role != MessageRole.system,
                    )
                    .all()
                ]
                count = len(message_ids)
                if count == 0:
                    return 0, []
                session.query(MessageModel).filter(
                    MessageModel.user_id == user_id, MessageModel.role != MessageRole.system
                ).delete(synchronize_session=False)
                session.commit()
                return count, message_ids

        count, message_ids = await asyncio.to_thread(_db_delete)
        if count == 0:
            return 0

        cache_provider = get_cache_provider()
        if cache_provider and message_ids:
            redis_keys = [f"{cache_provider.MESSAGE_PREFIX}{msg_id}" for msg_id in message_ids]
            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                await acache_delete_many(batch)

        return count

    @enforce_types
    def size(
        self,
        actor: PydanticClient,
        role: Optional[MessageRole] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Get the total count of messages with optional filters.

        Args:
            actor: The client requesting the count
            role: The role of the message
        """
        with self.session_maker() as session:
            return MessageModel.size(
                db_session=session,
                actor=actor,
                role=role,
                user_id=user_id,
                agent_id=agent_id,
            )

    @update_timezone
    @enforce_types
    def list_user_messages_for_agent(
        self,
        agent_id: str,
        actor: Optional[PydanticClient] = None,
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        filters: Optional[Dict] = None,
        query_text: Optional[str] = None,
        ascending: bool = True,
    ) -> List[PydanticMessage]:
        """List user messages with flexible filtering and pagination options.

        Args:
            cursor: Cursor-based pagination - return records after this ID (exclusive)
            start_date: Filter records created after this date
            end_date: Filter records created before this date
            limit: Maximum number of records to return
            filters: Additional filters to apply
            query_text: Optional text to search for in message content

        Returns:
            List[PydanticMessage] - List of messages matching the criteria
        """
        message_filters = {"role": "user"}
        if filters:
            message_filters.update(filters)

        return self.list_messages_for_agent(
            agent_id=agent_id,
            actor=actor,
            cursor=cursor,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            filters=message_filters,
            query_text=query_text,
            ascending=ascending,
        )

    @update_timezone
    @enforce_types
    def list_messages_for_agent(
        self,
        agent_id: str,
        actor: Optional[PydanticClient] = None,
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        filters: Optional[Dict] = None,
        query_text: Optional[str] = None,
        ascending: bool = True,
        use_cache: bool = True,
    ) -> List[PydanticMessage]:
        """List messages with flexible filtering and pagination options.

        Args:
            cursor: Cursor-based pagination - return records after this ID (exclusive)
            start_date: Filter records created after this date
            end_date: Filter records created before this date
            limit: Maximum number of records to return
            filters: Additional filters to apply
            query_text: Optional text to search for in message content
            use_cache: Control Redis cache behavior (default: True)
                      Note: Currently this method only queries PostgreSQL.
                      Redis list retrieval for messages is not yet implemented.

        Returns:
            List[PydanticMessage] - List of messages matching the criteria
        """
        with self.session_maker() as session:
            # Start with base filters
            message_filters = {"agent_id": agent_id}
            if actor:
                message_filters.update({"organization_id": actor.organization_id})
            if filters:
                message_filters.update(filters)

            results = MessageModel.list(
                db_session=session,
                cursor=cursor,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                query_text=query_text,
                ascending=ascending,
                **message_filters,
            )

            return [msg.to_pydantic() for msg in results]

    @enforce_types
    def delete_detached_messages_for_agent(self, agent_id: str, actor: PydanticClient) -> int:
        """
        Delete messages that belong to an agent but are not in the agent's current message_ids list.

        This is useful for cleaning up messages that were removed from context during
        context window management but still exist in the database.

        Args:
            agent_id: The ID of the agent to clean up messages for
            actor: The user performing this action

        Returns:
            int: Number of messages deleted
        """
        with self.session_maker() as session:
            # First, get the agent to access its current message_ids
            from mirix.orm.agent import Agent as AgentModel

            try:
                agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            except NoResultFound:
                raise ValueError(f"Agent with id {agent_id} not found.")

            # Get current message_ids (messages that should be kept)
            current_message_ids = set(agent.message_ids or [])

            # Find all messages for this agent
            all_messages = MessageModel.list(
                db_session=session,
                agent_id=agent_id,
                organization_id=actor.organization_id,
                limit=None,  # Get all messages
            )

            # Identify detached messages (not in current message_ids)
            detached_messages = [msg for msg in all_messages if msg.id not in current_message_ids]
            deleted_ids = [msg.id for msg in detached_messages]

            for msg in detached_messages:
                msg.hard_delete(session, actor=actor)
            session.commit()
            deleted_count = len(deleted_ids)

        # Cache delete after session is closed (no PG connection held during cache I/O)
        from mirix.database.cache_provider import (
            get_cache_provider,
            sync_cache_delete,
        )

        cache_provider = get_cache_provider()
        if cache_provider:
            for msg_id in deleted_ids:
                redis_key = f"{cache_provider.MESSAGE_PREFIX}{msg_id}"
                sync_cache_delete(redis_key)
        return deleted_count

    @enforce_types
    def cleanup_all_detached_messages(self, actor: PydanticClient) -> Dict[str, int]:
        """
        Cleanup detached messages for all agents in the organization.

        Args:
            actor: The user performing this action

        Returns:
            Dict[str, int]: Dictionary mapping agent_id to number of messages deleted
        """
        from mirix.orm.agent import Agent as AgentModel

        with self.session_maker() as session:
            # Get all agents for this organization
            agents = AgentModel.list(db_session=session, organization_id=actor.organization_id, limit=None)

            cleanup_results = {}
            total_deleted = 0

            all_deleted_ids: List[str] = []
            for agent in agents:
                # Get current message_ids for this agent
                current_message_ids = set(agent.message_ids or [])

                # Find all messages for this agent
                all_messages = MessageModel.list(
                    db_session=session,
                    agent_id=agent.id,
                    organization_id=actor.organization_id,
                    limit=None,
                )

                # Identify and delete detached messages
                detached_messages = [msg for msg in all_messages if msg.id not in current_message_ids]
                deleted_ids = [msg.id for msg in detached_messages]
                all_deleted_ids.extend(deleted_ids)

                for msg in detached_messages:
                    msg.hard_delete(session)
                deleted_count = len(deleted_ids)
                cleanup_results[agent.id] = deleted_count
                total_deleted += deleted_count

            session.commit()
            cleanup_results["total"] = total_deleted

        # Cache delete after session is closed (no PG connection held during cache I/O)
        from mirix.database.cache_provider import (
            get_cache_provider,
            sync_cache_delete,
        )

        cache_provider = get_cache_provider()
        if cache_provider:
            for msg_id in all_deleted_ids:
                redis_key = f"{cache_provider.MESSAGE_PREFIX}{msg_id}"
                sync_cache_delete(redis_key)
        return cleanup_results
