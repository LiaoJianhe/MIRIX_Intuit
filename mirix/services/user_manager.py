import asyncio
from typing import List, Optional, Tuple

from mirix.log import get_logger
from mirix.orm.errors import NoResultFound
from mirix.orm.organization import Organization as OrganizationModel
from mirix.orm.user import User as UserModel
from mirix.schemas.user import User as PydanticUser
from mirix.schemas.user import UserUpdate
from mirix.services.organization_manager import OrganizationManager
from mirix.utils import enforce_types

logger = get_logger(__name__)


class UserManager:
    """Manager class to handle business logic related to Users."""

    ADMIN_USER_NAME = "admin_user"
    ADMIN_USER_ID = "user-00000000-0000-4000-8000-000000000000"
    DEFAULT_USER_NAME = "default_user"  # Organization-specific default user for block templates
    DEFAULT_TIME_ZONE = "UTC (UTC+00:00)"

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_admin_user(self, org_id: str = OrganizationManager.DEFAULT_ORG_ID) -> PydanticUser:
        """Create the admin user."""
        with self.session_maker() as session:
            # Make sure the org id exists
            try:
                OrganizationModel.read(db_session=session, identifier=org_id)
            except NoResultFound:
                raise ValueError(f"No organization with {org_id} exists in the organization table.")

            # Try to retrieve the user
            try:
                user = UserModel.read(db_session=session, identifier=self.ADMIN_USER_ID)
            except NoResultFound:
                # If it doesn't exist, make it
                user = UserModel(
                    id=self.ADMIN_USER_ID,
                    name=self.ADMIN_USER_NAME,
                    status="active",
                    timezone=self.DEFAULT_TIME_ZONE,
                    organization_id=org_id,
                    is_admin=True,
                )
                user.create(session)

            return user.to_pydantic()

    @enforce_types
    def create_user(self, pydantic_user: PydanticUser) -> PydanticUser:
        """Create a new user if it doesn't already exist (with Redis caching).

        Args:
            pydantic_user: The user data
        """
        with self.session_maker() as session:
            user_data = pydantic_user.model_dump()
            new_user = UserModel(**user_data)
            new_user.create_with_redis(session, actor=None)  # Auto-caches to Redis
            return new_user.to_pydantic()

    @enforce_types
    def update_user(self, user_update: UserUpdate) -> PydanticUser:
        """Update user details (with cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(db_session=session, identifier=user_update.id)

            # Update only the fields that are provided in UserUpdate
            update_data = user_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_user, key, value)

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    def update_user_timezone(self, timezone_str: str, user_id: str) -> PydanticUser:
        """Update the timezone of a user (with cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(db_session=session, identifier=user_id)

            # Update the timezone
            existing_user.timezone = timezone_str

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    def update_user_status(self, user_id: str, status: str) -> PydanticUser:
        """Update the status of a user (with cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(db_session=session, identifier=user_id)

            # Update the status
            existing_user.status = status

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    async def delete_user_by_id(self, user_id: str):
        """
        Soft delete a user and cascade soft delete to all associated records using memory managers.

        Cleanup workflow:
        1. Soft delete all memory records using memory managers:
           - Episodic memories
           - Semantic memories
           - Procedural memories
           - Resource memories
           - Knowledge vault items
           - Messages
           - Blocks

        2. Database (PostgreSQL):
           - Set user.is_deleted = True

        3. Redis Cache:
           - Update user hash with is_deleted=true
           - Memory cache entries updated by managers with is_deleted=true

        Args:
            user_id: ID of the user to soft delete
        """
        from mirix.log import get_logger

        logger = get_logger(__name__)
        logger.info("Soft deleting user %s and all associated records using memory managers...", user_id)

        # Import memory managers
        from mirix.services.block_manager import BlockManager
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager
        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
        from mirix.services.message_manager import MessageManager
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager
        from mirix.services.resource_memory_manager import ResourceMemoryManager
        from mirix.services.semantic_memory_manager import SemanticMemoryManager

        # 1. Soft delete all memory records using memory managers
        episodic_manager = EpisodicMemoryManager()
        semantic_manager = SemanticMemoryManager()
        procedural_manager = ProceduralMemoryManager()
        resource_manager = ResourceMemoryManager()
        knowledge_manager = KnowledgeVaultManager()
        message_manager = MessageManager()
        block_manager = BlockManager()

        episodic_count = await episodic_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d episodic memories for user %s", episodic_count, user_id)

        semantic_count = await semantic_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d semantic memories for user %s", semantic_count, user_id)

        procedural_count = await procedural_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d procedural memories for user %s", procedural_count, user_id)

        resource_count = await resource_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d resource memories for user %s", resource_count, user_id)

        knowledge_count = await knowledge_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d knowledge vault items for user %s", knowledge_count, user_id)

        message_count = await message_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d messages for user %s", message_count, user_id)

        block_count = await block_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d blocks for user %s", block_count, user_id)

        # 2. Soft delete user
        def _db_soft_delete():
            with self.session_maker() as session:
                user = UserModel.read(db_session=session, identifier=user_id)
                if not user:
                    return False
                user.is_deleted = True
                user.set_updated_at()
                session.commit()
                logger.info("Soft deleted user %s from database", user_id)
                return True

        found = await asyncio.to_thread(_db_soft_delete)
        if not found:
            logger.warning("User %s not found", user_id)
            return

        # 3. Update cache to reflect soft delete
        try:
            from mirix.database.cache_provider import acache_delete, acache_update_hash_field, get_cache_provider

            cache_provider = get_cache_provider()
            if cache_provider:
                user_key = f"{cache_provider.USER_PREFIX}{user_id}"
                ok = await acache_update_hash_field(user_key, "is_deleted", "true")
                if not ok:
                    await acache_delete(user_key)
                logger.debug("Updated user %s in cache (is_deleted=true)", user_id)

                logger.info(
                    "User %s and all associated records soft deleted: "
                    "%d episodic, %d semantic, %d procedural, %d resource, %d knowledge_vault, %d messages, %d blocks",
                    user_id,
                    episodic_count,
                    semantic_count,
                    procedural_count,
                    resource_count,
                    knowledge_count,
                    message_count,
                    block_count,
                )
        except Exception as e:
            logger.warning("Failed to update Redis cache for user %s: %s", user_id, e)

    async def delete_memories_by_user_id(self, user_id: str):
        """
        Hard delete memories, messages, and blocks for a user using memory managers' bulk delete.

        This permanently removes data records while preserving the user record.
        Uses optimized bulk delete methods in each manager for efficient deletion.

        Cleanup workflow:
        1. Call each memory manager's delete_by_user_id() method
           - EpisodicMemoryManager.delete_by_user_id()
           - SemanticMemoryManager.delete_by_user_id()
           - ProceduralMemoryManager.delete_by_user_id()
           - ResourceMemoryManager.delete_by_user_id()
           - KnowledgeVaultManager.delete_by_user_id()
           - MessageManager.delete_by_user_id()
           - BlockManager.delete_by_user_id()
        2. Each manager handles:
           - Bulk database deletion
           - Redis cache cleanup
           - Business logic
        3. PRESERVE: user record

        Args:
            user_id: ID of the user whose memories to delete
        """
        from mirix.log import get_logger

        logger = get_logger(__name__)
        logger.info("Bulk deleting memories for user %s using memory managers (preserving user record)...", user_id)

        # Import managers
        from mirix.services.block_manager import BlockManager
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager
        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
        from mirix.services.message_manager import MessageManager
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager
        from mirix.services.resource_memory_manager import ResourceMemoryManager
        from mirix.services.semantic_memory_manager import SemanticMemoryManager

        # Initialize managers
        episodic_manager = EpisodicMemoryManager()
        semantic_manager = SemanticMemoryManager()
        procedural_manager = ProceduralMemoryManager()
        resource_manager = ResourceMemoryManager()
        knowledge_manager = KnowledgeVaultManager()
        message_manager = MessageManager()
        block_manager = BlockManager()

        # Use managers' bulk delete methods
        try:
            episodic_count = await episodic_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d episodic memories", episodic_count)

            semantic_count = await semantic_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d semantic memories", semantic_count)

            procedural_count = await procedural_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d procedural memories", procedural_count)

            resource_count = await resource_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d resource memories", resource_count)

            knowledge_count = await knowledge_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d knowledge vault items", knowledge_count)

            message_count = await message_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d messages", message_count)

            block_count = await block_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d blocks", block_count)

            # Clear message_ids from ALL agents in PostgreSQL
            def _db_clear_agents():
                with self.session_maker() as session:
                    from mirix.orm.agent import Agent as AgentModel

                    agents = session.query(AgentModel).all()
                    for agent in agents:
                        if agent.message_ids and len(agent.message_ids) > 1:
                            agent.message_ids = [agent.message_ids[0]]
                    session.commit()
                    logger.debug(
                        "Cleared conversation message_ids from %d agents in PostgreSQL (kept system messages)",
                        len(agents),
                    )

            await asyncio.to_thread(_db_clear_agents)

            # Invalidate agent caches that might reference deleted messages for this user
            from mirix.database.cache_provider import acache_delete_many, get_cache_provider
            from mirix.orm.agent import Agent as AgentModel

            cache_provider = get_cache_provider()
            if cache_provider:
                def _get_agent_ids():
                    with self.session_maker() as session:
                        return [row[0] for row in session.query(AgentModel.id).all()]

                agent_ids = await asyncio.to_thread(_get_agent_ids)
                if agent_ids:
                    agent_keys = [f"{cache_provider.AGENT_PREFIX}{aid}" for aid in agent_ids]
                    BATCH_SIZE = 1000
                    invalidated_count = 0
                    for i in range(0, len(agent_keys), BATCH_SIZE):
                        batch = agent_keys[i : i + BATCH_SIZE]
                        invalidated_count += await acache_delete_many(batch)
                    if invalidated_count > 0:
                        logger.debug(
                            "Invalidated %d agent caches due to user deletion",
                            invalidated_count,
                        )

            logger.info(
                "Bulk deleted all memories for user %s: "
                "%d episodic, %d semantic, %d procedural, %d resource, %d knowledge_vault, %d messages, %d blocks "
                "(user record preserved)",
                user_id,
                episodic_count,
                semantic_count,
                procedural_count,
                resource_count,
                knowledge_count,
                message_count,
                block_count,
            )
        except Exception as e:
            logger.error("Failed to bulk delete memories for user %s: %s", user_id, e)
            raise

    @enforce_types
    def get_user_by_id(self, user_id: str, use_cache: bool = True) -> PydanticUser:
        """Fetch a user by ID (with cache - Redis or IPS Cache)."""
        from mirix.log import get_logger

        logger = get_logger(__name__)
        cache_provider = None
        try:
            from mirix.database.cache_provider import (
                get_cache_provider,
                sync_cache_get_hash,
                sync_cache_set_hash,
            )

            cache_provider = get_cache_provider() if use_cache else None

            if cache_provider:
                cache_key = f"{cache_provider.USER_PREFIX}{user_id}"
                cached_data = sync_cache_get_hash(cache_key)
                if cached_data:
                    logger.debug("Cache HIT for user %s", user_id)
                    return PydanticUser(**cached_data)
        except Exception as e:
            logger.warning("Cache read failed for user %s: %s", user_id, e)

        with self.session_maker() as session:
            user = UserModel.read(db_session=session, identifier=user_id)
            pydantic_user = user.to_pydantic()

        # Cache after session is closed (no PG connection held during cache I/O)
        try:
            if cache_provider:
                from mirix.settings import settings

                cache_key = f"{cache_provider.USER_PREFIX}{user_id}"
                data = pydantic_user.model_dump(mode="json")
                sync_cache_set_hash(
                    cache_key, data, ttl=settings.redis_ttl_users
                )
                logger.debug("Populated cache for user %s", user_id)
        except Exception as e:
            logger.warning("Failed to populate cache for user %s: %s", user_id, e)

        return pydantic_user

    @enforce_types
    def get_admin_user(self) -> PydanticUser:
        """Fetch the admin user, creating it if it doesn't exist."""
        try:
            return self.get_user_by_id(self.ADMIN_USER_ID)
        except NoResultFound:
            # Admin user doesn't exist, create it
            # First ensure the default organization exists
            from mirix.services.organization_manager import OrganizationManager

            org_mgr = OrganizationManager()
            org_mgr.get_default_organization()  # Auto-creates if missing
            return self.create_admin_user(org_id=OrganizationManager.DEFAULT_ORG_ID)

    @enforce_types
    def get_or_create_org_default_user(self, org_id: str) -> PydanticUser:
        """
        Get or create the default template user for an organization.
        This user serves as the template for copying blocks to new users.

        Args:
            org_id: Organization ID

        Returns:
            PydanticUser: The default user for this organization
        """
        # Try to find existing default user for this org
        with self.session_maker() as session:
            try:
                user = (
                    session.query(UserModel)
                    .filter(
                        UserModel.name == self.DEFAULT_USER_NAME,
                        UserModel.organization_id == org_id,
                        UserModel.is_deleted == False,
                    )
                    .first()
                )

                if user:
                    logger.debug("Found existing default user %s for organization %s", user.id, org_id)
                    return user.to_pydantic()
            except Exception as e:
                logger.debug("Error finding default user: %s", e)

        # Default user doesn't exist, create it
        logger.info("Creating default template user for organization %s", org_id)

        # Generate a deterministic user_id for the default user
        default_user_id = f"user-default-{org_id}"

        try:
            # Try to get by ID first (in case it exists with that ID)
            cached_user = self.get_user_by_id(default_user_id)
            # Guard against stale cache: verify the user actually exists in the DB.
            # When a remote cache (e.g. IPS Cache) retains data across DB rebuilds,
            # the cache can return a user that no longer exists in PostgreSQL.
            with self.session_maker() as verify_session:
                db_check = verify_session.query(UserModel).filter(UserModel.id == default_user_id).first()
                if db_check is None:
                    logger.warning(
                        "Stale cache detected for user %s — exists in cache but not in DB. Proceeding to create.",
                        default_user_id,
                    )
                    raise NoResultFound("Stale cache - user not in DB")
            return cached_user
        except NoResultFound:
            pass

        # Create the default user, handling the race where another thread creates it first.
        try:
            with self.session_maker() as session:
                user = UserModel(
                    id=default_user_id,
                    name=self.DEFAULT_USER_NAME,
                    status="active",
                    timezone=self.DEFAULT_TIME_ZONE,
                    organization_id=org_id,
                )
                user.create(session)
                logger.info("Created default template user %s for organization %s", default_user_id, org_id)
                return user.to_pydantic()
        except Exception as e:
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info("Default user %s created by another thread, reading from DB", default_user_id)
                return self.get_user_by_id(default_user_id, use_cache=False)
            raise

    @enforce_types
    def get_user_or_admin(self, user_id: Optional[str] = None):
        """Fetch the user or admin user."""
        if not user_id:
            return self.get_admin_user()

        try:
            return self.get_user_by_id(user_id=user_id)
        except NoResultFound:
            return self.get_admin_user()

    @enforce_types
    def list_users(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        organization_id: Optional[str] = None,
    ) -> List[PydanticUser]:
        """List users with pagination using cursor (id) and limit.

        Args:
            cursor: Cursor for pagination
            limit: Maximum number of users to return
            organization_id: Filter by organization ID
        """
        with self.session_maker() as session:
            query = session.query(UserModel).filter(UserModel.is_deleted == False)

            if organization_id:
                query = query.filter(UserModel.organization_id == organization_id)

            query = query.order_by(UserModel.created_at.desc())

            if cursor:
                query = query.filter(UserModel.id < cursor)

            if limit:
                query = query.limit(limit)

            results = query.all()
            return [user.to_pydantic() for user in results]
