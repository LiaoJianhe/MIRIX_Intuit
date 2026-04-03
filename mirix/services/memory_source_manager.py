"""Manager for MemorySource CRUD operations."""

from datetime import datetime, timezone
from typing import Optional, Union

from sqlalchemy import select

from mirix.log import get_logger
from mirix.orm.memory_source import MemorySource as MemorySourceModel
from mirix.schemas.memory_source import MemorySource as PydanticMemorySource
from mirix.utils import enforce_types

logger = get_logger(__name__)


def parse_occurred_at(value: Union[str, datetime, None]) -> Optional[datetime]:
    """Parse an occurred_at value that may be an ISO 8601 string, datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Could not parse occurred_at value: %s", value)
            return None
    return None


class MemorySourceManager:
    """Manager for memory source persistence with INSERT ON CONFLICT DO NOTHING semantics."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def create(
        self,
        memory_source_id: str,
        client_id: str,
        user_id: str,
        organization_id: str,
        source_type: str = "conversation",
        external_id: Optional[str] = None,
        external_thread_id: Optional[str] = None,
        source_system: Optional[str] = None,
        source_metadata: Optional[dict] = None,
        occurred_at: Union[str, datetime, None] = None,
        summary: Optional[str] = None,
        summary_source: Optional[str] = None,
        batch_hash: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[PydanticMemorySource]:
        """Create a memory source record using INSERT ON CONFLICT DO NOTHING.

        Returns the record (whether just inserted or pre-existing).
        Cache write handled by the ORM layer via create_or_ignore_with_redis.
        """
        async with self.session_maker() as session:
            now = datetime.now(timezone.utc)
            occurred_at = parse_occurred_at(occurred_at)

            await MemorySourceModel.create_or_ignore_with_redis(
                db_session=session,
                use_cache=use_cache,
                id=memory_source_id,
                client_id=client_id,
                user_id=user_id,
                organization_id=organization_id,
                source_type=source_type,
                external_id=external_id,
                external_thread_id=external_thread_id,
                source_system=source_system,
                source_metadata=source_metadata,
                occurred_at=occurred_at,
                summary=summary,
                summary_source=summary_source,
                batch_hash=batch_hash,
                processing_complete=False,
                created_at=now,
                updated_at=now,
                is_deleted=False,
            )

        # Return the record (from cache or DB via get_by_id)
        return await self.get_by_id(memory_source_id, use_cache=use_cache)

    @enforce_types
    async def get_by_id(self, memory_source_id: str, use_cache: bool = True) -> Optional[PydanticMemorySource]:
        """Fetch a memory source by ID with cache-aside pattern."""
        # Try cache first
        if use_cache:
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.MEMORY_SOURCE_PREFIX}{memory_source_id}"
                    cached_data = await cache_provider.get_json(cache_key)
                    if cached_data:
                        logger.debug("Cache HIT for memory source %s", memory_source_id)
                        return PydanticMemorySource(**cached_data)
            except Exception as e:
                logger.warning("Cache read failed for memory source %s: %s", memory_source_id, e)

        # Fall back to DB
        async with self.session_maker() as session:
            result = await session.execute(
                select(MemorySourceModel).where(
                    MemorySourceModel.id == memory_source_id,
                    ~MemorySourceModel.is_deleted,
                )
            )
            record = result.scalar_one_or_none()
            if record is None:
                return None
            pydantic_source = record.to_pydantic()

        # Populate cache on miss
        if use_cache:
            try:
                from mirix.database.cache_provider import get_cache_provider
                from mirix.settings import settings

                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.MEMORY_SOURCE_PREFIX}{memory_source_id}"
                    data = pydantic_source.model_dump(mode="json")
                    await cache_provider.set_json(cache_key, data, ttl=settings.redis_ttl_default)
                    logger.debug("Populated cache for memory source %s", memory_source_id)
            except Exception as e:
                logger.warning("Cache write failed for memory source %s: %s", memory_source_id, e)

        return pydantic_source

    @enforce_types
    async def mark_processing_complete(self, memory_source_id: str) -> None:
        """Set processing_complete = True after all agents finish successfully.

        Uses ORM update_with_redis for consistent cache handling.
        """
        async with self.session_maker() as session:
            record = await MemorySourceModel.read(db_session=session, identifier=memory_source_id)
            record.processing_complete = True
            await record.update_with_redis(session)
            logger.info("Marked memory source %s as processing complete", memory_source_id)
