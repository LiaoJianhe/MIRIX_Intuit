"""Manager for MemorySource CRUD operations."""

from datetime import datetime, timezone
from typing import Optional, Union

from sqlalchemy import select, update

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

    def _get_cache_provider(self):
        """Get the cache provider, returning None if unavailable."""
        try:
            from mirix.database.cache_provider import get_cache_provider

            return get_cache_provider()
        except Exception:
            return None

    async def _cache_write(self, memory_source: PydanticMemorySource) -> None:
        """Best-effort cache write. Never raises."""
        try:
            cache_provider = self._get_cache_provider()
            if cache_provider:
                from mirix.settings import settings

                cache_key = f"{cache_provider.MEMORY_SOURCE_PREFIX}{memory_source.id}"
                data = memory_source.model_dump(mode="json")
                await cache_provider.set_json(cache_key, data, ttl=settings.redis_ttl_default)
                logger.debug("Cached memory source %s", memory_source.id)
        except Exception as e:
            logger.warning("Cache write failed for memory source %s: %s", memory_source.id, e)

    async def _read_from_db(self, memory_source_id: str) -> Optional[PydanticMemorySource]:
        """Read directly from DB, bypassing cache."""
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
            return record.to_pydantic()

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
    ) -> Optional[PydanticMemorySource]:
        """Create a memory source record using INSERT ON CONFLICT DO NOTHING.

        Returns the record (whether just inserted or pre-existing).
        Caches the result either way.
        """
        async with self.session_maker() as session:
            now = datetime.now(timezone.utc)
            occurred_at = parse_occurred_at(occurred_at)

            await MemorySourceModel.create_or_ignore(
                db_session=session,
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

        # Fetch from DB and populate cache (whether just inserted or pre-existing)
        result = await self._read_from_db(memory_source_id)
        if result:
            await self._cache_write(result)
        return result

    @enforce_types
    async def get_by_id(self, memory_source_id: str) -> Optional[PydanticMemorySource]:
        """Fetch a memory source by ID with cache-aside pattern."""
        # Try cache first
        try:
            cache_provider = self._get_cache_provider()
            if cache_provider:
                cache_key = f"{cache_provider.MEMORY_SOURCE_PREFIX}{memory_source_id}"
                cached_data = await cache_provider.get_json(cache_key)
                if cached_data:
                    logger.debug("Cache HIT for memory source %s", memory_source_id)
                    return PydanticMemorySource(**cached_data)
        except Exception as e:
            logger.warning("Cache read failed for memory source %s: %s", memory_source_id, e)

        # Fall back to DB and populate cache
        result = await self._read_from_db(memory_source_id)
        if result:
            await self._cache_write(result)
        return result

    @enforce_types
    async def mark_processing_complete(self, memory_source_id: str) -> None:
        """Set processing_complete = True after all agents finish successfully.

        Updates DB then overwrites cache with the updated record.
        """
        async with self.session_maker() as session:
            await session.execute(
                update(MemorySourceModel)
                .where(MemorySourceModel.id == memory_source_id)
                .values(
                    processing_complete=True,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            logger.info("Marked memory source %s as processing complete", memory_source_id)

        # Read fresh from DB and overwrite cache
        result = await self._read_from_db(memory_source_id)
        if result:
            await self._cache_write(result)
