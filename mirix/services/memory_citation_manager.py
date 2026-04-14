"""Manager for MemoryCitation CRUD operations."""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func, or_, select, tuple_

from mirix.log import get_logger
from mirix.orm.memory_citation import MemoryCitation as MemoryCitationModel
from mirix.schemas.memory_citation import MemoryCitation as PydanticMemoryCitation
from mirix.utils import enforce_types

logger = get_logger(__name__)


class MemoryCitationManager:
    """Manager for memory citation persistence with INSERT ON CONFLICT DO NOTHING semantics."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    def _exists_cache_key(self, memory_source_id: str, memory_type: str, memory_id: str) -> str:
        """Cache key for the check_exists lookup (hot-path query for citation-level dedup)."""
        return f"citation_exists:{memory_source_id}:{memory_type}:{memory_id}"

    async def _cache_exists(self, memory_source_id: str, memory_type: str, memory_id: str) -> None:
        """Cache a positive exists result. Never raises."""
        try:
            from mirix.database.cache_provider import get_cache_provider
            from mirix.settings import settings

            cache_provider = get_cache_provider()
            if cache_provider:
                key = self._exists_cache_key(memory_source_id, memory_type, memory_id)
                await cache_provider.set_json(key, {"exists": True}, ttl=settings.redis_ttl_default)
        except Exception as e:
            logger.warning("Cache write failed for citation exists check: %s", e)

    @enforce_types
    async def create(
        self,
        memory_source_id: str,
        memory_type: str,
        memory_id: str,
        citation_type: str,
        external_thread_id: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> Optional[PydanticMemoryCitation]:
        """Create a citation record using INSERT ON CONFLICT DO NOTHING.

        Returns the created record, or None if it already existed.
        Cache write for the citation record handled by ORM via create_or_ignore_with_redis.
        Also caches the exists check for the (source, type, id) triple.
        """
        citation_id = PydanticMemoryCitation._generate_id()
        now = datetime.now(timezone.utc)
        values = dict(
            id=citation_id,
            memory_source_id=memory_source_id,
            memory_type=memory_type,
            memory_id=memory_id,
            citation_type=citation_type,
            external_thread_id=external_thread_id,
            occurred_at=occurred_at,
            created_at=now,
            updated_at=now,
            is_deleted=False,
        )

        async with self.session_maker() as session:
            inserted = await MemoryCitationModel.create_or_ignore_with_redis(
                db_session=session,
                use_cache=use_cache,
                **values,
            )

        # Cache the exists result (whether just inserted or already existed)
        if use_cache:
            await self._cache_exists(memory_source_id, memory_type, memory_id)

        if inserted:
            logger.info(
                "Created citation %s: %s/%s -> source %s",
                citation_id,
                memory_type,
                memory_id,
                memory_source_id,
            )
            pydantic_values = {k: v for k, v in values.items() if k != "is_deleted"}
            return PydanticMemoryCitation(**pydantic_values)
        return None

    @enforce_types
    async def check_exists(
        self,
        memory_source_id: str,
        memory_type: str,
        memory_id: str,
        use_cache: bool = True,
    ) -> bool:
        """Check if a citation already exists for the given (source, type, id) triple.

        Uses cache-aside pattern since this is the hot-path check for citation-level dedup.
        """
        # Try cache first
        if use_cache:
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    key = self._exists_cache_key(memory_source_id, memory_type, memory_id)
                    cached_data = await cache_provider.get_json(key)
                    if cached_data:
                        logger.debug(
                            "Cache HIT for citation exists: %s/%s/%s", memory_source_id, memory_type, memory_id
                        )
                        return True
            except Exception as e:
                logger.warning("Cache read failed for citation exists check: %s", e)

        # Fall back to DB
        async with self.session_maker() as session:
            result = await session.execute(
                select(MemoryCitationModel.id).where(
                    MemoryCitationModel.memory_source_id == memory_source_id,
                    MemoryCitationModel.memory_type == memory_type,
                    MemoryCitationModel.memory_id == memory_id,
                    ~MemoryCitationModel.is_deleted,
                )
            )
            exists = result.scalar_one_or_none() is not None

        # Populate cache on positive result
        if exists and use_cache:
            await self._cache_exists(memory_source_id, memory_type, memory_id)

        return exists

    @enforce_types
    async def get_max_occurred_at(
        self,
        memory_type: str,
        memory_id: str,
    ) -> Optional[datetime]:
        """Get the most recent occurred_at for a given memory record across all citations.

        Used by the temporal guard to prevent backdated overwrites.
        Not cached — called infrequently and the max changes as new citations arrive.
        """
        async with self.session_maker() as session:
            result = await session.execute(
                select(func.max(MemoryCitationModel.occurred_at)).where(
                    MemoryCitationModel.memory_type == memory_type,
                    MemoryCitationModel.memory_id == memory_id,
                    ~MemoryCitationModel.is_deleted,
                )
            )
            return result.scalar_one_or_none()

    async def get_citations_for_memories(
        self,
        memory_keys: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], List[PydanticMemoryCitation]]:
        """Batch-fetch citations for a list of (memory_type, memory_id) pairs.

        Returns a dict keyed by (memory_type, memory_id) -> list of citations.
        Used by search endpoints to attach citation provenance to results.
        """
        if not memory_keys:
            return {}

        async with self.session_maker() as session:
            stmt = select(MemoryCitationModel).where(
                tuple_(MemoryCitationModel.memory_type, MemoryCitationModel.memory_id).in_(memory_keys),
                ~MemoryCitationModel.is_deleted,
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        grouped: Dict[Tuple[str, str], List[PydanticMemoryCitation]] = defaultdict(list)
        for row in rows:
            citation = PydanticMemoryCitation(
                id=row.id,
                memory_type=row.memory_type,
                memory_id=row.memory_id,
                memory_source_id=row.memory_source_id,
                external_thread_id=row.external_thread_id,
                occurred_at=row.occurred_at,
                citation_type=row.citation_type,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            grouped[(row.memory_type, row.memory_id)].append(citation)

        return dict(grouped)
