"""Manager for MemoryCitation CRUD operations."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

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

    @enforce_types
    async def create(
        self,
        memory_source_id: str,
        memory_type: str,
        memory_id: str,
        citation_type: str,
        external_thread_id: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        message_ids: Optional[List[str]] = None,
    ) -> Optional[PydanticMemoryCitation]:
        """Create a citation record using INSERT ON CONFLICT DO NOTHING.

        Returns the created record, or None if it already existed (conflict on
        unique constraint: memory_source_id + memory_type + memory_id).
        """
        citation_id = PydanticMemoryCitation._generate_id()

        async with self.session_maker() as session:
            from datetime import timezone

            now = datetime.now(timezone.utc)
            values = dict(
                id=citation_id,
                memory_source_id=memory_source_id,
                memory_type=memory_type,
                memory_id=memory_id,
                citation_type=citation_type,
                external_thread_id=external_thread_id,
                occurred_at=occurred_at,
                message_ids=message_ids,
                created_at=now,
                updated_at=now,
                is_deleted=False,
            )

            stmt = pg_insert(MemoryCitationModel).values(**values).on_conflict_do_nothing()
            result = await session.execute(stmt)
            await session.commit()

            if result.rowcount and result.rowcount > 0:
                logger.info(
                    "Created citation %s: %s/%s -> source %s",
                    citation_id,
                    memory_type,
                    memory_id,
                    memory_source_id,
                )
                return PydanticMemoryCitation(**values)
            return None

    @enforce_types
    async def check_exists(
        self,
        memory_source_id: str,
        memory_type: str,
        memory_id: str,
    ) -> bool:
        """Check if a citation already exists for the given (source, type, id) triple."""
        async with self.session_maker() as session:
            result = await session.execute(
                select(MemoryCitationModel.id).where(
                    MemoryCitationModel.memory_source_id == memory_source_id,
                    MemoryCitationModel.memory_type == memory_type,
                    MemoryCitationModel.memory_id == memory_id,
                    ~MemoryCitationModel.is_deleted,
                )
            )
            return result.scalar_one_or_none() is not None

    @enforce_types
    async def get_max_occurred_at(
        self,
        memory_type: str,
        memory_id: str,
    ) -> Optional[datetime]:
        """Get the most recent occurred_at for a given memory record across all citations.

        Used by the temporal guard to prevent backdated overwrites.
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
