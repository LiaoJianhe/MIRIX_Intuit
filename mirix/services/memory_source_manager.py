"""Manager for MemorySource CRUD operations."""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from mirix.log import get_logger
from mirix.orm.memory_source import MemorySource as MemorySourceModel
from mirix.schemas.memory_source import MemorySource as PydanticMemorySource
from mirix.utils import enforce_types

logger = get_logger(__name__)


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
        occurred_at: Optional[datetime] = None,
        summary: Optional[str] = None,
        summary_source: Optional[str] = None,
        batch_hash: Optional[str] = None,
    ) -> Optional[PydanticMemorySource]:
        """Create a memory source record using INSERT ON CONFLICT DO NOTHING.

        Returns the created record, or None if it already existed (conflict).
        """
        async with self.session_maker() as session:
            now = datetime.now(timezone.utc)
            values = dict(
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

            stmt = pg_insert(MemorySourceModel).values(**values).on_conflict_do_nothing()
            await session.execute(stmt)
            await session.commit()

            # Fetch the record (either just inserted or pre-existing)
            return await self.get_by_id(memory_source_id)

    @enforce_types
    async def get_by_id(self, memory_source_id: str) -> Optional[PydanticMemorySource]:
        """Fetch a memory source by ID. Returns None if not found."""
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
    async def mark_processing_complete(self, memory_source_id: str) -> None:
        """Set processing_complete = True after all agents finish successfully."""
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
