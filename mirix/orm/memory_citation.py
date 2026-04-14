from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.memory_citation import MemoryCitation as PydanticMemoryCitation
from mirix.settings import settings


class MemoryCitation(SqlalchemyBase):
    """
    ORM model for citations linking memory records back to their source.

    Each time a memory agent creates or updates a memory, a citation is written
    recording which MemorySource triggered the write. This enables progressive
    disclosure: memory -> citations -> source -> messages.
    """

    __tablename__ = "memory_citations"
    __pydantic_model__ = PydanticMemoryCitation

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this citation",
    )

    memory_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Memory type: episodic, semantic, procedural, resource, knowledge_vault, core",
    )

    memory_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Polymorphic reference to the specific memory record",
    )

    memory_source_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("memory_sources.id", ondelete="CASCADE"),
        nullable=False,
        doc="ID of the parent memory source",
    )

    external_thread_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Denormalized thread ID",
    )

    occurred_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Denormalized from memory source",
    )

    citation_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="How citation links source to memory: created or updated",
    )

    __table_args__ = tuple(
        filter(
            None,
            [
                # Unique constraint: one citation per (source, memory_type, memory_id)
                (
                    Index(
                        "uq_memory_citations_src_type_id",
                        "memory_source_id",
                        "memory_type",
                        "memory_id",
                        unique=True,
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Lookup by memory record
                (
                    Index(
                        "ix_memory_citations_memory",
                        "memory_type",
                        "memory_id",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Lookup by source
                (
                    Index(
                        "ix_memory_citations_source",
                        "memory_source_id",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Lookup by thread
                (
                    Index(
                        "ix_memory_citations_thread",
                        "external_thread_id",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )
