from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.memory_source import MemorySource as PydanticMemorySource
from mirix.settings import settings


class MemorySource(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    ORM model for memory sources — tracks the provenance of ingested conversations.

    Each call to POST /memory/add creates one MemorySource with linked SourceMessages.
    Citations link individual memory writes back to their source.
    """

    __tablename__ = "memory_sources"
    __pydantic_model__ = PydanticMemorySource

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Pre-generated src-{uuid4} by API handler",
    )

    external_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Client-provided stable dedup key",
    )

    external_thread_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Groups multiple saves into a thread",
    )

    client_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=False,
        doc="ID of the client application",
    )

    source_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="conversation",
        doc="Freeform source type label",
    )

    source_system: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Originating system label",
    )

    source_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        doc="Client-provided lineage context bag",
    )

    occurred_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the source event happened",
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Client-provided or generated summary",
    )

    summary_source: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="How the summary was produced: client or generated",
    )

    processing_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("FALSE"),
        doc="Whether all agents have finished processing this source (legacy; superseded by status)",
    )

    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
        doc="Processing state: pending | processing | completed | failed",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Error detail when status='failed' (provider-mapped exception class + message)",
    )

    delivery_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Number of times this source has been delivered to a worker",
    )

    batch_hash: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="SHA-256 fallback dedup hash",
    )

    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        doc="Custom filter tags for filtering and categorization (includes scope for access control)",
    )

    __table_args__ = tuple(
        filter(
            None,
            [
                # status must be one of the four allowed states
                CheckConstraint(
                    "status IN ('pending', 'processing', 'completed', 'failed')",
                    name="ck_memory_sources_status",
                ),
                # Partial unique index: external_id dedup (PostgreSQL only)
                (
                    Index(
                        "uq_memory_sources_ext_id",
                        "client_id",
                        "user_id",
                        "external_id",
                        unique=True,
                        postgresql_where=text("external_id IS NOT NULL"),
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Partial unique index: batch_hash dedup (PostgreSQL only)
                (
                    Index(
                        "uq_memory_sources_batch",
                        "client_id",
                        "user_id",
                        "batch_hash",
                        unique=True,
                        postgresql_where=text("batch_hash IS NOT NULL"),
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Thread query index
                (
                    Index(
                        "ix_memory_sources_thread",
                        "client_id",
                        "user_id",
                        "external_thread_id",
                        "occurred_at",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # GIN index on filter_tags for flexible tag queries
                (
                    Index(
                        "ix_memory_sources_filter_tags_gin",
                        text("(filter_tags::jsonb)"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Scope-based access control index (matches memory table pattern)
                (
                    Index(
                        "ix_memory_sources_org_filter_scope",
                        "organization_id",
                        text("((filter_tags->>'scope')::text)"),
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )
