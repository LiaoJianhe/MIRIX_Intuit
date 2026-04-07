from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

from mirix.schemas.mirix_base import MirixBase

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Cursor-based paginated response envelope."""

    items: List[T]
    next_cursor: Optional[str] = Field(None, description="Cursor for the next page (pass as ?cursor=). Null if no more results.")
    has_more: bool = Field(..., description="Whether there are more results beyond this page")


class MemorySourceBase(MirixBase):
    """Base schema for memory sources."""

    __id_prefix__ = "src"


class MemorySource(MemorySourceBase):
    """Full representation of a memory source record."""

    id: str = MemorySourceBase.generate_id_field()
    client_id: str = Field(..., description="ID of the client application")
    user_id: str = Field(..., description="ID of the user")
    organization_id: str = Field(..., description="ID of the organization")
    external_id: Optional[str] = Field(None, description="Client-provided stable dedup key")
    external_thread_id: Optional[str] = Field(None, description="Groups multiple saves into a thread")
    source_type: str = Field("conversation", description="Freeform source type label")
    source_system: Optional[str] = Field(None, description="Originating system label")
    source_metadata: Optional[Dict[str, Any]] = Field(None, description="Client-provided lineage context")
    occurred_at: Optional[datetime] = Field(None, description="When the source event happened")
    summary: Optional[str] = Field(None, description="Client-provided or generated summary")
    summary_source: Optional[str] = Field(None, description="How the summary was produced: client or generated")
    processing_complete: bool = Field(False, description="Whether all agents have finished processing")
    batch_hash: Optional[str] = Field(None, description="SHA-256 fallback dedup hash")
    filter_tags: Optional[Dict[str, Any]] = Field(None, description="Custom filter tags for filtering and categorization (includes scope for access control)")
    created_at: Optional[datetime] = Field(None, description="When the record was created")
    updated_at: Optional[datetime] = Field(None, description="When the record was last updated")


class MemorySourceUpdate(MemorySourceBase):
    """Schema for updating a memory source. Only id is required."""

    id: str = Field(..., description="ID of the memory source to update")
    summary: Optional[str] = Field(None, description="Summary text")
    summary_source: Optional[str] = Field(None, description="How the summary was produced")
    processing_complete: Optional[bool] = Field(None, description="Whether processing is complete")
