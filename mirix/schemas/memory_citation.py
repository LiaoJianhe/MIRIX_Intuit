from datetime import datetime
from typing import Optional

from pydantic import Field

from mirix.schemas.mirix_base import MirixBase


class MemoryCitationBase(MirixBase):
    """Base schema for memory citations."""

    __id_prefix__ = "cit"


class MemoryCitation(MemoryCitationBase):
    """Full representation of a memory citation record."""

    id: str = MemoryCitationBase.generate_id_field()
    memory_type: str = Field(
        ..., description="Memory type: episodic, semantic, procedural, resource, knowledge_vault, core"
    )
    memory_id: str = Field(..., description="Polymorphic reference to the specific memory record")
    memory_source_id: str = Field(..., description="ID of the parent memory source")
    external_thread_id: Optional[str] = Field(None, description="Denormalized thread ID")
    occurred_at: Optional[datetime] = Field(None, description="Denormalized from memory source")
    citation_type: str = Field(..., description="How citation links source to memory: created or updated")
    created_at: Optional[datetime] = Field(None, description="When the record was created")
    updated_at: Optional[datetime] = Field(None, description="When the record was last updated")
