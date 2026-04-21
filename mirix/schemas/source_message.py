from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import Field

from mirix.schemas.mirix_base import MirixBase


class SourceMessageBase(MirixBase):
    """Base schema for source messages."""

    __id_prefix__ = "smsg"


class SourceMessage(SourceMessageBase):
    """Full representation of a source message record."""

    id: str = SourceMessageBase.generate_id_field()
    memory_source_id: str = Field(..., description="ID of the parent memory source")
    external_thread_id: Optional[str] = Field(None, description="Denormalized thread ID from parent source")
    external_message_id: Optional[str] = Field(None, description="Client-provided stable message ID")
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: Dict[str, Any] = Field(..., description="Message content structure")
    occurred_at: Optional[datetime] = Field(None, description="Per-message timestamp")
    sequence_num: int = Field(..., description="Ordering within the source")
    content_hash: str = Field(..., description="SHA-256 of (role, content) for dedup")
    message_metadata: Optional[Dict[str, Any]] = Field(None, description="Client-provided per-message property bag")
    created_at: Optional[datetime] = Field(None, description="When the record was created")
    updated_at: Optional[datetime] = Field(None, description="When the record was last updated")
