"""Manager for SourceMessage CRUD operations."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import select

from mirix.log import get_logger
from mirix.services.memory_source_manager import parse_occurred_at
from mirix.orm.source_message import SourceMessage as SourceMessageModel
from mirix.schemas.memory_source import PaginatedResponse
from mirix.schemas.source_message import SourceMessage as PydanticSourceMessage
from mirix.utils import enforce_types

logger = get_logger(__name__)


def compute_content_hash(role: str, content: Any) -> str:
    """Compute SHA-256 hash of (role, content) for dedup.

    Uses length-prefixed encoding to prevent boundary collisions.
    """
    h = hashlib.sha256()
    role_bytes = role.encode("utf-8")
    h.update(len(role_bytes).to_bytes(4, "big"))
    h.update(role_bytes)

    content_bytes = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h.update(len(content_bytes).to_bytes(4, "big"))
    h.update(content_bytes)

    return h.hexdigest()


def compute_batch_hash(
    external_thread_id: Optional[str],
    occurred_at: Optional[str],
    messages: List[Dict[str, Any]],
) -> str:
    """Compute deterministic SHA-256 hash of an entire message batch for dedup.

    Uses length-prefixed encoding to prevent boundary collisions.
    Messages should already be normalized via normalize_message().
    """
    h = hashlib.sha256()
    for field in [external_thread_id or "", occurred_at or ""]:
        encoded = field.encode("utf-8")
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
    for msg in messages:
        for part in [msg["role"], json.dumps(msg["content"], sort_keys=True, ensure_ascii=False)]:
            encoded = part.encode("utf-8")
            h.update(len(encoded).to_bytes(4, "big"))
            h.update(encoded)
    return h.hexdigest()


def derive_external_id_from_message_ids(message_ids: List[str]) -> str:
    """Auto-derive an external_id from sorted external_message_id values.

    Returns a deterministic "auto-{sha256}" string when all messages in a batch
    have external_message_ids, giving clients implicit dedup even without
    providing an explicit external_id.
    """
    h = hashlib.sha256()
    for mid in sorted(message_ids):
        encoded = mid.encode("utf-8")
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
    return f"auto-{h.hexdigest()}"


def _get(msg, key, default=None):
    """Read a field from a dict or an object attribute."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def normalize_message(msg) -> Dict[str, Any]:
    """Convert a Message, MessageCreate, or plain dict into a dict for source message storage.

    Accepts three input shapes:
      - Pydantic MessageCreate objects (from agent processing path)
      - Plain dicts (from source_messages deserialization in the worker)
      - Message ORM objects

    Handles role extraction (str or enum), content normalization (str -> dict),
    and optional per-message fields (external_message_id, message_occurred_at).
    """
    role = _get(msg, "role") or "user"
    if hasattr(role, "value"):
        role = role.value
    role = str(role)

    content = _get(msg, "content")
    if content is None:
        content = _get(msg, "text")
    if content is None:
        content = str(msg)

    # Normalize content to dict
    if isinstance(content, str):
        content = {"text": content}
    elif not isinstance(content, dict):
        content = {"text": str(content)}

    result = {"role": role, "content": content}

    # Carry per-message fields if present
    ext_msg_id = _get(msg, "external_message_id")
    if ext_msg_id:
        result["external_message_id"] = ext_msg_id

    occurred = _get(msg, "message_occurred_at") or _get(msg, "occurred_at")
    if occurred:
        result["occurred_at"] = occurred

    metadata = _get(msg, "metadata")
    if metadata:
        result["metadata"] = metadata

    return result


class SourceMessageManager:
    """Manager for source message persistence with INSERT ON CONFLICT DO NOTHING semantics."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    async def bulk_insert(
        self,
        messages: List[Dict[str, Any]],
        memory_source_id: str,
        external_thread_id: Optional[str] = None,
        fallback_occurred_at: Optional[str] = None,
    ) -> int:
        """Insert source messages in bulk using ORM bulk_create_or_ignore.

        Each message dict should have: role, content, and optionally
        external_message_id, occurred_at, and metadata.

        Args:
            fallback_occurred_at: Top-level occurred_at from the request. Used when
                a message doesn't have its own per-message occurred_at.

        Returns the number of rows actually inserted (excludes conflicts).
        """
        if not messages:
            return 0

        now = datetime.now(timezone.utc)
        rows = []
        for seq, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            content_hash = compute_content_hash(role, content)

            rows.append(
                dict(
                    id=PydanticSourceMessage._generate_id(),
                    memory_source_id=memory_source_id,
                    external_thread_id=external_thread_id,
                    external_message_id=msg.get("external_message_id"),
                    role=role,
                    content=content if isinstance(content, dict) else {"text": content},
                    occurred_at=parse_occurred_at(msg.get("occurred_at") or fallback_occurred_at),
                    sequence_num=seq,
                    content_hash=content_hash,
                    message_metadata=msg.get("metadata"),
                    created_at=now,
                    updated_at=now,
                    is_deleted=False,
                )
            )

        async with self.session_maker() as session:
            inserted = await SourceMessageModel.bulk_create_or_ignore(
                db_session=session,
                rows=rows,
            )
            logger.info(
                "Bulk inserted %d/%d source messages for source %s",
                inserted,
                len(rows),
                memory_source_id,
            )
            return inserted

    @enforce_types
    async def bulk_insert_from_messages(
        self,
        input_messages: list,
        memory_source_id: str,
        external_thread_id: Optional[str] = None,
    ) -> int:
        """Convert raw Message/MessageCreate objects and bulk insert as source messages.

        Handles message normalization (role extraction, content conversion) internally.
        """
        msg_dicts = [normalize_message(msg) for msg in input_messages]
        return await self.bulk_insert(
            messages=msg_dicts,
            memory_source_id=memory_source_id,
            external_thread_id=external_thread_id,
        )

    async def get_messages_by_source_id(
        self,
        memory_source_id: str,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[PydanticSourceMessage]:
        """Fetch messages for a source, ordered by sequence_num ascending.

        Returns a PaginatedResponse with next_cursor and has_more.
        """
        async with self.session_maker() as session:
            query = (
                select(SourceMessageModel)
                .where(
                    SourceMessageModel.memory_source_id == memory_source_id,
                    ~SourceMessageModel.is_deleted,
                )
                .order_by(SourceMessageModel.sequence_num.asc())
            )

            if cursor:
                cursor_result = await session.execute(
                    select(SourceMessageModel).where(SourceMessageModel.id == cursor)
                )
                cursor_obj = cursor_result.scalar_one_or_none()
                if cursor_obj:
                    query = query.where(SourceMessageModel.sequence_num > cursor_obj.sequence_num)

            # Fetch limit+1 to determine has_more
            query = query.limit(limit + 1)
            result = await session.execute(query)
            records = result.scalars().all()

            has_more = len(records) > limit
            records = records[:limit]
            items = [rec.to_pydantic() for rec in records]

            return PaginatedResponse(
                items=items,
                next_cursor=items[-1].id if has_more and items else None,
                has_more=has_more,
            )

