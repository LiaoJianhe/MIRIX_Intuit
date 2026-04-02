"""Manager for SourceMessage CRUD operations."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert

from mirix.log import get_logger
from mirix.orm.source_message import SourceMessage as SourceMessageModel
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


class SourceMessageManager:
    """Manager for source message persistence with INSERT ON CONFLICT DO NOTHING semantics."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def bulk_insert(
        self,
        messages: List[Dict[str, Any]],
        memory_source_id: str,
        external_thread_id: Optional[str] = None,
    ) -> int:
        """Insert source messages in bulk using INSERT ON CONFLICT DO NOTHING.

        Each message dict should have: role, content, and optionally
        external_message_id and occurred_at.

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
                    occurred_at=msg.get("occurred_at"),
                    sequence_num=seq,
                    content_hash=content_hash,
                    created_at=now,
                    updated_at=now,
                    is_deleted=False,
                )
            )

        async with self.session_maker() as session:
            stmt = pg_insert(SourceMessageModel).values(rows).on_conflict_do_nothing()
            result = await session.execute(stmt)
            await session.commit()
            inserted = result.rowcount if result.rowcount else 0
            logger.info(
                "Bulk inserted %d/%d source messages for source %s",
                inserted,
                len(rows),
                memory_source_id,
            )
            return inserted
