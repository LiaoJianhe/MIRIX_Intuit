"""Manager for MemorySource CRUD operations."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import select

from mirix.log import get_logger
from mirix.orm.memory_source import MemorySource as MemorySourceModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.memory_source import MemorySource as PydanticMemorySource
from mirix.schemas.memory_source import PaginatedResponse
from mirix.utils import enforce_types

logger = get_logger(__name__)


def parse_occurred_at(value: Union[str, datetime, None]) -> Optional[datetime]:
    """Parse an occurred_at value that may be an ISO 8601 string, datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Could not parse occurred_at value: %s", value)
            return None
    return None


class MemorySourceManager:
    """Manager for memory source persistence with INSERT ON CONFLICT DO NOTHING semantics."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    async def create(
        self,
        memory_source_id: str,
        actor: PydanticClient,
        user_id: str,
        organization_id: str,
        source_type: str = "conversation",
        external_id: Optional[str] = None,
        external_thread_id: Optional[str] = None,
        source_system: Optional[str] = None,
        source_metadata: Optional[dict] = None,
        occurred_at: Union[str, datetime, None] = None,
        summary: Optional[str] = None,
        summary_source: Optional[str] = None,
        batch_hash: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Optional[PydanticMemorySource]:
        """Create a memory source record using INSERT ON CONFLICT DO NOTHING.

        Enforces scope injection: filter_tags["scope"] is always set to
        actor.write_scope, matching the pattern in raw_memory_manager.
        Client-provided filter_tags are preserved but scope cannot be overridden.

        Returns the record (whether just inserted or pre-existing).
        Cache write handled by the ORM layer via create_or_ignore_with_redis.
        """
        # Enforce scope from actor's write_scope (same pattern as raw_memory_manager)
        if actor.write_scope is None:
            raise ValueError("Client has no write_scope - cannot create memory sources")
        if filter_tags is None:
            filter_tags = {}
        filter_tags["scope"] = actor.write_scope

        occurred_at = parse_occurred_at(occurred_at)

        # Relational provider delegation (create — relies on uq_memory_sources_ext_id /
        # uq_memory_sources_batch unique constraints for dedup).
        # Conflict → look up the pre-existing row (caller wants the existing record back).
        # Transient errors are retried; permanent errors propagate.
        from mirix.database.provider_write_retry import (
            is_conflict,
            is_transient,
            retry_transient,
        )
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            now = datetime.now(timezone.utc)
            data_dict = {
                "id": memory_source_id,
                "client_id": actor.id,
                "user_id": user_id,
                "organization_id": organization_id,
                "source_type": source_type,
                "external_id": external_id,
                "external_thread_id": external_thread_id,
                "source_system": source_system,
                "source_metadata": source_metadata,
                "occurred_at": occurred_at.isoformat() if occurred_at else None,
                "summary": summary,
                "summary_source": summary_source,
                "batch_hash": batch_hash,
                "filter_tags": filter_tags,
                "processing_complete": False,
                "_created_by_id": actor.id,
                "_last_updated_by_id": actor.id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "is_deleted": False,
            }
            try:
                result = await retry_transient(
                    lambda: provider.create("memory_sources", data_dict),
                    op=f"memory_sources.create({memory_source_id})",
                )
                return PydanticMemorySource(**result)
            except Exception as e:
                if not is_conflict(e):
                    if is_transient(e):
                        logger.error(
                            "memory_sources create failed after retries for %s: %s",
                            memory_source_id,
                            e,
                        )
                    raise
                # Conflict — look up the pre-existing row by the dedup key the caller
                # provided. Without a dedup key we can't recover, so re-raise.
                #
                # VEPAGE-1107: use named queries that explicitly project all
                # columns + FK columns so PydanticMemorySource construction
                # works. The generic ``provider.list`` path here was also
                # failing because ``client_id`` and ``user_id`` filter
                # property resolution didn't include the right relationship
                # logical names for memory_sources.
                logger.debug(
                    "memory_sources conflict for %s — looking up existing row",
                    memory_source_id,
                )
                if external_id is not None:
                    records = await provider.find_using_named_query(
                        "memory_sources",
                        "memory_source_manager.find_by_external_id",
                        params={
                            # NQ binds :clientId against the client_id FK column
                            # (matches uq_memory_sources_ext_id). The actor IS the
                            # client when MIRIX runs under ECMS, so actor.id is the
                            # client id; we use the parameter name "clientId" so the
                            # provider's APP=-prefixing of :createdById doesn't fire.
                            "clientId": actor.id,
                            "userId": user_id,
                            "externalId": external_id,
                        },
                        page_size=1,
                    )
                elif batch_hash is not None:
                    records = await provider.find_using_named_query(
                        "memory_sources",
                        "memory_source_manager.find_by_batch_hash",
                        params={
                            # See note in find_by_external_id above re: clientId vs createdById.
                            "clientId": actor.id,
                            "userId": user_id,
                            "batchHash": batch_hash,
                        },
                        page_size=1,
                    )
                else:
                    raise
                if records:
                    return PydanticMemorySource(**records[0])
                raise

        async with self.session_maker() as session:
            now = datetime.now(timezone.utc)

            await MemorySourceModel.create_or_ignore_with_redis(
                db_session=session,
                use_cache=use_cache,
                id=memory_source_id,
                client_id=actor.id,
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
                filter_tags=filter_tags,
                processing_complete=False,
                _created_by_id=actor.id,
                _last_updated_by_id=actor.id,
                created_at=now,
                updated_at=now,
                is_deleted=False,
            )

        # Return the record (from cache or DB via get_by_id)
        return await self.get_by_id(memory_source_id, use_cache=use_cache)

    @enforce_types
    async def get_by_id(
        self, memory_source_id: str, use_cache: bool = True
    ) -> Optional[PydanticMemorySource]:
        """Fetch a memory source by ID with cache-aside pattern."""
        # Relational provider delegation (read by ID) — provider is source of truth, skip cache
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            result = await provider.read("memory_sources", memory_source_id)
            if result is None:
                return None
            return PydanticMemorySource(**result)

        # Try cache first
        if use_cache:
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = (
                        f"{cache_provider.MEMORY_SOURCE_PREFIX}{memory_source_id}"
                    )
                    cached_data = await cache_provider.get_json(cache_key)
                    if cached_data:
                        logger.debug("Cache HIT for memory source %s", memory_source_id)
                        # Strip Redis-internal fields (_ts suffixes) that pydantic rejects
                        known_fields = PydanticMemorySource.model_fields
                        clean = {
                            k: v for k, v in cached_data.items() if k in known_fields
                        }
                        return PydanticMemorySource(**clean)
            except Exception as e:
                logger.warning(
                    "Cache read failed for memory source %s: %s", memory_source_id, e
                )

        # Fall back to DB
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
            pydantic_source = record.to_pydantic()

        # Populate cache on miss
        if use_cache:
            try:
                from mirix.database.cache_provider import get_cache_provider
                from mirix.settings import settings

                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = (
                        f"{cache_provider.MEMORY_SOURCE_PREFIX}{memory_source_id}"
                    )
                    data = pydantic_source.model_dump(mode="json")
                    await cache_provider.set_json(
                        cache_key, data, ttl=settings.redis_ttl_default
                    )
                    logger.debug(
                        "Populated cache for memory source %s", memory_source_id
                    )
            except Exception as e:
                logger.warning(
                    "Cache write failed for memory source %s: %s", memory_source_id, e
                )

        return pydantic_source

    async def get_sources_by_thread_id(
        self,
        external_thread_id: str,
        scopes: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[PydanticMemorySource]:
        """Fetch all sources in a thread, ordered by occurred_at ascending.

        Access control uses scope-based filtering via filter_tags->>'scope',
        matching the pattern used by memory tables. If user_id is provided,
        results are further filtered to that user.

        Returns a PaginatedResponse with next_cursor and has_more.
        """
        # Relational provider delegation — fetch matching rows via provider.list, then
        # sort/paginate client-side (provider.list doesn't expose ordering or cursors).
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # VEPAGE-1107: NQ encodes the optional user_id filter and the
            # scopes CSV inline so the response carries all the columns
            # PydanticMemorySource needs without depending on the generic
            # filter-query path's relationship resolution.
            scopes_csv = ",".join(scopes) if scopes else None
            records = await provider.find_using_named_query(
                "memory_sources",
                "memory_source_manager.list_sources_by_thread_id",
                params={
                    "externalThreadId": external_thread_id,
                    "userId": user_id,
                    "scopesCsv": scopes_csv,
                },
                page_size=1500,
            )
            # Sort ascending by (occurred_at, created_at)
            records.sort(
                key=lambda r: (r.get("occurred_at") or "", r.get("created_at") or "")
            )
            # Apply cursor (skip until we pass cursor row by id)
            if cursor:
                idx = next(
                    (i for i, r in enumerate(records) if r.get("id") == cursor), None
                )
                if idx is not None:
                    records = records[idx + 1 :]
            has_more = len(records) > limit
            records = records[:limit]
            items = [PydanticMemorySource(**r) for r in records]
            return PaginatedResponse(
                items=items,
                next_cursor=items[-1].id if has_more and items else None,
                has_more=has_more,
            )

        from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy

        async with self.session_maker() as session:
            query = (
                select(MemorySourceModel)
                .where(
                    MemorySourceModel.external_thread_id == external_thread_id,
                    ~MemorySourceModel.is_deleted,
                )
                .order_by(
                    MemorySourceModel.occurred_at.asc(),
                    MemorySourceModel.created_at.asc(),
                )
            )

            # Scope-based access control (same pattern as memory tables)
            query = apply_filter_tags_sqlalchemy(
                query, MemorySourceModel, None, scopes=scopes
            )

            if user_id:
                query = query.where(MemorySourceModel.user_id == user_id)

            if cursor:
                cursor_result = await session.execute(
                    select(MemorySourceModel).where(MemorySourceModel.id == cursor)
                )
                cursor_obj = cursor_result.scalar_one_or_none()
                if cursor_obj:
                    # Use created_at for stable ordering when occurred_at may be null
                    ref = cursor_obj.occurred_at or cursor_obj.created_at
                    query = query.where(
                        (MemorySourceModel.occurred_at > ref)
                        | (
                            (MemorySourceModel.occurred_at == ref)
                            & (MemorySourceModel.created_at > cursor_obj.created_at)
                        )
                    )

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

    async def list_sources(
        self,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        scope: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[PydanticMemorySource]:
        """List memory sources ordered by occurred_at descending.

        No scope-based access control — intended for admin use.
        Supports filtering by organization_id (required for the IPSR path),
        user_id, client_id, scope, and time range.

        organization_id is optional only for the legacy ORM fallback path
        (which has no IPSR-style org isolation requirement). The IPSR path
        requires it because the underlying named query uses it as the
        org-scoped access predicate.
        """
        # Relational provider delegation
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # VEPAGE-1144: route through list_sources_admin NQ. The NQ
            # filters and orders server-side (occurredAt DESC NULLS LAST,
            # ipsrcreatedon DESC); cursor pagination is still applied
            # client-side by walking the returned page (matches the
            # get_sources_by_thread_id pattern). The 1500-row IPSR cap
            # applies to a single page fetch — datasets larger than that
            # need to switch to pageNum-based iteration, which is a
            # follow-up.
            if not organization_id:
                raise ValueError(
                    "list_sources requires organization_id when an IPSR "
                    "provider is registered (the NQ uses it as the "
                    "org-scoped access predicate)"
                )
            records = await provider.find_using_named_query(
                "memory_sources",
                "memory_source_manager.list_sources_admin",
                params={
                    "organizationId": organization_id,
                    "userId": user_id,
                    "clientId": client_id,
                    "scope": scope,
                    "since": since.isoformat() if since else None,
                    "until": until.isoformat() if until else None,
                },
                page_size=1500,
            )
            # Apply cursor: skip until we pass the cursor row's id.
            # Server-side ordering is already correct, so no client-side
            # sort is needed here.
            if cursor:
                idx = next(
                    (i for i, r in enumerate(records) if r.get("id") == cursor), None
                )
                if idx is not None:
                    records = records[idx + 1 :]
            has_more = len(records) > limit
            records = records[:limit]
            items = [PydanticMemorySource(**r) for r in records]
            return PaginatedResponse(
                items=items,
                next_cursor=items[-1].id if has_more and items else None,
                has_more=has_more,
            )

        async with self.session_maker() as session:
            query = (
                select(MemorySourceModel)
                .where(~MemorySourceModel.is_deleted)
                .order_by(
                    MemorySourceModel.occurred_at.desc().nulls_last(),
                    MemorySourceModel.created_at.desc(),
                )
            )

            if organization_id:
                query = query.where(
                    MemorySourceModel.organization_id == organization_id
                )
            if user_id:
                query = query.where(MemorySourceModel.user_id == user_id)
            if client_id:
                query = query.where(MemorySourceModel.client_id == client_id)
            if scope:
                query = query.where(
                    MemorySourceModel.filter_tags["scope"].astext == scope
                )
            if since:
                query = query.where(MemorySourceModel.occurred_at >= since)
            if until:
                query = query.where(MemorySourceModel.occurred_at <= until)

            if cursor:
                cursor_result = await session.execute(
                    select(MemorySourceModel).where(MemorySourceModel.id == cursor)
                )
                cursor_obj = cursor_result.scalar_one_or_none()
                if cursor_obj:
                    ref = cursor_obj.occurred_at or cursor_obj.created_at
                    query = query.where(
                        (MemorySourceModel.occurred_at < ref)
                        | (
                            (MemorySourceModel.occurred_at == ref)
                            & (MemorySourceModel.created_at < cursor_obj.created_at)
                        )
                    )

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

    @enforce_types
    async def mark_processing_complete(self, memory_source_id: str) -> None:
        """Set processing_complete = True after all agents finish successfully.

        Uses read-then-update via ORM for consistent cache handling.
        Safe from lost updates because:
        - processing_complete is a one-way flag (False → True), so concurrent
          writers always converge to the same value
        - SQLAlchemy only UPDATEs dirty columns, so this won't overwrite
          concurrent changes to other fields (e.g. summary)
        """
        # Relational provider delegation (partial update)
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            await provider.update(
                "memory_sources", memory_source_id, {"processing_complete": True}
            )
            logger.info(
                "Marked memory source %s as processing complete", memory_source_id
            )
            return

        async with self.session_maker() as session:
            record = await MemorySourceModel.read(
                db_session=session, identifier=memory_source_id
            )
            record.processing_complete = True
            await record.update_with_redis(session)
            logger.info(
                "Marked memory source %s as processing complete", memory_source_id
            )

    @enforce_types
    async def update_summary(
        self, memory_source_id: str, summary: str, summary_source: str
    ) -> None:
        """Write a summary to an existing memory source.

        Used after processing completes to store a generated summary.
        Safe from lost updates: only touches summary/summary_source columns.
        """
        # Relational provider delegation (partial update)
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            await provider.update(
                "memory_sources",
                memory_source_id,
                {"summary": summary, "summary_source": summary_source},
            )
            logger.info(
                "Updated summary for memory source %s (source=%s)",
                memory_source_id,
                summary_source,
            )
            return

        async with self.session_maker() as session:
            record = await MemorySourceModel.read(
                db_session=session, identifier=memory_source_id
            )
            record.summary = summary
            record.summary_source = summary_source
            await record.update_with_redis(session)
            logger.info(
                "Updated summary for memory source %s (source=%s)",
                memory_source_id,
                summary_source,
            )
