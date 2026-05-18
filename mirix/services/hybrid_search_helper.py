"""
Hybrid search helper for engine_operation paths.

Combines IPS Search results with IPS Relational recent-window records for
the hybrid read strategy. Used only when ``call_origin == engine_operation``
(queue workers, build_system_prompt_with_memories, etc.).

For ``call_origin == client_api`` paths, managers delegate directly to
``search_provider.search()`` (Search-only) without calling this helper.

Fail-closed: if either Search or Relational step raises, the exception
propagates — no partial results from a single backend.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from mirix.database.call_context import get_hybrid_window_seconds
from mirix.log import get_logger

logger = get_logger(__name__)

HYBRID_COUNT_RECENT_LIMIT = 500


async def hybrid_search(
    table: str,
    search_provider: Any,
    relational_provider: Any,
    *,
    query_text: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    search_method: str = "embedding",
    search_field: Optional[str] = None,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    filter_tags: Optional[Dict[str, Any]] = None,
    scopes: Optional[List[str]] = None,
    limit: Optional[int] = None,
    start_date: Any = None,
    end_date: Any = None,
    similarity_threshold: Optional[float] = None,
    sort: Optional[str] = None,
    cursor: Optional[str] = None,
    time_range: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Hybrid search: IPS Search results + IPS Relational recent-window merge.

    1. Query IPS Search for the main result set.
    2. Query IPS Relational for records updated/created within the hybrid
       window (to capture recently written data not yet indexed).
    3. Merge and deduplicate using timestamp-based precedence.
    4. Sort by timestamp descending, apply limit.

    Returns a ``(results, next_cursor)`` tuple. ``next_cursor`` is the cursor
    returned by ``search_provider.search()`` and lets callers paginate the
    underlying IPS Search query. The recent-window relational records are not
    paginated separately; they are intentionally bounded by the hybrid
    time-window cutoff and the per-call ``limit``.

    Both steps must succeed; if either raises, the exception propagates
    (fail-closed — no partial results).
    """
    effective_limit = limit or 50
    window_seconds = get_hybrid_window_seconds()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

    search_results, next_cursor = await search_provider.search(
        table,
        query_text=query_text,
        query_embedding=query_embedding,
        search_method=search_method,
        search_field=search_field,
        user_id=user_id,
        organization_id=organization_id,
        filter_tags=filter_tags,
        scopes=scopes,
        limit=effective_limit,
        start_date=start_date,
        end_date=end_date,
        similarity_threshold=similarity_threshold,
        sort=sort,
        cursor=cursor,
        time_range=time_range,
        **kwargs,
    )

    recent_records = await relational_provider.list(
        table,
        user_id=user_id,
        organization_id=organization_id,
        filter_tags=filter_tags,
        scopes=scopes,
        time_range={
            "updated_at__gte": cutoff.isoformat(),
            "created_at__gte": cutoff.isoformat(),
        },
        time_range_or_null_updated=True,
        limit=effective_limit,
    )

    merged = _merge_and_deduplicate(search_results, recent_records, effective_limit)
    return merged, next_cursor


async def hybrid_count(
    table: str,
    search_provider: Any,
    relational_provider: Any,
    *,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    filter_tags: Optional[Dict[str, Any]] = None,
    scopes: Optional[List[str]] = None,
) -> int:
    """
    Hybrid count: IPS Search count + IPS Relational recent-window extras.

    Returns the search count plus relational records in the hybrid window
    that are NOT yet in search. Both steps must succeed (fail-closed).
    """
    search_count = await search_provider.count(
        table,
        user_id=user_id,
        organization_id=organization_id,
        filter_tags=filter_tags,
        scopes=scopes,
    )

    window_seconds = get_hybrid_window_seconds()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

    recent_records = await relational_provider.list(
        table,
        user_id=user_id,
        organization_id=organization_id,
        filter_tags=filter_tags,
        scopes=scopes,
        time_range={
            "updated_at__gte": cutoff.isoformat(),
            "created_at__gte": cutoff.isoformat(),
        },
        time_range_or_null_updated=True,
        limit=HYBRID_COUNT_RECENT_LIMIT,
    )

    extra_count = 0
    for record in recent_records:
        record_id = record.get("id")
        if record_id:
            in_search = await search_provider.get_by_id(
                table, record_id, user_id=user_id
            )
            if not in_search:
                extra_count += 1

    return search_count + extra_count


def _record_timestamp(record: Dict) -> str:
    """Return the best timestamp for ordering/precedence comparison."""
    return record.get("updated_at") or record.get("created_at") or ""


def _merge_and_deduplicate(
    search_results: List[Dict],
    recent_records: List[Dict],
    limit: int,
) -> List[Dict]:
    """
    Merge search results with relational recent-window records.

    For records appearing in both sets (same id), keep the version with
    the more recent updated_at; if null or equal, compare created_at.
    Then sort by timestamp descending and cap at limit.
    """
    by_id: Dict[str, Dict] = {}

    for record in search_results:
        record_id = record.get("id")
        if record_id:
            by_id[record_id] = record

    for record in recent_records:
        record_id = record.get("id")
        if not record_id:
            continue
        existing = by_id.get(record_id)
        if existing is None:
            by_id[record_id] = record
        else:
            if _record_timestamp(record) >= _record_timestamp(existing):
                by_id[record_id] = record

    merged = list(by_id.values())
    merged.sort(key=_record_timestamp, reverse=True)

    return merged[:limit]
