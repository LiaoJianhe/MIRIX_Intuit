"""
Save-flow helper: labeled-bucket candidate fetcher for owning sub-agents.

When a sub-agent is about to decide create-vs-update for a memory type, it
needs to see (a) the most relevant existing memories ranked by the Search
provider and (b) any rows that were just written and may not be indexed yet
in the Search provider. Merging the two and presenting a single ranked list
hides the distinction the LLM needs to make a good dedup judgment, so this
helper returns them as labeled buckets instead.

Used from ``Agent.build_system_prompt_with_memories`` for each
``is_owning_agent`` branch. Read paths (manager ``list_*`` / ``search_*`` /
``count_*``) do not call this helper; they delegate directly to
``search_provider.search()``.

Fail-closed: if either the Search call or the Relational call raises, the
exception propagates. We deliberately do not return partial results from a
single backend.

The recent-window duration is configured once at ECMS startup via
``set_hybrid_window_seconds`` (called from
``common.ips_provider_setup.register_ips_providers``).
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

_hybrid_window_seconds: int = 5

DEFAULT_RECENT_CAP = 500


def set_hybrid_window_seconds(seconds: int) -> None:
    """Set the recent-window duration used by ``fetch_and_dedup_candidates``.

    Called once at ECMS startup from
    ``common.ips_provider_setup.register_ips_providers`` after reading
    ``config.ips_hybrid_read_window_seconds``. MIRIX reads this value via
    ``get_hybrid_window_seconds()`` without importing ECMS ``SvcSettings``.
    """
    global _hybrid_window_seconds
    _hybrid_window_seconds = seconds


def get_hybrid_window_seconds() -> int:
    """Get the configured recent-window duration in seconds."""
    return _hybrid_window_seconds


@dataclass(frozen=True)
class DedupCandidates:
    """Two labeled candidate buckets for the dedup-deciding sub-agent.

    Attributes:
        relevant: Ranked results from the Search provider, in Search-provider
            order, truncated to ``limit``. Used by the LLM to find the closest
            existing memory for the incoming content.
        recent: Rows written within the recent window from the Relational
            provider, capped at ``recent_cap``. No ranking is applied; these
            are the candidates the Search provider may not have indexed yet.
            Used by the LLM to detect near-duplicate writes that just landed.
    """

    relevant: List[Dict[str, Any]] = field(default_factory=list)
    recent: List[Dict[str, Any]] = field(default_factory=list)


async def fetch_and_dedup_candidates(
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
    recent_cap: int = DEFAULT_RECENT_CAP,
    start_date: Any = None,
    end_date: Any = None,
    similarity_threshold: Optional[float] = None,
    sort: Optional[str] = None,
    cursor: Optional[str] = None,
    time_range: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> DedupCandidates:
    """Fetch the relevant + recent buckets for a save-flow dedup decision.

    Both calls run in parallel via ``asyncio.gather`` so the save path takes
    one round-trip wall-clock rather than two.

    Args:
        table: Logical table name (e.g. ``"semantic_memory"``, ``"block"``).
        search_provider: Registered Search provider instance.
        relational_provider: Registered Relational DB provider instance.
        query_text: Optional text query for the Search call.
        query_embedding: Optional precomputed embedding for the Search call.
        search_method: ``"embedding"`` / ``"bm25"`` / ``"string_match"`` etc.
        search_field: Field name to search against on the Search side.
        user_id: User scope for both calls. None for org-wide queries.
        organization_id: Organization scope.
        filter_tags: Tag filter dict applied to both backends.
        scopes: Optional scope list for filter_tags.scope.
        limit: Max results in the ``relevant`` bucket. Defaults to 50.
        recent_cap: Max rows in the ``recent`` bucket. Defaults to 500.
            The cap exists to protect prompt size against a pathological
            burst of writes in the recent window.
        start_date, end_date, similarity_threshold, sort, cursor, time_range,
            **kwargs: Passed through to ``search_provider.search``.

    Returns:
        ``DedupCandidates(relevant, recent)``. Caller decides ordering and
        formatting when rendering these into the prompt.

    Raises:
        Any exception raised by either provider call. Fail-closed.
    """
    effective_limit = limit or 50

    window_seconds = get_hybrid_window_seconds()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

    search_coro = search_provider.search(
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

    recent_coro = relational_provider.list(
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
        limit=recent_cap,
    )

    (search_results, _next_cursor), recent_records = await asyncio.gather(
        search_coro, recent_coro
    )

    return DedupCandidates(
        relevant=list(search_results),
        recent=list(recent_records),
    )
