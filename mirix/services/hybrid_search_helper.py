"""
Save-flow helper: recent-window fetcher for owning sub-agents.

When a sub-agent is about to decide create-vs-update for a memory type, the
ranked Search results it gets from ``manager.list_*`` may omit rows that were
written so recently the Search provider has not indexed them yet. This helper
fetches those just-written rows from the Relational provider so the caller can
union them into the ranked list and avoid creating a duplicate.

Used from ``Agent._fetch_recent_indexing_lag_window`` for each
``is_owning_agent`` branch in ``build_system_prompt_with_memories``. Read paths
(manager ``list_*`` / ``search_*`` / ``count_*``) do not call this helper; they
delegate directly to ``search_provider.search()``. The ranked ("relevant")
bucket is obtained by the caller via the manager ``list_*`` call (which
preserves ``@update_timezone`` and the manager's post-processing), so this
helper deliberately does not issue a Search call.

Fail-closed: if the Relational call raises, the exception propagates.

The recent-window duration is configured once at ECMS startup via
``set_hybrid_window_seconds`` (called from
``common.ips_provider_setup.register_ips_providers``).
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

_hybrid_window_seconds: int = 5

DEFAULT_RECENT_CAP = 500


def set_hybrid_window_seconds(seconds: int) -> None:
    """Set the recent-window duration used by ``fetch_recent_window``.

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


async def fetch_recent_window(
    table: str,
    relational_provider: Any,
    *,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    filter_tags: Optional[Dict[str, Any]] = None,
    scopes: Optional[List[str]] = None,
    recent_cap: int = DEFAULT_RECENT_CAP,
) -> List[Dict[str, Any]]:
    """Fetch rows written within the recent indexing-lag window.

    Queries the Relational provider for rows whose ``created_at`` or
    ``updated_at`` falls within the configured recent window. These are the
    just-written candidates the Search provider may not have indexed yet; the
    caller unions them into its ranked list to detect near-duplicate writes.

    Args:
        table: Logical table name (e.g. ``"semantic_memory"``).
        relational_provider: Registered Relational DB provider instance.
        user_id: User scope. None for org-wide queries.
        organization_id: Organization scope.
        filter_tags: Tag filter dict.
        scopes: Optional scope list for filter_tags.scope.
        recent_cap: Max rows returned. Defaults to 500. The cap protects
            prompt size against a pathological burst of writes in the window.

    Returns:
        A list of recent rows as raw dicts, in Relational-provider order
        (no ranking applied), capped at ``recent_cap``.

    Raises:
        Any exception raised by the Relational call. Fail-closed.
    """
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
        limit=recent_cap,
    )

    return list(recent_records)
