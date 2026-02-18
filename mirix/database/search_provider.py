"""
Search provider interface and registry for Mirix.

Search providers implement the interface via duck typing (no base class).
Used for vector similarity, full-text, and recency-sorted search. When no
search provider is registered, managers fall back to database-only paths.

Expected interface (duck typing):
    - search_recent(index_name, limit, user_id, organization_id, ...) -> List[Dict]
    - search_vector(index_name, embedding, vector_field, limit, ...) -> List[Dict]
    - search_text(index_name, query, search_fields, limit, ...) -> List[Dict]
    - search_recent_by_org(index_name, limit, organization_id, ...) -> List[Dict]
    - search_vector_by_org(index_name, embedding, vector_field, limit, ...) -> List[Dict]
    - search_text_by_org(index_name, query_text, search_field, search_method, limit, ...) -> List[Dict]
    - clean_search_fields(items: List[Dict]) -> List[Dict]

Index name constants (e.g. EPISODIC_INDEX, SEMANTIC_INDEX) are provided by
the implementation (e.g. RedisSearchProvider).
"""

from typing import Any, Dict, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

_search_providers: Dict[str, Any] = {}
_active_search_provider: Optional[str] = None


def register_search_provider(name: str, provider: Any) -> None:
    """
    Register a search provider.

    Args:
        name: Provider identifier (e.g. "redis").
        provider: Provider instance implementing the search interface.
    """
    global _search_providers, _active_search_provider
    _search_providers[name] = provider
    _active_search_provider = name
    logger.info("Registered search provider: %s", name)


def get_search_provider() -> Optional[Any]:
    """
    Get the active search provider.

    Returns None if no provider is registered (managers fall back to DB).
    """
    if _active_search_provider and _active_search_provider in _search_providers:
        return _search_providers[_active_search_provider]
    return None


def unregister_search_provider(name: str) -> None:
    """Unregister a search provider."""
    global _search_providers, _active_search_provider
    if name in _search_providers:
        del _search_providers[name]
        if _active_search_provider == name:
            _active_search_provider = None
        logger.info("Unregistered search provider: %s", name)
