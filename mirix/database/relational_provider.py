"""
Relational database provider interface and registry for Mirix.

Relational providers implement the interface via duck typing (no base class
required). Same pattern as cache_provider.py.
All methods are async; callers must await them.

Expected methods (duck typing, all async):

    async create(table: str, data: dict, event_context=None) -> dict
    async read(table: str, identifier: str, include_relationships=None) -> Optional[dict]
    async list(
        table: str, *,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        scopes: Optional[List[str]] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        time_range: Optional[Dict[str, Optional[datetime]]] = None,
        include_relationships: Optional[list] = None,
        **kwargs,
    ) -> list[dict]
    async update(table: str, identifier: str, data: dict, event_context=None) -> dict
    async delete(table: str, identifier: str, soft: bool = True, event_context=None) -> bool
    async hard_delete(table: str, identifier: str, event_context=None) -> bool
    async size(
        table: str, *,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        scopes: Optional[List[str]] = None,
    ) -> int
    async find_using_filter(
        table: str, filter_item: Any, pageable_filter: Any = None,
        limit: Optional[int] = None, sort: Optional[str] = None,
    ) -> list[dict]
    async bulk_delete(
        table: str, identifiers: list[str], soft: bool = True,
        event_contexts: Optional[Dict[str, Any]] = None,
    ) -> dict
    async bulk_upsert(
        table: str, entities: list[dict],
        event_contexts: Optional[Dict[str, Any]] = None,
    ) -> dict
    async find_using_named_query(
        table: str, query_name: str, *,
        params: Optional[Dict[str, Any]] = None,
        hash_params: Optional[Dict[str, Any]] = None,
        page_size: int = 50, page_num: int = 0,
        page_type=None,
        result_set_entity_class: Optional[type] = None,
    ) -> list
    async mutate_using_named_query(
        table: str, query_name: str, *,
        params: Optional[Dict[str, Any]] = None,
        hash_params: Optional[Dict[str, Any]] = None,
        event_context: Any = None,
    ) -> int

Interface design notes:
    - ``list`` includes organization_id, user_id, filter_tags, scopes, sort,
      time_range, and cursor because MIRIX managers pass these for scoped,
      filtered, paginated queries (matching apply_filter_tags_sqlalchemy,
      apply_access_predicate, keyset pagination).
    - ``size`` includes scoping parameters because get_total_number_of_items
      counts are always scoped.
    - ``find_using_filter`` exposes the Relational DB provider SDK's
      find_using_filter_query for ad-hoc filter queries needed by admin
      endpoints and hybrid-read recent-record lookups.
    - CUD methods accept optional event_context for domain event propagation
      to Search provider (memory tables only).
    - ``bulk_delete`` and ``bulk_upsert`` support batch operations with
      per-record event contexts.

Usage:
    # In ECMS startup
    from mirix.database.relational_provider import register_relational_provider
    provider = AsyncIPSRelationalProvider(config)
    await provider.initialize()
    register_relational_provider("ips_relational", provider)

    # In Mirix service managers
    from mirix.database.relational_provider import get_relational_provider
    provider = get_relational_provider()
    if provider:
        result = await provider.read("agents", agent_id, include_relationships=["tools"])
"""

from typing import Any, Dict, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

_relational_providers: Dict[str, Any] = {}
_active_provider_name: Optional[str] = None


def register_relational_provider(name: str, provider: Any) -> None:
    """
    Register a relational database provider with Mirix.

    Last registered provider becomes the active one.

    Args:
        name: Provider identifier (e.g., "ips_relational").
        provider: Provider instance implementing the relational interface.
    """
    global _relational_providers, _active_provider_name

    _relational_providers[name] = provider
    _active_provider_name = name
    logger.info("Registered relational provider: %s", name)


def get_relational_provider() -> Optional[Any]:
    """
    Get the active relational database provider.

    Returns None if no provider is registered (graceful fallback to PostgreSQL).

    Returns:
        Relational provider instance or None.
    """
    if _active_provider_name and _active_provider_name in _relational_providers:
        return _relational_providers[_active_provider_name]
    return None


def unregister_relational_provider(name: str) -> None:
    """
    Unregister a relational provider (primarily for test isolation).

    Args:
        name: Provider identifier.
    """
    global _relational_providers, _active_provider_name

    if name in _relational_providers:
        del _relational_providers[name]
        if _active_provider_name == name:
            _active_provider_name = None
        logger.info("Unregistered relational provider: %s", name)


def get_registered_relational_providers() -> Dict[str, Any]:
    """
    Get all registered relational providers (for tests/inspection).

    Returns:
        Dictionary of provider_name -> provider_instance.
    """
    return dict(_relational_providers)
