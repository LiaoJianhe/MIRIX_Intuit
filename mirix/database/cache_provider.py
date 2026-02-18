"""
Cache provider interface and registry for Mirix.

Cache providers implement the interface via duck typing (no base class
required). Similar to the auth_provider pattern in mirix.llm_api.auth_provider.

Expected sync methods (duck typing - existing, unchanged):
    - get(key: str) -> Optional[Dict[str, Any]]
    - set(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - delete(key: str) -> bool
    - get_hash(key: str) -> Optional[Dict[str, Any]]
    - set_hash(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - get_json(key: str) -> Optional[Dict[str, Any]]
    - set_json(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool

Optional async methods (providers MAY implement for zero-thread async I/O):
    - aget(key: str) -> Optional[Dict[str, Any]]
    - aset(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - adelete(key: str) -> bool
    - aget_hash(key: str) -> Optional[Dict[str, Any]]
    - aset_hash(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - aget_json(key: str) -> Optional[Dict[str, Any]]
    - aset_json(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool

Async dispatch helpers (acache_get, acache_set_hash, etc.):
    If the active provider has async methods -> call them directly (zero threads).
    If not -> fall back to asyncio.to_thread() around the sync method.
    This makes RedisCacheProvider (sync-only) work without any changes.

Key prefix constants (all providers should define these):
    BLOCK_PREFIX, MESSAGE_PREFIX, EPISODIC_PREFIX, SEMANTIC_PREFIX,
    PROCEDURAL_PREFIX, RESOURCE_PREFIX, KNOWLEDGE_PREFIX, RAW_MEMORY_PREFIX,
    ORGANIZATION_PREFIX, USER_PREFIX, CLIENT_PREFIX, AGENT_PREFIX, TOOL_PREFIX

Usage:
    # In external project
    from mirix.database.cache_provider import register_cache_provider
    cache_provider = MyCustomCacheProvider(config)
    register_cache_provider("my_cache", cache_provider)

    # In Mirix service managers (sync)
    from mirix.database.cache_provider import get_cache_provider
    cache_provider = get_cache_provider()
    if cache_provider:
        data = cache_provider.get_hash(f"{cache_provider.MESSAGE_PREFIX}{msg_id}")

    # In async REST handlers
    from mirix.database.cache_provider import acache_get_hash
    data = await acache_get_hash(f"{cache_provider.MESSAGE_PREFIX}{msg_id}")
"""

import asyncio
from typing import Any, Dict, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Global cache provider registry (simple dictionary)
_cache_providers: Dict[str, Any] = {}
_active_provider_name: Optional[str] = None


def register_cache_provider(name: str, provider: Any) -> None:
    """
    Register a cache provider with Mirix.

    Similar to register_auth_provider() pattern. Last registered provider
    becomes the active one.

    Args:
        name: Provider identifier (e.g., "redis", "ips_cache").
        provider: Provider instance implementing the cache interface.
    """
    global _cache_providers, _active_provider_name

    _cache_providers[name] = provider
    _active_provider_name = name
    logger.info("Registered cache provider: %s", name)


def get_cache_provider() -> Optional[Any]:
    """
    Get the active cache provider.

    Returns None if no provider is registered (graceful fallback to PostgreSQL).

    Returns:
        Cache provider instance or None.
    """
    if _active_provider_name and _active_provider_name in _cache_providers:
        return _cache_providers[_active_provider_name]
    return None


def unregister_cache_provider(name: str) -> None:
    """
    Unregister a cache provider.

    Args:
        name: Provider identifier.
    """
    global _cache_providers, _active_provider_name

    if name in _cache_providers:
        del _cache_providers[name]
        if _active_provider_name == name:
            _active_provider_name = None
        logger.info("Unregistered cache provider: %s", name)


def get_registered_providers() -> Dict[str, Any]:
    """
    Get all registered cache providers (for tests).

    Returns:
        Dictionary of provider_name -> provider_instance.
    """
    return dict(_cache_providers)


# ── Async dispatch ──────────────────────────────────────────────────


async def _acache_dispatch(method: str, *args, **kwargs) -> Any:
    """
    Generic async dispatch for any cache provider method.

    Checks whether the active provider exposes an async variant (a-prefixed).
    If yes -> await it directly (zero threads, ideal for IPS Cache).
    If no  -> fall back to asyncio.to_thread() around the sync method
             (works for RedisCacheProvider without any changes).
    """
    provider = get_cache_provider()
    if provider is None:
        return None
    async_method = f"a{method}"
    if hasattr(provider, async_method):
        return await getattr(provider, async_method)(*args, **kwargs)
    return await asyncio.to_thread(
        getattr(provider, method), *args, **kwargs
    )


# ── Typed public wrappers ───────────────────────────────────────────


async def acache_get(key: str) -> Optional[Dict[str, Any]]:
    """Async get (string/JSON value)."""
    return await _acache_dispatch("get", key)


async def acache_set(
    key: str, data: Dict[str, Any], ttl: Optional[int] = None
) -> bool:
    """Async set (string/JSON value)."""
    return await _acache_dispatch("set", key, data, ttl) or False


async def acache_delete(key: str) -> bool:
    """Async delete."""
    return await _acache_dispatch("delete", key) or False


async def acache_get_hash(key: str) -> Optional[Dict[str, Any]]:
    """Async get_hash (Redis HGETALL equivalent)."""
    return await _acache_dispatch("get_hash", key)


async def acache_set_hash(
    key: str, data: Dict[str, Any], ttl: Optional[int] = None
) -> bool:
    """Async set_hash (Redis HSET equivalent)."""
    return await _acache_dispatch("set_hash", key, data, ttl) or False


async def acache_get_json(key: str) -> Optional[Dict[str, Any]]:
    """Async get_json."""
    return await _acache_dispatch("get_json", key)


async def acache_set_json(
    key: str, data: Dict[str, Any], ttl: Optional[int] = None
) -> bool:
    """Async set_json."""
    return await _acache_dispatch("set_json", key, data, ttl) or False
