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

Extended sync methods (duck typing):
    - delete_many(keys: List[str]) -> int
    - update_hash_field(key: str, field: str, value: str) -> bool
    - set_string(key: str, value: str, ttl: Optional[int] = None) -> bool
    - get_string(key: str) -> Optional[str]
    - delete_string(key: str) -> bool

Optional async methods (providers MAY implement for zero-thread async I/O):
    - aget(key: str) -> Optional[Dict[str, Any]]
    - aset(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - adelete(key: str) -> bool
    - aget_hash(key: str) -> Optional[Dict[str, Any]]
    - aset_hash(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - aget_json(key: str) -> Optional[Dict[str, Any]]
    - aset_json(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - adelete_many(keys: List[str]) -> int
    - aupdate_hash_field(key: str, field: str, value: str) -> bool
    - aset_string(key: str, value: str, ttl: Optional[int] = None) -> bool
    - aget_string(key: str) -> Optional[str]
    - adelete_string(key: str) -> bool

Async dispatch helpers (acache_get, acache_set_hash, etc.):
    If the active provider has async methods -> call them directly (zero threads).
    If not -> fall back to asyncio.to_thread() around the sync method.

Providers should implement both sync and async methods. Sync callers (ORM,
managers) use sync_cache_* which call provider sync methods directly. Async
callers (REST handlers) use acache_* which call provider async methods if
available, or fall back to asyncio.to_thread(sync_method).

Key prefix constants (all providers should define these):
    BLOCK_PREFIX, MESSAGE_PREFIX, EPISODIC_PREFIX, SEMANTIC_PREFIX,
    PROCEDURAL_PREFIX, RESOURCE_PREFIX, KNOWLEDGE_PREFIX, RAW_MEMORY_PREFIX,
    ORGANIZATION_PREFIX, USER_PREFIX, CLIENT_PREFIX, AGENT_PREFIX, TOOL_PREFIX

Search provider: For vector/text/recency search, use mirix.database.search_provider
(get_search_provider(), register_search_provider). When Redis is enabled, both
cache and search providers are registered; managers use get_cache_provider() for
key-value operations and get_search_provider() for search, with DB fallback when
no provider is registered.

Usage:
    # Provider implementing both sync and async methods (recommended)
    from mirix.database.cache_provider import register_cache_provider
    from mirix.database.redis_cache_provider import RedisUnifiedCacheProvider
    from mirix.database.redis_client import RedisMemoryClient
    from mirix.database.redis_sync_client import RedisSyncMemoryClient
    async_client = RedisMemoryClient(redis_uri=...)
    sync_client = RedisSyncMemoryClient(redis_uri=...)
    register_cache_provider("redis", RedisUnifiedCacheProvider(async_client, sync_client))

    # In Mirix service managers (sync)
    from mirix.database.cache_provider import get_cache_provider, sync_cache_get_hash
    cache_provider = get_cache_provider()
    if cache_provider:
        data = sync_cache_get_hash(f"{cache_provider.MESSAGE_PREFIX}{msg_id}")

    # In async REST handlers
    from mirix.database.cache_provider import acache_get_hash
    data = await acache_get_hash(f"{cache_provider.MESSAGE_PREFIX}{msg_id}")
"""

import asyncio
import threading
from typing import Any, Dict, List, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Registry: protected by _registry_lock for writes; _active_provider is read lock-free
_registry_lock = threading.Lock()
_cache_providers: Dict[str, Any] = {}
_active_provider_name: Optional[str] = None
_active_provider: Optional[Any] = None  # Single reference for lock-free get_cache_provider()


def register_cache_provider(name: str, provider: Any) -> None:
    """
    Register a cache provider with Mirix.

    Similar to register_auth_provider() pattern. Last registered provider
    becomes the active one.

    Providers should implement both sync and async methods for the cache
    interface (duck typing). Sync callers (managers) use sync_cache_* which
    call provider sync methods directly. Async callers (REST handlers) use
    acache_* which call provider async methods if available, or fall back to
    asyncio.to_thread(sync_method).

    Args:
        name: Provider identifier (e.g., "redis", "ips_cache").
        provider: Provider instance implementing the cache interface.
    """
    global _active_provider_name, _active_provider

    with _registry_lock:
        _cache_providers[name] = provider
        _active_provider_name = name
        _active_provider = provider
    logger.info("Registered cache provider: %s", name)


def get_cache_provider() -> Optional[Any]:
    """
    Get the active cache provider.

    Lock-free read of the active provider reference. Returns None if no
    provider is registered (graceful fallback to PostgreSQL).

    Returns:
        Cache provider instance or None.
    """
    return _active_provider


def unregister_cache_provider(name: str) -> None:
    """
    Unregister a cache provider.

    Args:
        name: Provider identifier.
    """
    global _active_provider_name, _active_provider

    with _registry_lock:
        if name in _cache_providers:
            del _cache_providers[name]
            if _active_provider_name == name:
                _active_provider_name = None
                _active_provider = None
            logger.info("Unregistered cache provider: %s", name)


def get_registered_providers() -> Dict[str, Any]:
    """
    Get all registered cache providers (for tests).

    Returns:
        Dictionary of provider_name -> provider_instance.
    """
    with _registry_lock:
        return dict(_cache_providers)


# ── Async dispatch ──────────────────────────────────────────────────


async def _acache_dispatch(method: str, *args, **kwargs) -> Any:
    """
    Generic async dispatch for any cache provider method.

    Checks whether the active provider exposes an async variant (a-prefixed).
    If yes -> await it directly (zero threads).
    If no  -> fall back to asyncio.to_thread() around the sync method.
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


async def acache_delete_many(keys: List[str]) -> int:
    """Async batch delete; returns count of keys deleted."""
    result = await _acache_dispatch("delete_many", keys)
    return result if isinstance(result, int) else 0


async def acache_update_hash_field(
    key: str, field: str, value: str
) -> bool:
    """Async partial hash update (e.g. hset key field value)."""
    return await _acache_dispatch("update_hash_field", key, field, value) or False


async def acache_set_string(
    key: str, value: str, ttl: Optional[int] = None
) -> bool:
    """Async set string key with optional TTL."""
    return await _acache_dispatch("set_string", key, value, ttl) or False


async def acache_get_string(key: str) -> Optional[str]:
    """Async get string key."""
    return await _acache_dispatch("get_string", key)


async def acache_delete_string(key: str) -> bool:
    """Async delete string key."""
    return await _acache_dispatch("delete_string", key) or False


# ── Sync cache API (for sync callers; supports async_only providers via bridge) ──


def _sync_cache_dispatch(
    method: str,
    default: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Run cache operation from sync context. Calls provider's sync method directly.
    """
    provider = get_cache_provider()
    if provider is None:
        return default
    fn = getattr(provider, method, None)
    if fn is None:
        logger.warning("Cache provider missing sync method: %s", method)
        return default
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning("Sync cache (provider.%s) failed: %s", method, e)
        return default


def sync_cache_get_hash(key: str) -> Optional[Dict[str, Any]]:
    """Sync get_hash; calls provider.get_hash directly."""
    return _sync_cache_dispatch("get_hash", None, key)


def sync_cache_set_hash(
    key: str, data: Dict[str, Any], ttl: Optional[int] = None
) -> bool:
    """Sync set_hash; calls provider.set_hash directly."""
    result = _sync_cache_dispatch("set_hash", False, key, data, ttl)
    return bool(result)


def sync_cache_get_json(key: str) -> Optional[Dict[str, Any]]:
    """Sync get_json; calls provider.get_json directly."""
    return _sync_cache_dispatch("get_json", None, key)


def sync_cache_set_json(
    key: str, data: Dict[str, Any], ttl: Optional[int] = None
) -> bool:
    """Sync set_json; calls provider.set_json directly."""
    result = _sync_cache_dispatch("set_json", False, key, data, ttl)
    return bool(result)


def sync_cache_delete(key: str) -> bool:
    """Sync delete; calls provider.delete directly."""
    result = _sync_cache_dispatch("delete", False, key)
    return bool(result)


def sync_cache_delete_many(keys: List[str]) -> int:
    """Sync delete_many; calls provider.delete_many directly."""
    result = _sync_cache_dispatch("delete_many", 0, keys)
    return result if isinstance(result, int) else 0


def sync_cache_update_hash_field(key: str, field: str, value: str) -> bool:
    """Sync update_hash_field; calls provider.update_hash_field directly."""
    result = _sync_cache_dispatch("update_hash_field", False, key, field, value)
    return bool(result)


def sync_cache_set_string(
    key: str, value: str, ttl: Optional[int] = None
) -> bool:
    """Sync set_string; calls provider.set_string directly."""
    result = _sync_cache_dispatch("set_string", False, key, value, ttl)
    return bool(result)


def sync_cache_get_string(key: str) -> Optional[str]:
    """Sync get_string; calls provider.get_string directly."""
    return _sync_cache_dispatch("get_string", None, key)


def sync_cache_delete_string(key: str) -> bool:
    """Sync delete_string; calls provider.delete_string directly."""
    result = _sync_cache_dispatch("delete_string", False, key)
    return bool(result)
