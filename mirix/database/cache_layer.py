"""
Generic async cache-aside helper for Mirix.

Provides a reusable function that replaces the need for per-manager async
method variants. Each REST API handler composes this with its manager's
sync DB method.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Type

from mirix.database.cache_provider import (
    acache_delete,
    acache_get_hash,
    acache_get_json,
    acache_set_hash,
    acache_set_json,
)

logger = logging.getLogger(__name__)

_GETTERS = {"hash": acache_get_hash, "json": acache_get_json}
_SETTERS = {"hash": acache_set_hash, "json": acache_set_json}


async def async_cache_read(
    cache_key: str,
    db_fn: Callable[[], Any],
    *,
    method: str = "hash",
    ttl: Optional[int] = None,
    model_class: Optional[Type] = None,
    fixups: Optional[Dict[str, Any]] = None,
    serializer: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> Any:
    """
    Async cache-aside read.

    1. Try async cache read (zero threads if provider has async methods).
    2. On miss: run db_fn in a thread (SQLAlchemy is sync).
    3. Populate cache asynchronously.

    Args:
        cache_key:    Full cache key including prefix (e.g., "block:<id>").
        db_fn:        Zero-arg callable for the DB fallback. Will be run via
                      asyncio.to_thread(). Should return a Pydantic model or
                      None. Typically a lambda wrapping the manager's sync
                      method with use_cache=False.
        method:       "hash" or "json" -- which cache storage method to use.
        ttl:          TTL in seconds for cache population.
        model_class:  Pydantic model class for deserializing cached data.
                      If None, returns the raw dict from cache.
        fixups:       Dict of {field: default_value} applied to cached data
                      before deserialization.
        serializer:   Optional callable(result) -> dict for custom cache
                      serialization. Defaults to result.model_dump(mode="json").

    Returns:
        The entity (from cache or DB), or None if not found.
    """
    getter = _GETTERS[method]
    setter = _SETTERS[method]

    # 1. Async cache read (zero threads if provider supports async)
    try:
        cached_data = await getter(cache_key)
        if cached_data is not None:
            if fixups:
                for field, default in fixups.items():
                    if field not in cached_data or cached_data[field] is None:
                        cached_data[field] = default
            if model_class:
                return model_class(**cached_data)
            return cached_data
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", cache_key, e)

    # 2. DB fallback in thread (SQLAlchemy is sync)
    result = await asyncio.to_thread(db_fn)
    if result is None:
        return None

    # 3. Populate cache asynchronously
    try:
        if serializer:
            data = serializer(result)
        elif hasattr(result, "model_dump"):
            data = result.model_dump(mode="json")
        else:
            data = result
        await setter(cache_key, data, ttl=ttl)
    except Exception as e:
        logger.warning("Failed to populate cache for %s: %s", cache_key, e)

    return result


async def async_cache_invalidate(*cache_keys: str) -> None:
    """
    Invalidate one or more cache keys asynchronously.

    For use in REST API handlers that need to invalidate cache after an
    async operation. For sync write paths (ORM lifecycle), use
    sync_cache_delete() (or other sync_cache_* helpers) so cache works
    with async-only providers when the sync bridge is set.

    Args:
        *cache_keys: One or more full cache keys to delete.
    """
    for key in cache_keys:
        try:
            await acache_delete(key)
        except Exception as e:
            logger.warning("Cache invalidation failed for %s: %s", key, e)
