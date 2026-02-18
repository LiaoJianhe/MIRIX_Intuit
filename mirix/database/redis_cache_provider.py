"""
Redis cache provider for Mirix.

Wraps the existing RedisMemoryClient to implement the cache provider interface.
Used when Mirix runs standalone with Redis; ECMS can register IPS Cache instead.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mirix.log import get_logger

if TYPE_CHECKING:
    from mirix.database.redis_client import RedisMemoryClient

logger = get_logger(__name__)

_SYNC_CACHE_TIMEOUT = 5


class RedisCacheProvider:
    """
    Redis cache provider implementation.

    Wraps RedisMemoryClient to provide the cache provider interface used by
    service managers and ORM. All operations delegate to the Redis client;
    errors are logged and return None/False for graceful fallback.
    """

    # Key prefixes (must match RedisMemoryClient for key compatibility)
    BLOCK_PREFIX = "block:"
    MESSAGE_PREFIX = "msg:"
    EPISODIC_PREFIX = "episodic:"
    SEMANTIC_PREFIX = "semantic:"
    PROCEDURAL_PREFIX = "procedural:"
    RESOURCE_PREFIX = "resource:"
    KNOWLEDGE_PREFIX = "knowledge:"
    RAW_MEMORY_PREFIX = "raw_memory:"
    ORGANIZATION_PREFIX = "org:"
    USER_PREFIX = "user:"
    CLIENT_PREFIX = "client:"
    AGENT_PREFIX = "agent:"
    TOOL_PREFIX = "tool:"

    def __init__(self, redis_client: "RedisMemoryClient") -> None:
        """
        Initialize Redis cache provider.

        Args:
            redis_client: Existing RedisMemoryClient instance.
        """
        self.redis_client = redis_client
        logger.info("Initialized RedisCacheProvider")

    def _run_async(self, coro: Any) -> Any:
        """Run async coroutine from sync context (for ORM base)."""
        import threading

        from mirix.database.sync_bridge import get_event_loop, get_event_loop_thread_id

        loop = get_event_loop()
        if loop is None:
            return None
        # Deadlock: if we're on the thread that runs the event loop, blocking in
        # future.result() prevents the loop from running the scheduled coroutine.
        # Return None (cache miss) so callers fall back to DB.
        loop_thread_id = get_event_loop_thread_id()
        if loop_thread_id is not None and threading.current_thread().ident == loop_thread_id:
            return None
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=_SYNC_CACHE_TIMEOUT)
        except Exception as e:
            logger.warning("Redis sync bridge failed: %s", e)
            return None

    async def aget(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value (string/JSON) from Redis (async)."""
        try:
            return await self.redis_client.get_json(key)
        except Exception as e:
            logger.warning("Redis get failed for key %s: %s", key, e)
            return None

    async def aset(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value (JSON) in Redis (async)."""
        try:
            return await self.redis_client.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set failed for key %s: %s", key, e)
            return False

    async def adelete(self, key: str) -> bool:
        """Delete key from Redis (async)."""
        try:
            return await self.redis_client.delete(key)
        except Exception as e:
            logger.warning("Redis delete failed for key %s: %s", key, e)
            return False

    async def aget_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Get hash from Redis (async)."""
        try:
            return await self.redis_client.get_hash(key)
        except Exception as e:
            logger.warning("Redis get_hash failed for key %s: %s", key, e)
            return None

    async def aset_hash(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set hash in Redis (async)."""
        try:
            return await self.redis_client.set_hash(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set_hash failed for key %s: %s", key, e)
            return False

    async def aget_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON from Redis (async)."""
        try:
            return await self.redis_client.get_json(key)
        except Exception as e:
            logger.warning("Redis get_json failed for key %s: %s", key, e)
            return None

    async def aset_json(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set JSON in Redis (async)."""
        try:
            return await self.redis_client.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set_json failed for key %s: %s", key, e)
            return False

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value (string/JSON) from Redis (sync wrapper for ORM)."""
        return self._run_async(self.redis_client.get_json(key))

    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value (JSON) in Redis (sync wrapper for ORM)."""
        result = self._run_async(self.redis_client.set_json(key, data, ttl))
        return result is True

    def delete(self, key: str) -> bool:
        """Delete key from Redis (sync wrapper for ORM)."""
        result = self._run_async(self.redis_client.delete(key))
        return result is True

    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Get hash from Redis (sync wrapper for ORM)."""
        return self._run_async(self.redis_client.get_hash(key))

    def set_hash(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set hash in Redis (sync wrapper for ORM)."""
        result = self._run_async(self.redis_client.set_hash(key, data, ttl))
        return result is True

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON from Redis (sync wrapper for ORM)."""
        return self._run_async(self.redis_client.get_json(key))

    def set_json(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set JSON in Redis (sync wrapper for ORM)."""
        result = self._run_async(self.redis_client.set_json(key, data, ttl))
        return result is True

    # ── Extended interface: batch delete, hash field, string keys ─────────────

    def delete_many(self, keys: List[str]) -> int:
        """Batch delete; returns count of keys deleted (sync)."""
        if not keys:
            return 0
        result = self._run_async(self.redis_client.client.delete(*keys))
        return result if isinstance(result, int) else 0

    async def adelete_many(self, keys: List[str]) -> int:
        """Batch delete (async)."""
        if not keys:
            return 0
        try:
            return await self.redis_client.client.delete(*keys)
        except Exception as e:
            logger.warning("Redis delete_many failed: %s", e)
            return 0

    def update_hash_field(self, key: str, field: str, value: str) -> bool:
        """Set a single hash field (sync)."""
        result = self._run_async(
            self.redis_client.client.hset(key, field, value)
        )
        return result is not None

    async def aupdate_hash_field(self, key: str, field: str, value: str) -> bool:
        """Set a single hash field (async)."""
        try:
            await self.redis_client.client.hset(key, field, value)
            return True
        except Exception as e:
            logger.warning("Redis hset failed for key %s: %s", key, e)
            return False

    def set_string(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set string key with optional TTL (sync)."""
        async def _set() -> bool:
            await self.redis_client.client.set(key, value)
            if ttl is not None:
                await self.redis_client.client.expire(key, ttl)
            return True
        result = self._run_async(_set())
        return result is True

    async def aset_string(
        self, key: str, value: str, ttl: Optional[int] = None
    ) -> bool:
        """Set string key with optional TTL (async)."""
        try:
            await self.redis_client.client.set(key, value)
            if ttl is not None:
                await self.redis_client.client.expire(key, ttl)
            return True
        except Exception as e:
            logger.warning("Redis set_string failed for key %s: %s", key, e)
            return False

    def get_string(self, key: str) -> Optional[str]:
        """Get string value (sync)."""
        raw = self._run_async(self.redis_client.client.get(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    async def aget_string(self, key: str) -> Optional[str]:
        """Get string value (async)."""
        try:
            raw = await self.redis_client.client.get(key)
        except Exception as e:
            logger.warning("Redis get failed for key %s: %s", key, e)
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def delete_string(self, key: str) -> bool:
        """Delete string key (sync)."""
        result = self._run_async(self.redis_client.client.delete(key))
        return result is True

    async def adelete_string(self, key: str) -> bool:
        """Delete string key (async)."""
        try:
            return await self.redis_client.client.delete(key)
        except Exception as e:
            logger.warning("Redis delete failed for key %s: %s", key, e)
            return False
