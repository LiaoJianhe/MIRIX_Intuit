"""
Redis cache provider for Mirix.

RedisUnifiedCacheProvider wraps both async and sync Redis clients. Sync methods
use sync client, async methods use async client. This is the only Redis cache
provider used in Mirix.

RedisCacheProvider is an alias for RedisUnifiedCacheProvider for backward compatibility.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mirix.log import get_logger

if TYPE_CHECKING:
    from mirix.database.redis_client import RedisMemoryClient
    from mirix.database.redis_sync_client import RedisSyncMemoryClient

logger = get_logger(__name__)




class RedisUnifiedCacheProvider:
    """
    Unified Redis cache provider (default).

    Wraps both async and sync Redis clients. Sync methods use sync client;
    async methods use async client. Provides both sync and async cache operations.
    """

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

    def __init__(
        self,
        async_client: "RedisMemoryClient",
        sync_client: "RedisSyncMemoryClient",
    ) -> None:
        """Initialize with async and sync Redis clients."""
        self.async_client = async_client
        self.sync_client = sync_client
        logger.info("Initialized RedisUnifiedCacheProvider")

    # ========================================================================
    # ASYNC METHODS (delegate to async_client)
    # ========================================================================

    async def aget(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value (string/JSON) from Redis (async)."""
        try:
            return await self.async_client.get_json(key)
        except Exception as e:
            logger.warning("Redis get failed for key %s: %s", key, e)
            return None

    async def aset(
        self, key: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set value (JSON) in Redis (async)."""
        try:
            return await self.async_client.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set failed for key %s: %s", key, e)
            return False

    async def adelete(self, key: str) -> bool:
        """Delete key from Redis (async)."""
        try:
            return await self.async_client.delete(key)
        except Exception as e:
            logger.warning("Redis delete failed for key %s: %s", key, e)
            return False

    async def aget_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Get hash from Redis (async)."""
        try:
            return await self.async_client.get_hash(key)
        except Exception as e:
            logger.warning("Redis get_hash failed for key %s: %s", key, e)
            return None

    async def aset_hash(
        self, key: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set hash in Redis (async)."""
        try:
            return await self.async_client.set_hash(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set_hash failed for key %s: %s", key, e)
            return False

    async def aget_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON from Redis (async)."""
        try:
            return await self.async_client.get_json(key)
        except Exception as e:
            logger.warning("Redis get_json failed for key %s: %s", key, e)
            return None

    async def aset_json(
        self, key: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set JSON in Redis (async)."""
        try:
            return await self.async_client.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set_json failed for key %s: %s", key, e)
            return False

    async def adelete_many(self, keys: List[str]) -> int:
        """Batch delete (async)."""
        if not keys:
            return 0
        try:
            return await self.async_client.client.delete(*keys)
        except Exception as e:
            logger.warning("Redis delete_many failed: %s", e)
            return 0

    async def aupdate_hash_field(
        self, key: str, field: str, value: str
    ) -> bool:
        """Set a single hash field (async). On WRONGTYPE, update as JSON."""
        try:
            await self.async_client.client.hset(key, field, value)
            return True
        except Exception as e:
            err_str = str(e).upper()
            if "WRONGTYPE" in err_str:
                try:
                    data = await self.async_client.get_json(key)
                    if data is not None:
                        data[field] = value
                        await self.async_client.set_json(key, data)
                        return True
                except Exception as fallback_e:
                    logger.warning(
                        "Redis update_hash_field fallback failed for key %s: %s",
                        key,
                        fallback_e,
                    )
                    return False
            logger.warning("Redis hset failed for key %s: %s", key, e)
            return False

    async def aset_string(
        self, key: str, value: str, ttl: Optional[int] = None
    ) -> bool:
        """Set string key with optional TTL (async)."""
        try:
            await self.async_client.client.set(key, value)
            if ttl is not None:
                await self.async_client.client.expire(key, ttl)
            return True
        except Exception as e:
            logger.warning("Redis set_string failed for key %s: %s", key, e)
            return False

    async def aget_string(self, key: str) -> Optional[str]:
        """Get string value (async)."""
        try:
            raw = await self.async_client.client.get(key)
        except Exception as e:
            logger.warning("Redis get failed for key %s: %s", key, e)
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    async def adelete_string(self, key: str) -> bool:
        """Delete string key (async)."""
        try:
            return await self.async_client.client.delete(key)
        except Exception as e:
            logger.warning("Redis delete failed for key %s: %s", key, e)
            return False

    # ========================================================================
    # SYNC METHODS (delegate to sync_client)
    # ========================================================================

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value (JSON) from Redis (sync)."""
        try:
            return self.sync_client.get_json(key)
        except Exception as e:
            logger.warning("Redis get failed for key %s: %s", key, e)
            return None

    def set(
        self, key: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set value (JSON) in Redis (sync)."""
        try:
            return self.sync_client.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set failed for key %s: %s", key, e)
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis (sync)."""
        try:
            return self.sync_client.delete(key)
        except Exception as e:
            logger.warning("Redis delete failed for key %s: %s", key, e)
            return False

    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Get hash from Redis (sync)."""
        try:
            return self.sync_client.get_hash(key)
        except Exception as e:
            logger.warning("Redis get_hash failed for key %s: %s", key, e)
            return None

    def set_hash(
        self, key: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set hash in Redis (sync)."""
        try:
            return self.sync_client.set_hash(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set_hash failed for key %s: %s", key, e)
            return False

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON from Redis (sync)."""
        try:
            return self.sync_client.get_json(key)
        except Exception as e:
            logger.warning("Redis get_json failed for key %s: %s", key, e)
            return None

    def set_json(
        self, key: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set JSON in Redis (sync)."""
        try:
            return self.sync_client.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set_json failed for key %s: %s", key, e)
            return False

    def delete_many(self, keys: List[str]) -> int:
        """Batch delete (sync); returns count of keys deleted."""
        if not keys:
            return 0
        try:
            result = self.sync_client.client.delete(*keys)
            return result if isinstance(result, int) else 0
        except Exception as e:
            logger.warning("Redis delete_many failed: %s", e)
            return 0

    def update_hash_field(self, key: str, field: str, value: str) -> bool:
        """Set a single hash field (sync). On WRONGTYPE, update as JSON."""
        try:
            self.sync_client.client.hset(key, field, value)
            return True
        except Exception as e:
            err_str = str(e).upper()
            if "WRONGTYPE" in err_str:
                try:
                    data = self.sync_client.get_json(key)
                    if data is not None:
                        data[field] = value
                        self.sync_client.set_json(key, data)
                        return True
                except Exception as fallback_e:
                    logger.warning(
                        "Redis update_hash_field fallback failed for key %s: %s",
                        key,
                        fallback_e,
                    )
                    return False
            logger.warning("Redis hset failed for key %s: %s", key, e)
            return False

    def set_string(
        self, key: str, value: str, ttl: Optional[int] = None
    ) -> bool:
        """Set string key with optional TTL (sync)."""
        try:
            self.sync_client.client.set(key, value)
            if ttl is not None:
                self.sync_client.client.expire(key, ttl)
            return True
        except Exception as e:
            logger.warning("Redis set_string failed for key %s: %s", key, e)
            return False

    def get_string(self, key: str) -> Optional[str]:
        """Get string value (sync)."""
        try:
            raw = self.sync_client.client.get(key)
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
        try:
            return self.sync_client.delete(key)
        except Exception as e:
            logger.warning("Redis delete failed for key %s: %s", key, e)
            return False


# Backward compatibility alias
RedisCacheProvider = RedisUnifiedCacheProvider


