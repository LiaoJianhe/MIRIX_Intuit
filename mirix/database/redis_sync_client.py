"""
Synchronous Redis client for Mirix cache operations.

Uses redis.Redis (sync) for key-value and hash operations. No index creation
or search — those remain in the async RedisMemoryClient. Used by
RedisUnifiedCacheProvider for sync cache operations.
"""

import json
import socket
from typing import Any, Dict, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Global sync Redis client instance (optional; used when sync provider is registered)
_redis_sync_client: Optional["RedisSyncMemoryClient"] = None


class RedisSyncMemoryClient:
    """
    Synchronous Redis client for Mirix cache (hash + JSON key-value only).

    No RediSearch indexes or vector search — use RedisMemoryClient (async)
    for search provider. This client is for cache provider only.
    """

    def __init__(
        self,
        redis_uri: str,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        socket_keepalive: bool = True,
        retry_on_timeout: bool = True,
    ):
        """
        Initialize sync Redis client with connection pool.

        Args:
            redis_uri: Redis connection URI
            max_connections: Maximum connections (default: 50)
            socket_timeout: Read/write timeout in seconds (default: 5)
            socket_connect_timeout: Connect timeout in seconds (default: 5)
            socket_keepalive: Enable TCP keepalive (default: True)
            retry_on_timeout: Retry on timeout errors (default: True)
        """
        try:
            from redis import ConnectionPool, Redis

            self.redis_uri = redis_uri

            socket_keepalive_options = {}
            if socket_keepalive and hasattr(socket, "TCP_KEEPIDLE"):
                socket_keepalive_options = {
                    socket.TCP_KEEPIDLE: 60,
                    socket.TCP_KEEPINTVL: 10,
                    socket.TCP_KEEPCNT: 3,
                }

            self.pool = ConnectionPool.from_url(
                redis_uri,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                socket_keepalive=socket_keepalive,
                socket_keepalive_options=socket_keepalive_options,
                retry_on_timeout=retry_on_timeout,
                decode_responses=True,
                health_check_interval=30,
            )

            self.client = Redis(connection_pool=self.pool)

            logger.info(
                "Redis sync connection pool initialized: %s (max_connections=%d)",
                self._mask_uri(redis_uri),
                max_connections,
            )
        except ImportError:
            logger.error(
                "Redis library not installed. Install with: pip install redis[hiredis]"
            )
            raise
        except Exception as e:
            logger.error("Failed to initialize Redis sync client: %s", e)
            raise

    def _mask_uri(self, uri: str) -> str:
        """Mask password in URI for logging."""
        if "@" in uri and ":" in uri:
            parts = uri.split("@")
            if len(parts) == 2:
                protocol = parts[0].split("://")[0]
                return f"{protocol}://****@{parts[1]}"
        return uri

    def ping(self) -> bool:
        """Test Redis connection."""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error("Redis sync ping failed: %s", e)
            return False

    def close(self) -> None:
        """Close Redis connection pool."""
        try:
            if self.pool:
                self.pool.disconnect()
                logger.info("Redis sync connection pool closed")
        except Exception as e:
            logger.error("Error closing Redis sync pool: %s", e)

    def set_hash(
        self, key: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Store data as Redis Hash (sync)."""
        try:
            flattened = self._flatten_dict(data)
            self.client.hset(key, mapping=flattened)
            if ttl:
                self.client.expire(key, ttl)
            logger.debug("Stored Hash: %s (%d fields)", key, len(flattened))
            return True
        except Exception as e:
            logger.error("Failed to set hash for %s: %s", key, e)
            return False

    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from Redis Hash (sync)."""
        try:
            data = self.client.hgetall(key)
            if not data:
                return None
            result = self._unflatten_dict(data)
            logger.debug("Retrieved Hash: %s", key)
            return result
        except Exception as e:
            logger.error("Failed to get hash for %s: %s", key, e)
            return None

    def set_json(
        self, key: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Store data as Redis JSON (sync)."""
        try:
            self.client.json().set(key, "$", data)
            if ttl:
                self.client.expire(key, ttl)
            logger.debug("Stored JSON: %s", key)
            return True
        except Exception as e:
            logger.error("Failed to set JSON for %s: %s", key, e)
            return False

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from Redis JSON (sync)."""
        try:
            data = self.client.json().get(key)
            if data is None:
                return None
            logger.debug("Retrieved JSON: %s", key)
            return data
        except Exception as e:
            logger.error("Failed to get JSON for %s: %s", key, e)
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from Redis (sync)."""
        try:
            self.client.delete(key)
            logger.debug("Deleted key: %s", key)
            return True
        except Exception as e:
            logger.error("Failed to delete key %s: %s", key, e)
            return False

    def _flatten_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Flatten dictionary for Hash storage. All values converted to strings."""
        flattened = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flattened[key] = json.dumps(value)
            elif isinstance(value, (list, tuple)):
                flattened[key] = json.dumps(value)
            elif value is None:
                flattened[key] = ""
            elif isinstance(value, bool):
                flattened[key] = "true" if value else "false"
            else:
                flattened[key] = str(value)
        return flattened

    def _unflatten_dict(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Convert Hash data back to proper Python types."""
        result = {}
        for key, value in data.items():
            if key in ("limit",):
                try:
                    result[key] = float(value) if "." in value else int(value)
                    continue
                except (ValueError, AttributeError):
                    pass
            if value.lower() in ("true", "false"):
                result[key] = value.lower() == "true"
                continue
            if value and (value.startswith("{") or value.startswith("[")):
                try:
                    result[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            result[key] = value if value else None
        return result


def initialize_redis_sync_client() -> Optional[RedisSyncMemoryClient]:
    """
    Initialize global sync Redis client from settings.

    Used by RedisUnifiedCacheProvider to provide sync cache operations.
    Does not set event loop for sync bridge.
    """
    global _redis_sync_client

    if _redis_sync_client is not None:
        return _redis_sync_client

    try:
        from mirix.settings import settings

        if not settings.redis_enabled:
            logger.info("Redis is disabled (MIRIX_REDIS_ENABLED=false)")
            return None

        redis_uri = settings.mirix_redis_uri
        if not redis_uri:
            logger.warning("Redis enabled but no URI configured")
            return None

        _redis_sync_client = RedisSyncMemoryClient(
            redis_uri=redis_uri,
            max_connections=settings.redis_max_connections_per_worker,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            socket_keepalive=settings.redis_socket_keepalive,
            retry_on_timeout=settings.redis_retry_on_timeout,
        )

        if not _redis_sync_client.ping():
            logger.error("Redis sync ping failed - disabling sync client")
            _redis_sync_client = None
            return None

        logger.info("Redis sync client initialized successfully")
        return _redis_sync_client

    except Exception as e:
        logger.error("Failed to initialize Redis sync client: %s", e)
        _redis_sync_client = None
        return None


def get_redis_sync_client() -> Optional[RedisSyncMemoryClient]:
    """Get the global sync Redis client instance."""
    return _redis_sync_client


def close_redis_sync_client() -> None:
    """Close the global sync Redis client."""
    global _redis_sync_client
    if _redis_sync_client:
        _redis_sync_client.close()
        _redis_sync_client = None
