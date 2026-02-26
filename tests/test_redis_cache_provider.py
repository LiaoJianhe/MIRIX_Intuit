"""
Redis cache provider tests for Mirix.

Tests that RedisCacheProvider (alias for RedisUnifiedCacheProvider) correctly
delegates to both async and sync Redis clients. Also includes unit tests for
RedisUnifiedCacheProvider specifically.

Usage:
    pytest tests/test_redis_cache_provider.py -v
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from mirix.database.redis_cache_provider import RedisCacheProvider


@pytest.fixture
def mock_redis_clients():
    """Mock async and sync Redis clients."""
    async_client = Mock()
    async_client.get_json = AsyncMock(return_value={"test": "data"})
    async_client.set_json = AsyncMock(return_value=True)
    async_client.get_hash = AsyncMock(return_value={"hash": "data"})
    async_client.set_hash = AsyncMock(return_value=True)
    async_client.delete = AsyncMock(return_value=True)
    async_client.client = Mock()
    async_client.client.delete = AsyncMock(return_value=1)
    async_client.client.hset = AsyncMock(return_value=True)
    async_client.client.set = AsyncMock(return_value=True)
    async_client.client.get = AsyncMock(return_value="value")

    sync_client = Mock()
    sync_client.get_json = Mock(return_value={"test": "data"})
    sync_client.set_json = Mock(return_value=True)
    sync_client.get_hash = Mock(return_value={"hash": "data"})
    sync_client.set_hash = Mock(return_value=True)
    sync_client.delete = Mock(return_value=True)
    sync_client.client = Mock()
    sync_client.client.delete = Mock(return_value=1)
    sync_client.client.hset = Mock(return_value=True)
    sync_client.client.set = Mock(return_value=True)
    sync_client.client.get = Mock(return_value="value")
    sync_client.client.expire = Mock()

    return async_client, sync_client


def test_redis_provider_has_prefix_constants():
    """Provider exposes same key prefixes as RedisMemoryClient."""
    async_client = Mock()
    sync_client = Mock()
    provider = RedisCacheProvider(async_client, sync_client)
    assert provider.MESSAGE_PREFIX == "msg:"
    assert provider.BLOCK_PREFIX == "block:"
    assert provider.RAW_MEMORY_PREFIX == "raw_memory:"


def test_redis_provider_aget_hash(mock_redis_clients):
    """aget_hash delegates to async client get_hash."""
    async_client, sync_client = mock_redis_clients
    async_client.get_hash = AsyncMock(return_value={"id": "b1", "value": "v1"})
    provider = RedisCacheProvider(async_client, sync_client)

    async def _run():
        return await provider.aget_hash("block:b1")

    result = asyncio.run(_run())
    assert result == {"id": "b1", "value": "v1"}
    async_client.get_hash.assert_called_once_with("block:b1")


def test_redis_provider_aset_hash(mock_redis_clients):
    """aset_hash delegates to async client set_hash with ttl."""
    async_client, sync_client = mock_redis_clients
    provider = RedisCacheProvider(async_client, sync_client)

    async def _run():
        return await provider.aset_hash(
            "block:b1", {"id": "b1", "value": "v1"}, ttl=120
        )

    result = asyncio.run(_run())
    assert result is True
    async_client.set_hash.assert_called_once_with(
        "block:b1", {"id": "b1", "value": "v1"}, ttl=120
    )


def test_redis_provider_aget_hash_returns_none_on_error(mock_redis_clients):
    """aget_hash returns None when async client raises."""
    async_client, sync_client = mock_redis_clients
    async_client.get_hash.side_effect = Exception("Redis error")
    provider = RedisCacheProvider(async_client, sync_client)

    async def _run():
        return await provider.aget_hash("block:b1")

    result = asyncio.run(_run())
    assert result is None


def test_redis_provider_has_async_methods():
    """Provider implements async interface for zero-thread async I/O."""
    async_client = Mock()
    sync_client = Mock()
    provider = RedisCacheProvider(async_client, sync_client)
    assert hasattr(provider, "aget_hash")
    assert hasattr(provider, "aset_hash")
    assert hasattr(provider, "aget_json")
    assert hasattr(provider, "aset_json")
    assert callable(getattr(provider, "aget_hash"))
    assert callable(getattr(provider, "aset_hash"))


def test_redis_provider_has_sync_methods():
    """Provider implements sync interface for direct sync calls."""
    async_client = Mock()
    sync_client = Mock()
    provider = RedisCacheProvider(async_client, sync_client)
    assert hasattr(provider, "get_hash")
    assert hasattr(provider, "set_hash")
    assert hasattr(provider, "get_json")
    assert hasattr(provider, "set_json")
    assert callable(getattr(provider, "get_hash"))
    assert callable(getattr(provider, "set_hash"))


# ── RedisUnifiedCacheProvider unit tests ──


def test_unified_provider_has_prefix_constants():
    """RedisUnifiedCacheProvider exposes all 13 key prefix constants."""
    from mirix.database.redis_cache_provider import RedisUnifiedCacheProvider

    async_client = Mock()
    sync_client = Mock()
    provider = RedisUnifiedCacheProvider(async_client, sync_client)

    assert provider.BLOCK_PREFIX == "block:"
    assert provider.MESSAGE_PREFIX == "msg:"
    assert provider.EPISODIC_PREFIX == "episodic:"
    assert provider.SEMANTIC_PREFIX == "semantic:"
    assert provider.PROCEDURAL_PREFIX == "procedural:"
    assert provider.RESOURCE_PREFIX == "resource:"
    assert provider.KNOWLEDGE_PREFIX == "knowledge:"
    assert provider.RAW_MEMORY_PREFIX == "raw_memory:"
    assert provider.ORGANIZATION_PREFIX == "org:"
    assert provider.USER_PREFIX == "user:"
    assert provider.CLIENT_PREFIX == "client:"
    assert provider.AGENT_PREFIX == "agent:"
    assert provider.TOOL_PREFIX == "tool:"


def test_unified_provider_aget_hash_delegates_to_async_client():
    """aget_hash delegates to async_client.get_hash."""
    import asyncio

    from mirix.database.redis_cache_provider import RedisUnifiedCacheProvider

    async_client = Mock()
    async_client.get_hash = AsyncMock(return_value={"id": "1", "value": "test"})
    sync_client = Mock()

    provider = RedisUnifiedCacheProvider(async_client, sync_client)

    async def _run():
        return await provider.aget_hash("block:1")

    result = asyncio.run(_run())
    assert result == {"id": "1", "value": "test"}
    async_client.get_hash.assert_called_once_with("block:1")


def test_unified_provider_get_hash_delegates_to_sync_client():
    """get_hash delegates to sync_client.get_hash."""
    from mirix.database.redis_cache_provider import RedisUnifiedCacheProvider

    async_client = Mock()
    sync_client = Mock()
    sync_client.get_hash = Mock(return_value={"id": "2", "value": "sync_test"})

    provider = RedisUnifiedCacheProvider(async_client, sync_client)

    result = provider.get_hash("block:2")
    assert result == {"id": "2", "value": "sync_test"}
    sync_client.get_hash.assert_called_once_with("block:2")


def test_unified_provider_async_error_handling():
    """aget_hash returns None on async client errors."""
    import asyncio

    from mirix.database.redis_cache_provider import RedisUnifiedCacheProvider

    async_client = Mock()
    async_client.get_hash = AsyncMock(side_effect=RuntimeError("connection failed"))
    sync_client = Mock()

    provider = RedisUnifiedCacheProvider(async_client, sync_client)

    async def _run():
        return await provider.aget_hash("block:1")

    result = asyncio.run(_run())
    assert result is None


def test_unified_provider_sync_error_handling():
    """get_hash returns None on sync client errors."""
    from mirix.database.redis_cache_provider import RedisUnifiedCacheProvider

    async_client = Mock()
    sync_client = Mock()
    sync_client.get_hash = Mock(side_effect=RuntimeError("connection failed"))

    provider = RedisUnifiedCacheProvider(async_client, sync_client)

    result = provider.get_hash("block:1")
    assert result is None


def test_unified_provider_registered_works_both_paths():
    """Register unified provider; verify both sync_cache_* and acache_* work."""
    import asyncio

    from mirix.database.cache_provider import (
        acache_get_hash,
        register_cache_provider,
        sync_cache_get_hash,
        unregister_cache_provider,
    )
    from mirix.database.redis_cache_provider import RedisUnifiedCacheProvider

    async_client = Mock()
    async_client.get_hash = AsyncMock(return_value={"async": "data"})
    sync_client = Mock()
    sync_client.get_hash = Mock(return_value={"sync": "data"})

    provider = RedisUnifiedCacheProvider(async_client, sync_client)
    register_cache_provider("test_unified", provider)

    # Test sync path
    sync_result = sync_cache_get_hash("key")
    assert sync_result == {"sync": "data"}

    # Test async path
    async def _run_async():
        return await acache_get_hash("key")

    async_result = asyncio.run(_run_async())
    assert async_result == {"async": "data"}

    unregister_cache_provider("test_unified")
