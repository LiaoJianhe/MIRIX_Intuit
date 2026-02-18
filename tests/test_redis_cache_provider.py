"""
Redis cache provider tests for Mirix.

Tests that RedisCacheProvider correctly delegates to RedisMemoryClient
and handles errors gracefully (returns None/False).

Usage:
    pytest tests/test_redis_cache_provider.py -v
"""

import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from mirix.database.redis_cache_provider import RedisCacheProvider


@pytest.fixture(scope="module")
def event_loop_for_sync_bridge():
    """Run an event loop in a background thread so sync provider methods can use run_coroutine_threadsafe.
    RedisCacheProvider uses sync_bridge.get_event_loop() (not Redis-specific)."""
    loop = asyncio.new_event_loop()

    def run():
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    # Give thread time to start and set its ident
    import time
    time.sleep(0.05)

    from mirix.database import sync_bridge

    sync_bridge._event_loop = loop
    sync_bridge._event_loop_thread_id = t.ident
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        sync_bridge._event_loop = None
        sync_bridge._event_loop_thread_id = None


@pytest.fixture
def mock_redis_client(event_loop_for_sync_bridge):
    """Mock RedisMemoryClient with async methods (sync tests use provider's _run_async)."""
    client = Mock()
    client.get_json = AsyncMock(return_value={"test": "data"})
    client.set_json = AsyncMock(return_value=True)
    client.get_hash = AsyncMock(return_value={"hash": "data"})
    client.set_hash = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    return client


def test_redis_provider_get_json(mock_redis_client):
    """get_json delegates to client and returns data."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.get_json("test_key")
    assert result == {"test": "data"}
    mock_redis_client.get_json.assert_called_once_with("test_key")


def test_redis_provider_set_json(mock_redis_client):
    """set_json delegates to client with ttl."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.set_json("test_key", {"k": "v"}, ttl=300)
    assert result is True
    mock_redis_client.set_json.assert_called_once_with("test_key", {"k": "v"}, 300)


def test_redis_provider_get_hash(mock_redis_client):
    """get_hash delegates to client."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.get_hash("test_key")
    assert result == {"hash": "data"}
    mock_redis_client.get_hash.assert_called_once_with("test_key")


def test_redis_provider_set_hash(mock_redis_client):
    """set_hash delegates to client with ttl."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.set_hash("test_key", {"a": 1}, ttl=60)
    assert result is True
    mock_redis_client.set_hash.assert_called_once_with("test_key", {"a": 1}, 60)


def test_redis_provider_delete(mock_redis_client):
    """delete delegates to client."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.delete("test_key")
    assert result is True
    mock_redis_client.delete.assert_called_once_with("test_key")


def test_redis_provider_get_returns_none_on_error(mock_redis_client):
    """get_json returns None when client raises."""
    mock_redis_client.get_json.side_effect = Exception("Redis error")
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.get_json("test_key")
    assert result is None


def test_redis_provider_set_returns_false_on_error(mock_redis_client):
    """set_json returns False when client raises."""
    mock_redis_client.set_json.side_effect = Exception("Redis error")
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.set_json("test_key", {})
    assert result is False


def test_redis_provider_has_prefix_constants():
    """Provider exposes same key prefixes as RedisMemoryClient."""
    client = Mock()
    provider = RedisCacheProvider(client)
    assert provider.MESSAGE_PREFIX == "msg:"
    assert provider.BLOCK_PREFIX == "block:"
    assert provider.RAW_MEMORY_PREFIX == "raw_memory:"


# ---------------------------------------------------------------------------
# PR 55: Async cache provider methods (aget_hash, aset_hash, etc.)
# ---------------------------------------------------------------------------


def test_redis_provider_aget_hash(mock_redis_client):
    """PR 55: aget_hash delegates to client async get_hash."""
    import asyncio
    mock_redis_client.get_hash = AsyncMock(return_value={"id": "b1", "value": "v1"})
    provider = RedisCacheProvider(mock_redis_client)

    async def _run():
        return await provider.aget_hash("block:b1")

    result = asyncio.run(_run())
    assert result == {"id": "b1", "value": "v1"}
    mock_redis_client.get_hash.assert_called_once_with("block:b1")


def test_redis_provider_aset_hash(mock_redis_client):
    """PR 55: aset_hash delegates to client async set_hash with ttl."""
    import asyncio
    provider = RedisCacheProvider(mock_redis_client)

    async def _run():
        return await provider.aset_hash("block:b1", {"id": "b1", "value": "v1"}, ttl=120)

    result = asyncio.run(_run())
    assert result is True
    mock_redis_client.set_hash.assert_called_once_with(
        "block:b1", {"id": "b1", "value": "v1"}, ttl=120
    )


def test_redis_provider_aget_hash_returns_none_on_error(mock_redis_client):
    """PR 55: aget_hash returns None when client raises."""
    import asyncio
    mock_redis_client.get_hash.side_effect = Exception("Redis error")
    provider = RedisCacheProvider(mock_redis_client)

    async def _run():
        return await provider.aget_hash("block:b1")

    result = asyncio.run(_run())
    assert result is None


def test_redis_provider_has_async_methods():
    """PR 55: Provider implements async interface for zero-thread async I/O."""
    client = Mock()
    provider = RedisCacheProvider(client)
    assert hasattr(provider, "aget_hash")
    assert hasattr(provider, "aset_hash")
    assert hasattr(provider, "aget_json")
    assert hasattr(provider, "aset_json")
    assert callable(getattr(provider, "aget_hash"))
    assert callable(getattr(provider, "aset_hash"))
