"""
Tests for the async cache provider path (async client + async provider + dispatcher).

Providers with async methods: async callers use provider async methods directly;
sync callers use asyncio.to_thread(sync_method) if provider has sync methods.
"""

import asyncio

import pytest

from mirix.database.cache_provider import (
    acache_get_hash,
    get_registered_providers,
    register_cache_provider,
    sync_cache_get_hash,
    sync_cache_set_hash,
    unregister_cache_provider,
)


class MockAsyncCacheProvider:
    """Async-only mock cache provider (only a-prefixed methods)."""

    MESSAGE_PREFIX = "msg:"
    BLOCK_PREFIX = "block:"

    async def aget(self, key: str):
        return {"async": "data"}

    async def aset(self, key: str, data: dict, ttl=None):
        return True

    async def adelete(self, key: str):
        return True

    async def aget_hash(self, key: str):
        return {"async": "hash"}

    async def aset_hash(self, key: str, data: dict, ttl=None):
        return True

    async def aget_json(self, key: str):
        return {"async": "json"}

    async def aset_json(self, key: str, data: dict, ttl=None):
        return True

    async def adelete_many(self, keys: list):
        return len(keys)

    async def aupdate_hash_field(self, key: str, field: str, value: str):
        return True

    async def aset_string(self, key: str, value: str, ttl=None):
        return True

    async def aget_string(self, key: str):
        return "async_string"

    async def adelete_string(self, key: str):
        return True


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up cache provider registry before and after each test."""
    for name in list(get_registered_providers().keys()):
        unregister_cache_provider(name)
    yield
    for name in list(get_registered_providers().keys()):
        unregister_cache_provider(name)


def test_async_provider_direct_async_calls():
    """acache_get_hash calls aget_hash directly (zero threads)."""
    import asyncio

    provider = MockAsyncCacheProvider()
    register_cache_provider("async_mock", provider)

    data = asyncio.run(acache_get_hash("mykey"))
    assert data == {"async": "hash"}

    unregister_cache_provider("async_mock")


def test_async_only_provider_sync_returns_default():
    """When provider has only async methods (no sync), sync_cache_* returns None/False."""
    provider = MockAsyncCacheProvider()
    register_cache_provider("async_mock", provider)

    # Provider has no sync get_hash method -> getattr returns None -> returns default
    data = sync_cache_get_hash("key")
    assert data is None
    ok = sync_cache_set_hash("key", {"k": "v"})
    assert ok is False

    unregister_cache_provider("async_mock")


