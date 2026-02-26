"""
Tests for the sync cache provider path (sync client + sync provider + dispatcher).

Sync provider: sync callers use provider sync methods directly; async callers
use asyncio.to_thread(sync_method) fallback.
"""

import pytest

from mirix.database.cache_provider import (
    acache_get_hash,
    get_registered_providers,
    register_cache_provider,
    sync_cache_get_hash,
    sync_cache_set_hash,
    unregister_cache_provider,
)


class MockSyncCacheProvider:
    """Sync-only mock cache provider (no a-prefixed methods)."""

    MESSAGE_PREFIX = "msg:"
    BLOCK_PREFIX = "block:"
    AGENT_PREFIX = "agent:"
    TOOL_PREFIX = "tool:"

    def get(self, key: str):
        return {"sync": "data"}

    def set(self, key: str, data: dict, ttl=None):
        return True

    def delete(self, key: str):
        return True

    def get_hash(self, key: str):
        return {"sync": "hash"}

    def set_hash(self, key: str, data: dict, ttl=None):
        return True

    def get_json(self, key: str):
        return {"sync": "json"}

    def set_json(self, key: str, data: dict, ttl=None):
        return True

    def delete_many(self, keys: list):
        return len(keys)

    def update_hash_field(self, key: str, field: str, value: str):
        return True

    def set_string(self, key: str, value: str, ttl=None):
        return True

    def get_string(self, key: str):
        return "sync_string"

    def delete_string(self, key: str):
        return True


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up cache provider registry before and after each test."""
    for name in list(get_registered_providers().keys()):
        unregister_cache_provider(name)
    yield
    for name in list(get_registered_providers().keys()):
        unregister_cache_provider(name)


def test_sync_provider_direct_sync_calls():
    """Register sync-only mock; sync_cache_* call sync methods."""
    provider = MockSyncCacheProvider()
    register_cache_provider("sync_mock", provider)

    data = sync_cache_get_hash("mykey")
    assert data == {"sync": "hash"}
    ok = sync_cache_set_hash("mykey", {"k": "v"})
    assert ok is True

    unregister_cache_provider("sync_mock")


def test_sync_provider_async_fallback():
    """With sync-only provider, acache_get_hash uses asyncio.to_thread(sync get_hash)."""
    import asyncio

    provider = MockSyncCacheProvider()
    register_cache_provider("sync_mock", provider)

    data = asyncio.run(acache_get_hash("mykey"))
    assert data == {"sync": "hash"}

    unregister_cache_provider("sync_mock")


def test_sync_provider_no_bridge_required():
    """Sync dispatch works without set_event_loop_for_sync_bridge()."""
    provider = MockSyncCacheProvider()
    register_cache_provider("sync_mock", provider)

    # No event loop set anywhere; sync path does not need it
    data = sync_cache_get_hash("key")
    assert data == {"sync": "hash"}

    unregister_cache_provider("sync_mock")


