# How Cache Providers Work

This document describes how Mirix behaves under three configurations: **no cache provider**, **Redis** (sync provider), and an **async cache provider** (e.g. a future IPS Cache implementation). The code paths are the same; only the presence and type of the registered provider change.

---

## 1. No Cache Provider

When no provider is registered, `get_cache_provider()` returns `None`. The system always falls back to PostgreSQL.

### 1.1 Registration

Nothing is registered. Either Redis is disabled (`redis_enabled=False`), Redis failed to connect, or no other provider was registered. The global `_active_provider_name` stays `None`, and `get_cache_provider()` returns `None`.

```python
# cache_provider.py
def get_cache_provider() -> Optional[Any]:
    if _active_provider_name and _active_provider_name in _cache_providers:
        return _cache_providers[_active_provider_name]
    return None
```

### 1.2 Sync path — managers

Managers still call `get_cache_provider()` when `use_cache=True` (default). They get `None` and skip all cache logic:

```python
# Example: block_manager.get_block_by_id(..., use_cache=True)
cache_provider = get_cache_provider() if use_cache else None   # → None

if cache_provider:   # False — skipped
    cached_data = cache_provider.get_hash(cache_key)
    ...

with self.session_maker() as session:
    block = BlockModel.read(...)   # Straight to PostgreSQL
    if cache_provider:   # False — no cache write
        cache_provider.set_hash(...)
    return pydantic_block
```

So: **no cache read, no cache write; every request hits the database.**

### 1.3 Async path — REST handlers

REST handlers that support caching first check for a provider. If there is none, they skip `async_cache_read` and call the manager directly in a thread:

```python
# Example: GET /blocks/{block_id} in rest_api.py
cache_provider = get_cache_provider()
if cache_provider:
    return await async_cache_read(cache_key=..., db_fn=..., ...)
return await asyncio.to_thread(server.block_manager.get_block_by_id, block_id, user)
```

With no provider, the handler takes the `else` branch: **one `asyncio.to_thread(manager.get_block_by_id, ...)`**, and the manager again sees `get_cache_provider() → None` (with default `use_cache=True`), so it goes straight to the DB. No cache layer is involved.

### 1.4 Summary (no provider)

| Path        | What happens                                                                 |
|------------|-------------------------------------------------------------------------------|
| Sync       | `get_cache_provider()` → `None`; manager skips cache, reads/writes DB only. |
| Async REST | Handler sees no provider; calls `asyncio.to_thread(manager.get_*_by_id, ...)`; manager sees no provider, DB only. |

---

## 2. Redis as Cache Provider

When Redis is available, the server registers it as the active cache provider. Redis is **sync-only** (no `aget_hash`, `aset_hash`, etc.), so async calls use a thread pool.

### 2.1 Registration at startup

On server startup, `server.py` calls `initialize_redis_client()`. Inside `redis_client.py`, after the connection and indexes are created, Redis **auto-registers** as the cache provider:

```python
# redis_client.py (inside initialize_redis_client())
from mirix.database.cache_provider import register_cache_provider
from mirix.database.redis_cache_provider import RedisCacheProvider

redis_provider = RedisCacheProvider(_redis_client)
register_cache_provider("redis", redis_provider)
```

`register_cache_provider("redis", redis_provider)` stores the instance in `_cache_providers` and sets `_active_provider_name = "redis"`. `get_cache_provider()` then returns that `RedisCacheProvider` instance.

### 2.2 The adapter: RedisCacheProvider

`RedisCacheProvider` wraps the existing `RedisMemoryClient` and implements only the **sync** cache interface:

- `get` / `set` / `delete`
- `get_hash` / `set_hash`
- `get_json` / `set_json`

Each method delegates to `self.redis_client` and catches exceptions for graceful fallback:

```python
# redis_cache_provider.py
def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
    try:
        return self.redis_client.get_hash(key)
    except Exception as e:
        logger.warning("Redis get_hash failed for key %s: %s", key, e)
        return None
```

There are **no** async methods (`aget_hash`, `aset_hash`, etc.).

### 2.3 Sync path — managers

When sync code (agents, workers, other managers) calls a manager with `use_cache=True`:

```python
cache_provider = get_cache_provider()   # → RedisCacheProvider
cached_data = cache_provider.get_hash(key)   # Sync Redis HGETALL
```

Flow: **manager → RedisCacheProvider.get_hash() → RedisMemoryClient.get_hash() → Redis.** On cache miss, the manager hits PostgreSQL and then calls `cache_provider.set_hash(...)` to populate the cache.

### 2.4 Async path — REST handlers and _acache_dispatch

Async handlers use `async_cache_read()`, which uses the public helpers `acache_get_hash` / `acache_set_hash`. Those call `_acache_dispatch()`:

```python
# cache_provider.py
async def _acache_dispatch(method: str, *args, **kwargs) -> Any:
    provider = get_cache_provider()   # → RedisCacheProvider
    if provider is None:
        return None
    async_method = f"a{method}"   # e.g. "aget_hash"
    if hasattr(provider, async_method):   # False for Redis
        return await getattr(provider, async_method)(*args, **kwargs)
    return await asyncio.to_thread(
        getattr(provider, method), *args, **kwargs
    )
```

Because `RedisCacheProvider` does not define `aget_hash`, `aset_hash`, etc., **every** async cache call goes through `asyncio.to_thread(provider.get_hash, ...)` (or the corresponding set/delete). So Redis I/O runs in a thread and does not block the event loop.

### 2.5 Full async read path (e.g. GET /blocks/{id})

End-to-end for a REST read when Redis is the provider:

```
REST handler (async)
  └─ cache_provider = get_cache_provider()  → RedisCacheProvider
  └─ async_cache_read(cache_key, db_fn=lambda: manager.get_block_by_id(..., use_cache=False))
       │
       ├─ Step 1: await acache_get_hash(key)
       │    └─ _acache_dispatch("get_hash", key)
       │         └─ hasattr(RedisCacheProvider, "aget_hash") → False
       │         └─ asyncio.to_thread(provider.get_hash, key)   ← sync Redis in thread
       │
       ├─ Cache HIT → deserialize and return
       │
       ├─ Cache MISS:
       │    └─ Step 2: asyncio.to_thread(db_fn)
       │         └─ manager.get_block_by_id(..., use_cache=False)
       │              └─ cache_provider = None (skipped)
       │              └─ PostgreSQL only
       │
       └─ Step 3: await acache_set_hash(key, data, ttl)
            └─ asyncio.to_thread(provider.set_hash, key, data, ttl)
```

### 2.6 Summary (Redis)

| Path        | What happens                                                                 |
|------------|-------------------------------------------------------------------------------|
| Sync       | Manager uses `get_cache_provider()` → `RedisCacheProvider`; sync get/set to Redis; DB on miss and for write-back. |
| Async REST | Handler uses `async_cache_read` → `acache_get_hash` / `acache_set_hash` → `_acache_dispatch` → `asyncio.to_thread(sync method)` so Redis runs in a thread. |

---

## 3. Async Cache Provider

An async cache provider implements the **optional** async methods (e.g. `aget_hash`, `aset_hash`, `aget_json`, `aset_json`). The same dispatch logic is used; no extra code paths in managers or REST handlers.

### 3.1 Registration

Another service (e.g. ECMS) or startup code creates a provider that implements both sync and async interfaces, then registers it:

```python
from mirix.database.cache_provider import register_cache_provider

async_provider = MyAsyncCacheProvider(config)   # has get_hash, set_hash, aget_hash, aset_hash, ...
register_cache_provider("ips_cache", async_provider)
```

The last registered provider is active, so `get_cache_provider()` returns `MyAsyncCacheProvider`.

### 3.2 Provider interface (async part)

The provider still implements the sync methods (for sync callers). In addition it implements the async variants:

- `aget` / `aset` / `adelete`
- `aget_hash` / `aset_hash`
- `aget_json` / `aset_json`

So the same keys and semantics as Redis; the difference is that async callers can use the `a*` methods.

### 3.3 Sync path — managers

Unchanged. Sync code still calls `get_cache_provider()` and then the **sync** methods:

```python
cache_provider = get_cache_provider()   # → MyAsyncCacheProvider
cached_data = cache_provider.get_hash(key)   # Sync method (e.g. over HTTP or local impl)
```

So managers do not need to know whether the provider is sync or async; they always use the sync interface.

### 3.4 Async path — _acache_dispatch uses async methods

Async REST handlers still go through `async_cache_read` → `acache_get_hash` / `acache_set_hash` → `_acache_dispatch`. Now the provider has `aget_hash`:

```python
async_method = f"a{method}"   # "aget_hash"
if hasattr(provider, async_method):   # True
    return await getattr(provider, async_method)(*args, **kwargs)   # Direct await, no thread
```

So **no** `asyncio.to_thread` is used for cache I/O; the event loop calls the provider’s async implementation directly (e.g. async HTTP to IPS Cache or native async client).

### 3.5 Full async read path (e.g. GET /blocks/{id})

With an async provider:

```
REST handler (async)
  └─ async_cache_read(cache_key, db_fn=lambda: manager.get_block_by_id(..., use_cache=False))
       │
       ├─ Step 1: await acache_get_hash(key)
       │    └─ _acache_dispatch("get_hash", key)
       │         └─ hasattr(provider, "aget_hash") → True
       │         └─ await provider.aget_hash(key)   ← zero threads
       │
       ├─ Cache HIT → return
       │
       ├─ Cache MISS:
       │    └─ Step 2: asyncio.to_thread(db_fn)   ← DB still sync (SQLAlchemy)
       │
       └─ Step 3: await acache_set_hash(key, data, ttl)
            └─ await provider.aset_hash(key, data, ttl)   ← zero threads
```

Only the DB fallback runs in a thread; all cache operations are awaited natively.

### 3.6 Summary (async provider)

| Path        | What happens                                                                 |
|------------|-------------------------------------------------------------------------------|
| Sync       | Same as Redis: manager uses sync `get_hash` / `set_hash` on the provider.   |
| Async REST | `_acache_dispatch` sees `aget_hash` / `aset_hash` and calls them with `await`; no thread pool for cache I/O. |

---

## Comparison

| Scenario        | get_cache_provider() | Sync manager path        | Async cache path (REST)                    |
|----------------|----------------------|--------------------------|-------------------------------------------|
| No provider    | `None`               | DB only                  | Handler skips cache; to_thread(manager) → DB only |
| Redis          | `RedisCacheProvider` | Sync Redis get/set + DB | to_thread(provider.get_hash/set_hash) + to_thread(db_fn) |
| Async provider | e.g. `MyAsyncCacheProvider` | Sync get/set on provider + DB | await provider.aget_hash/aset_hash + to_thread(db_fn) |

The same REST and manager code supports all three cases; only the registered provider and whether it implements async methods change the behavior.
