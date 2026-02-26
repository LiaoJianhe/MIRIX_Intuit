# How Cache Providers Work

This document describes how Mirix behaves under **no cache provider**, **Redis** (sync + async), a **sync-only provider**, and an **async cache provider** (e.g. a future IPS Cache implementation). The code paths are the same; only the presence and type of the registered provider change.

**Dependency order:** Cache provider logic (including async dispatch in PR 55) applies **on top of** the entity and access model from PR 52 (1 user per org), PR 53 (multi-scope clients), and PR 54 (scoped core memory). Cached entity shapes (User, Client, Block, Agent, Message, etc.) and key prefixes are defined by those PRs; PR 55 only adds async I/O paths (`acache_*`, `async_cache_read`) that use the same keys and data shapes.

**Sync bridge and search provider:** The **sync bridge** (`mirix.database.sync_bridge`) holds the application event loop and thread so sync callers (ORM, queue worker) can run async cache code via `run_coroutine_threadsafe` without depending on Redis. Call `set_event_loop_for_sync_bridge()` from async startup and `clear_sync_bridge()` on shutdown. **Search provider** (`mirix.database.search_provider`) is a separate registry for vector/text/recency search; when Redis is enabled it registers both a cache provider and a search provider. Managers use `get_search_provider()` for list/search and fall back to DB when no provider is registered. The cache provider interface also includes **extended methods**: `delete_many`, `update_hash_field`, `set_string`, `get_string`, `delete_string` (and async `acache_*` variants) for batch deletes, soft deletes, and reverse-key mappings.

---

## 1. No Cache Provider

When no provider is registered, `get_cache_provider()` returns `None`. The system always falls back to PostgreSQL.

### 1.1 Registration

Nothing is registered. Either Redis is disabled (`redis_enabled=False`), Redis failed to connect, or no other provider was registered. The globals `_active_provider_name` and `_active_provider` stay `None`, and `get_cache_provider()` returns `None` via a single lock-free read:

```python
# cache_provider.py
def get_cache_provider() -> Optional[Any]:
    return _active_provider  # None when no provider registered
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

Mirix supports two Redis registration patterns:

- **Async path (default):** `RedisAsyncCacheProvider` wraps `RedisMemoryClient` (redis.asyncio.Redis). Registered with `async_only=True`. Sync callers use `sync_cache_*`, which route through the sync bridge; async callers use `acache_*` and call the provider's async methods directly.
- **Sync path (optional):** `RedisSyncCacheProvider` wraps `RedisSyncMemoryClient` (sync redis.Redis). Registered with `async_only=False`. Sync callers call provider sync methods directly (no bridge); async callers use `asyncio.to_thread(sync_method)`.

### 2.1 Registration at startup (async path — default)

On server startup, `initialize_redis_client()` runs. Inside `redis_client.py`, after the connection and indexes are created, Redis **auto-registers** as the cache provider with **async_only=True**:

```python
# redis_client.py (inside initialize_redis_client())
from mirix.database.cache_provider import register_cache_provider
from mirix.database.redis_cache_provider import RedisAsyncCacheProvider
from mirix.database.redis_search_provider import RedisSearchProvider
from mirix.database.search_provider import register_search_provider
from mirix.database.sync_bridge import set_event_loop_for_sync_bridge

set_event_loop_for_sync_bridge()
redis_cache_provider = RedisAsyncCacheProvider(_redis_client)
register_cache_provider("redis", redis_cache_provider, async_only=True)
redis_search_provider = RedisSearchProvider(_redis_client)
register_search_provider("redis", redis_search_provider)
```

`register_cache_provider("redis", redis_cache_provider, async_only=True)` stores the provider and sets `_active_provider_async_only = True`. Sync callers use the **sync_cache_*** helpers (e.g. `sync_cache_get_hash`), which run the corresponding `acache_*` coroutine via the sync bridge. Async callers use `acache_*` and hit the provider's async methods directly. `RedisCacheProvider` is a backward-compat alias for `RedisAsyncCacheProvider`.

### 2.2 The adapters: RedisAsyncCacheProvider and RedisSyncCacheProvider

- **RedisAsyncCacheProvider** wraps `RedisMemoryClient` (async). It implements **only** async methods; sync callers go through `sync_cache_*` → sync bridge → `acache_*`.
- **RedisSyncCacheProvider** wraps `RedisSyncMemoryClient` (sync redis.Redis). It implements **only** sync methods; async callers use `acache_*` which fall back to `asyncio.to_thread(sync_method)`. To use the sync path: `initialize_redis_sync_client()` then `register_cache_provider("redis", RedisSyncCacheProvider(sync_client))`.


### 2.3 Sync path — managers (async provider)

When the **async** Redis provider is registered (`async_only=True`), sync code uses the dispatcher helpers: `sync_cache_get_hash(key)` etc. These run `acache_get_hash` via the sync bridge. When the **sync** Redis provider is registered (`async_only=False`), the dispatcher calls `provider.get_hash(key)` directly (no bridge).

### 2.4 Async path — REST handlers and _acache_dispatch

Async handlers use `async_cache_read()`, which uses `acache_get_hash` / `acache_set_hash`. Those call `_acache_dispatch()`:

```python
# cache_provider.py
async def _acache_dispatch(method: str, *args, **kwargs) -> Any:
    provider = get_cache_provider()   # → RedisCacheProvider
    if provider is None:
        return None
    async_method = f"a{method}"   # e.g. "aget_hash"
    if hasattr(provider, async_method):   # True for RedisCacheProvider
        return await getattr(provider, async_method)(*args, **kwargs)
    return await asyncio.to_thread(
        getattr(provider, method), *args, **kwargs
    )
```

Because `RedisCacheProvider` **does** define `aget_hash`, `aset_hash`, etc., async cache calls **await the provider’s async methods directly** (no `to_thread`). Redis I/O is non-blocking on the event loop.

### 2.5 Full async read path (e.g. GET /blocks/{id})

End-to-end for a REST read when Redis is the provider:

```
REST handler (async)
  └─ cache_provider = get_cache_provider()  → RedisCacheProvider
  └─ async_cache_read(cache_key, db_fn=lambda: manager.get_block_by_id(..., use_cache=False))
       │
       ├─ Step 1: await acache_get_hash(key)
       │    └─ _acache_dispatch("get_hash", key)
       │         └─ hasattr(RedisCacheProvider, "aget_hash") → True
       │         └─ await provider.aget_hash(key)   ← async Redis, no thread
       │
       ├─ Cache HIT → deserialize and return
       │
       ├─ Cache MISS:
       │    └─ Step 2: asyncio.to_thread(db_fn)
       │         └─ manager.get_block_by_id(..., use_cache=False)
       │              └─ PostgreSQL only (sync DB in thread)
       │
       └─ Step 3: await acache_set_hash(key, data, ttl)
            └─ await provider.aset_hash(key, data, ttl)   ← async Redis, no thread
```

### 2.6 Summary (Redis)

| Path        | What happens                                                                 |
|------------|-------------------------------------------------------------------------------|
| Sync (async provider) | Sync callers use `sync_cache_get_hash` etc. → dispatcher runs `acache_*` via sync bridge (run_coroutine_threadsafe). Requires `set_event_loop_for_sync_bridge()` from async startup. |
| Sync (sync provider)  | Sync callers use `sync_cache_get_hash` etc. → dispatcher calls `provider.get_hash` directly (no bridge). |
| Async REST | Handler uses `async_cache_read` → `acache_get_hash` / `acache_set_hash` → `_acache_dispatch` → `await provider.aget_hash` / `aset_hash` (direct async Redis, no thread). With sync provider, `_acache_dispatch` falls back to `asyncio.to_thread(provider.get_hash)`. |

---

## 3. Async Cache Provider

An async cache provider implements the **optional** async methods (e.g. `aget_hash`, `aset_hash`, `aget_json`, `aset_json`). The same dispatch logic is used; no extra code paths in managers or REST handlers.

### 3.1 Registration

Another service or startup code creates a provider that implements both sync and async interfaces, then registers it:

```python
from mirix.database.cache_provider import register_cache_provider

async_provider = MyAsyncCacheProvider(config)   # has get_hash, set_hash, aget_hash, aset_hash, ...
register_cache_provider("my_cache", async_provider)
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

## 4. Search provider (separate from cache)

The **search provider** is a separate registry from the cache provider. It is used for **vector similarity**, **full-text**, and **recency-sorted** search over memory and entity indexes. When Redis is enabled, the server registers both a cache provider (key-value read/write) and a search provider (query over indexes); when no search provider is registered, managers fall back to PostgreSQL for list/search.

### 4.1 What the search provider does

- **Cache provider** (`get_cache_provider()`): key-value operations — get/set/delete by key (e.g. `block:<id>`, `user:<id>`). Used for cache-aside reads and cache invalidation.
- **Search provider** (`get_search_provider()`): query operations — search by recency, vector embedding, or text (BM25/string match). Used by memory managers (episodic, semantic, procedural, resource, knowledge vault) and any code that needs to list or search over many items.

Managers that support search (e.g. `episodic_memory_manager.search_episodic_memory`) call `get_search_provider()`. If a provider is present and `use_cache=True`, they call its async search methods first; on miss or error they fall back to PostgreSQL.

### 4.2 Registry and usage

Registry lives in `mirix.database.search_provider`:

- **register_search_provider(name, provider)** — register a provider; the last registered one is active.
- **get_search_provider()** — returns the active provider instance or `None`.
- **unregister_search_provider(name)** — remove a provider (e.g. on shutdown).

When Redis is used, both providers are registered in `redis_client.py` after the client and indexes are created:

```python
# redis_client.py (inside initialize_redis_client())
redis_cache_provider = RedisAsyncCacheProvider(_redis_client)
register_cache_provider("redis", redis_cache_provider, async_only=True)
redis_search_provider = RedisSearchProvider(_redis_client)
register_search_provider("redis", redis_search_provider)
```

With Redis, the **same backing store** is used for both: the cache provider writes keys (e.g. `episodic:<id>`), and Redis indexes those keys; the search provider runs queries against those indexes. So no separate “indexing” step is needed for Redis — writing via the cache populates the search indexes.

### 4.3 Search provider interface (duck typing)

A search provider implements the following **async** methods and a static helper. No base class is required.

| Method | Purpose |
|--------|--------|
| `search_recent(index_name, limit, user_id, organization_id, sort_by, return_fields, filter_tags, start_date, end_date)` | Recency-sorted list (e.g. by `created_at_ts` or `occurred_at_ts`). |
| `search_vector(index_name, embedding, vector_field, limit, ...)` | Vector similarity search (e.g. KNN on embedding). |
| `search_text(index_name, query, search_fields, limit, ...)` | Full-text search (BM25/string match). |
| `search_recent_by_org(index_name, limit, organization_id, ...)` | Recency by organization (admin/list views). |
| `search_vector_by_org(index_name, embedding, vector_field, limit, organization_id, ...)` | Vector search by org. |
| `search_text_by_org(index_name, query_text, search_field, search_method, limit, organization_id, ...)` | Text search by org. |
| `clean_search_fields(items: List[Dict]) -> List[Dict]` | Strip backend-specific fields so results can be passed to Pydantic models. |

The provider must also expose **index name constants** so callers can pass the correct index (e.g. `search_provider.EPISODIC_INDEX`). For Redis these are: `BLOCK_INDEX`, `MESSAGE_INDEX`, `EPISODIC_INDEX`, `SEMANTIC_INDEX`, `PROCEDURAL_INDEX`, `RESOURCE_INDEX`, `KNOWLEDGE_INDEX`, `ORGANIZATION_INDEX`, `USER_INDEX`, `AGENT_INDEX`, `TOOL_INDEX`. See `mirix.database.redis_search_provider.RedisSearchProvider` for the exact signatures and default index names.

### 4.4 How to register and use a new search provider

1. **Implement the interface** in your own class (duck typing):
   - Implement all of the async methods above with the same semantics (filters, limits, return shape).
   - Implement `clean_search_fields(items)` so that returned dicts match what the managers expect (e.g. strip `id`/internal fields added by your backend).
   - Define the same index name constants (or a subset) that managers use (e.g. `EPISODIC_INDEX`, `SEMANTIC_INDEX`, …).

2. **Ensure the search backend is populated.** With Redis, cache writes (via the cache provider) automatically populate the indexes. For another backend (e.g. Elasticsearch, IPS Search):
   - Either index documents when your app writes to the cache or DB (e.g. in the same code paths that today write to Redis), or
   - Run a separate sync job from DB/cache into your search store. Managers do not call the search provider to index; they only call it to search.

3. **Register at startup** (after your backend/client is ready):

   ```python
   from mirix.database.search_provider import register_search_provider

   my_search_provider = MySearchProvider(config)  # your client/index setup
   register_search_provider("my_search", my_search_provider)
   ```

4. **Unregister on shutdown** (optional but recommended):

   ```python
   from mirix.database.search_provider import unregister_search_provider
   unregister_search_provider("my_search")
   ```

No changes are required in managers or REST handlers: they already call `get_search_provider()` and, if non-`None`, use the active provider’s search methods and index constants. Registering a new provider and making it the last registered one is enough for it to be used.

---

## Cache content and PR 52 / 53 / 54

The **data** stored in the cache (entity fields, key prefixes, and serialization) comes from the models and managers updated in:

- **PR 52 (1 user per org):** User and Message (and related) are organization- and user-scoped; cache entries use `organization_id`, `user_id`, `created_by_id` where applicable. Agent cache does not pass `memory` or `children_ids` into `AgentState` (schema has no such fields).
- **PR 53 (multi-scope clients):** Client cache includes `read_scopes` (list) and `write_scope`; manager deserializes `read_scopes` from string when reading from cache.
- **PR 54 (scoped core memory):** Block cache includes `filter_tags` (scope); managers and agent reconstruction deserialize `filter_tags` from string when reading from cache.

**PR 55 (Async Cache Provider)** does not change these shapes or keys. It only adds:

- Optional async methods on the provider (`aget_hash`, `aset_hash`, etc.).
- `_acache_dispatch` and `acache_*` in `cache_provider.py` to call async methods when present, else `asyncio.to_thread(sync_method)`.
- `async_cache_read` in `cache_layer.py` for REST handlers to do cache-aside with async I/O when the provider supports it.

Same cache keys, same payloads; only the I/O path (async vs sync) changes.

---

## Comparison

| Scenario        | get_cache_provider() | Sync manager path        | Async cache path (REST)                    |
|----------------|----------------------|--------------------------|-------------------------------------------|
| No provider    | `None`               | DB only                  | Handler skips cache; to_thread(manager) → DB only |
| Redis          | `RedisCacheProvider` | Sync Redis get/set + DB | await provider.aget_hash/aset_hash + to_thread(db_fn) |
| Sync-only      | e.g. `MySyncCacheProvider` | Sync get/set on provider + DB | to_thread(provider.get_hash/set_hash) + to_thread(db_fn) |
| Async provider | e.g. `MyAsyncCacheProvider` | Sync get/set on provider + DB | await provider.aget_hash/aset_hash + to_thread(db_fn) |

The same REST and manager code supports all cases; only the registered provider and whether it implements async methods change the behavior. A **sync-only** provider (sync methods only, no `aget_*`/`aset_*`) is supported: async callers use `_acache_dispatch`, which falls back to `asyncio.to_thread(sync_method)` when the async variant is missing.

### Option 2: Async-only cache provider (sync_cache_* API)

To support an **async-only** cache provider (one that implements only `aget_*`/`aset_*` and no sync `get_*`/`set_*`), register it with **`async_only=True`**:

```python
register_cache_provider("ips_cache", async_only_provider, async_only=True)
```

When `async_only=True`, all sync callers (managers, ORM lifecycle) use the **sync cache API** (`sync_cache_get_hash`, `sync_cache_set_hash`, `sync_cache_get_json`, `sync_cache_set_json`, `sync_cache_delete`, `sync_cache_delete_many`, `sync_cache_update_hash_field`, `sync_cache_set_string`, `sync_cache_get_string`, `sync_cache_delete_string`). These helpers run the corresponding `acache_*` coroutine via the sync bridge (`run_coroutine_threadsafe` on the stored event loop). The provider does not need to implement any sync methods.

**Requirement:** When using `async_only=True`, the application must call **`set_event_loop_for_sync_bridge()`** from async context (e.g. in REST server startup or Redis init) before any sync code runs cache operations. Otherwise sync_cache_* will return None/False/0 when the loop is missing.
