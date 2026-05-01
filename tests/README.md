# Mirix Tests

## Setup

```bash
# Install dependencies
pip install -e .

# Set API key
export GEMINI_API_KEY=your_api_key_here
# Or on Windows:
# set GEMINI_API_KEY=your_api_key_here
```

## Test Files Overview

The `tests/` directory contains ~45 test modules. Only integration tests require a running server; everything else runs against an in-process `AsyncServer()` or directly exercises ORM/managers.

**Foundational unit tests:**
- `test_memory_server.py` — in-process server, all 5 memory types + search methods
- `test_memory_integration.py` — REST API via real server + client
- `test_user_manager.py`, `test_user.py`, `test_raw_memory.py`, `test_message_handling.py`, `test_queue.py`, `test_local_client.py`, `test_langfuse_integration.py`, `test_cache_provider.py`, `test_redis_cache_provider.py`, `test_redis_integration.py`

**VEPAGE-760 (memory sources, citations, idempotency, temporal guard):**
- `test_memory_source_managers.py`, `test_memory_source_schemas.py` — source sidecar tables
- `test_citation_writes.py`, `test_citation_writes_db.py`, `test_search_citations.py`, `test_search_citations_integration.py` — citation emission + search integration
- `test_source_idempotency.py`, `test_source_idempotency_db.py` — L1 source-level dedup
- `test_processing_skip.py` — L2 processing-complete short-circuit
- `test_idempotency_skip_spans.py` — LangFuse skip-span emission for L1/L2/L3
- `test_temporal_guard.py` — MAX(occurred_at) guard for out-of-order writes
- `test_summary_generation.py` — summary-agent task dispatch + span
- `test_retrieval_endpoints.py` — `GET /memory-sources/{id}` + `/messages`
- `test_add_memory_direct_writes.py`, `test_direct_write_episodic_handler.py`, `test_direct_write_meta_agent.py`, `test_direct_writes_queue.py`, `test_direct_writes_worker.py` — caller-authored memory writes bypassing the LLM pipeline

**Access control / filtering:**
- `test_multi_scope_access.py`, `test_scoped_blocks.py`, `test_search_all_users.py`, `test_search_single_user_core_memory.py`, `test_client_agent_isolation.py`
- `test_block_filter_tag_updates.py`, `test_block_filter_tags_update_mode.py`, `test_filter_tags_db.py`, `test_filter_tags_query.py`, `test_filter_function_args.py`, `test_remote_client_block_filter_tags.py`

**Misc:** `test_agent_prompt_update.py`, `test_auth_provider.py`, `test_deletion_apis.py`, `test_memory_agent_toolcall_truncation.py`, `test_memory_tool_inserts.py`, `test_orm_to_pydantic_safe.py`, `test_raw_memory_with_real_embeddings.py`, `test_temporal_queries.py`.

## Run Tests

```bash
# Run all tests (server + integration)
pytest -v

# Server-side tests only (fast, no real server needed)
pytest tests/test_memory_server.py -v

# Integration tests only (requires manually started server - see below)
pytest tests/test_memory_integration.py -v -m integration -s

# Skip integration tests (runs server tests only)
pytest -m "not integration" -v
```

## Test Coverage

### Server Tests (`test_memory_server.py`)
Comprehensive coverage of all 5 memory types with all search methods:
- ✅ **Episodic Memory**: Insert events, search by summary/details (bm25, embedding)
- ✅ **Procedural Memory**: Insert procedures, search by summary/steps (bm25, embedding)
- ✅ **Resource Memory**: Insert resources, search by summary (bm25, embedding) / content (bm25 only)
- ✅ **Knowledge Vault**: Insert knowledge, search by caption (bm25, embedding) / secret_value (bm25)
- ✅ **Semantic Memory**: Insert items, search by name/summary/details (bm25, embedding)
- ✅ **Cross-memory search**: Search across all memory types

### Integration Tests (`test_memory_integration.py`)
Core API operations via client-server:
- ✅ `client.add()`: Add memories via conversation
- ✅ `client.retrieve_with_conversation()`: Retrieve with context
- ✅ `client.retrieve_with_topic()`: Retrieve by topic
- ✅ `client.search()`: Search memories (bm25, embedding)
- ✅ `include_citations=True` (covered in `test_search_citations_integration.py`)
- ✅ `GET /memory-sources/{id}` + `/messages` (covered in `test_retrieval_endpoints.py`)

### VEPAGE-760 coverage highlights
- Idempotency (L1/L2/L3) — `test_source_idempotency*`, `test_processing_skip`, `test_idempotency_skip_spans`
- Temporal guard — `test_temporal_guard`
- Direct writes — `test_add_memory_direct_writes`, `test_direct_writes_*`
- Summary agent — `test_summary_generation`
- Citation emission + search — `test_citation_writes*`, `test_search_citations*`

## Prerequisites

**API Key Required**: Set `GEMINI_API_KEY` environment variable:

```bash
export GEMINI_API_KEY=your_api_key_here
# Or on Windows:
set GEMINI_API_KEY=your_api_key_here
```

**Automatic Initialization**: Both test files will automatically:
- Create `demo-user` in `demo-org` organization
- Initialize meta agent and all sub-agents (episodic, procedural, resource, knowledge vault, semantic)
- No manual setup needed!

## Running Integration Tests

Integration tests require a **manually started server** on port 8899:

```bash
# Terminal 1: Start server
python scripts/start_server.py --port 8899

# Terminal 2: Run integration tests (will auto-initialize on first run)
pytest tests/test_memory_integration.py -v -m integration -s
```

**Note**: Server tests (`test_memory_server.py`) don't need a running server.

## Common Options

```bash
# Show print statements
pytest -v -s

# Run specific test
pytest tests/test_memory_server.py::TestDirectEpisodicMemory::test_insert_event -v

# With coverage
pytest --cov=mirix --cov-report=html

# Debug on failure
pytest --pdb
```
