# MIRIX - Claude Code Instructions

## What This Project Does
MIRIX is an async-native multi-agent personal assistant with a six-component memory system
(Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault). It captures memories
from conversations, storing them in PostgreSQL with vector search via pgvector. Screen
activity capture exists in OSS MIRIX but is **not used by the ECMS integration** (this
fork's primary consumer).

## This fork's relationship to ECMS
ECMS (`context-and-memory-service`) imports MIRIX as a Python library — it calls
`mirix.server.rest_api` functions **directly** and does not start MIRIX's FastAPI app.
ECMS configures MIRIX to run with `CHAINING_FOR_MEMORY_UPDATE=false` and
`CHAINING_FOR_META_AGENT=false` (each agent runs exactly once) and registers Intuit
auth/cache providers at startup.

## Key Architecture
- **Entry point**: `mirix/server/rest_api.py` (FastAPI, port 8531 when run standalone)
- **Orchestration**: `mirix/agent/meta_agent.py` → 6 sub-agents. A **Summary Agent** task
  is dispatched in parallel (via `asyncio.create_task`) under a LangFuse "Summary Agent"
  span — it is not a registered `AgentType`.
- **Data flow**: ORM (`mirix/orm/`) → Schemas (`mirix/schemas/`) → Managers (`mirix/services/`) → API (`mirix/server/`)
- **Queue**: Kafka-backed (`mirix/queue/`) for async memory extraction
- **Provenance sidecar** (VEPAGE-760): three tables — `memory_source`, `memory_citation`,
  `source_message` — written by the meta-agent pipeline via `_persist_memory_source` and
  `_write_citation`. Never written by direct DB access; always go through these paths so
  citations stay consistent.
- **Idempotency** (3 layers): (L1) DB uniqueness on `source_messages`; (L2)
  `processing_complete` flag on `memory_source` short-circuits re-processing; (L3)
  citation-level dedup inside `_write_citation`. Skip events emit a LangFuse span via
  `mirix/observability/skip_spans.py`.
- **Temporal guard**: `MAX(occurred_at)` per `(memory_type, memory_id)` across all
  citations — out-of-order writes are dropped, not applied. Scope is per-memory, not
  per-thread.
- **Retrieval endpoints**: `GET /memory-sources/{id}` (metadata + summary) and
  `GET /memory-sources/{id}/messages` (paginated raw turns). Search accepts
  `include_citations=True` and attaches `citations: [{memory_source_id, ...}]` to each
  derived memory.
- **`AddMemoryRequest` provenance fields**: `external_id`, `external_thread_id`,
  `source_type`, `source_system`, `source_metadata`, `summary`, `summarize`,
  `batch_hash`, `filter_tags`, `memory_source_id`, plus `direct_writes=[{memory_type,
  payload}]` for caller-authored memories bypassing the LLM pipeline.
- **All I/O is async** — never introduce sync blocking calls (see `docs/Mirix_async_native_changes.md`)

## Local Development Setup

### Prerequisites
- Docker + Docker Compose
- Python 3.10+
- At least one LLM API key (OpenAI, Anthropic, or Google Gemini)

### Start Infrastructure
```bash
cp docker/env.example .env   # then add your API keys to .env
docker-compose up -d         # starts PostgreSQL (5432), Redis (6379), API (8531), Dashboard (5173)
docker-compose ps            # verify all services healthy
```

### Run API Locally (without Docker)
```bash
pip install -e .
export GEMINI_API_KEY=your-key   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
python scripts/start_server.py --port 8531
```

### Access Points
- Dashboard: http://localhost:5173
- API Swagger: http://localhost:8531/docs
- API ReDoc: http://localhost:8531/redoc

## Running Tests

The preferred way to run tests is via the dockerized test script, which handles infrastructure automatically:

```bash
# Full suite with verbose output (preferred)
./scripts/run_tests_with_docker.sh --podman -s -v --log-cli-level=INFO

# Pass any pytest args after the flags
./scripts/run_tests_with_docker.sh --podman -s -v --log-cli-level=INFO -k test_message_handling
./scripts/run_tests_with_docker.sh --podman -s -v --log-cli-level=INFO -m "not integration"
```

**Required env var for tests**: `GEMINI_API_KEY`

### Running without Docker (manual infra)
```bash
# Fast unit tests — no running server needed (~20s)
pytest tests/test_memory_server.py -v

# All tests except integration
pytest -m "not integration" -v

# Integration tests — requires server on port 8899
python scripts/start_server.py --port 8899          # Terminal 1
pytest tests/test_memory_integration.py -v -m integration -s   # Terminal 2
```

## Common Dev Tasks

### Add a new API endpoint
1. Add Pydantic request/response schemas to `mirix/schemas/`
2. Add business logic method to the relevant manager in `mirix/services/`
3. Add the route to `mirix/server/rest_api.py`
4. Add the corresponding method to `mirix/client/remote_client.py`

### Add a new memory type
1. Create ORM model in `mirix/orm/`
2. Create Pydantic schemas in `mirix/schemas/`
3. Create manager in `mirix/services/`
4. Create sub-agent in `mirix/agent/`
5. Register in `mirix/agent/meta_agent.py`
6. Emit citations: any new write path for the memory type must call
   `_write_citation(...)` so the `memory_citations` row is created with the
   originating `memory_source_id` / `source_message_id` (and L3 dedup applies).
   Direct DB inserts bypass provenance and break progressive disclosure.

### Format & lint
```bash
# Preferred (poetry)
poetry run black . && poetry run isort .

# Alternatively via make
make format   # ruff import sort + format
make lint     # ruff check + pyright
make check    # format + lint + test
```

## Async Rules (Critical)
- **All new manager methods must be `async def`**
- **Never use `asyncio.run()` inside the server** — the event loop is already running
- Only 5 intentional sync touch-points exist (LangFuse, Gmail OAuth, SQLAlchemy DDL at startup, cleanup job entry, pure CPU helpers) — do not add more
- Use `asyncio.to_thread()` to wrap any unavoidably sync third-party calls

## Commit Convention
Prefix commits with the Jira ticket: `[VEPAGE-NNN] Description`

## Do Not
- Confuse queue messages (transient) with database messages (persistent)
- Call `step_manager.get_step()` — steps are write-only audit logs
- Skip `create_or_get_user()` — always ensure users exist first
- Write memories via direct DB access — use the memory-tool functions or
  `_apply_direct_write` so citations get emitted
- Re-enable agent chaining when running under ECMS — ECMS configures chaining off
  and expects every agent to run exactly once
- Commit secrets or API keys
- Create README or doc files unless explicitly requested
