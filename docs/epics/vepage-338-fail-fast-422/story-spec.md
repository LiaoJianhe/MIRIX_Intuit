# VEPAGE-338 — Fail fast on LLM 422 (Risk Screening)

**Ticket:** [VEPAGE-338](https://jira.cloud.intuit.com/browse/VEPAGE-338) — "422 from SRF cripples the system"
**Status:** Design
**Author:** Lucas Parzych

## Problem

When OpenAI's Risk Screening Filter (or any upstream wrapper) returns **422 Unprocessable Entity** — e.g., "Toxicity detected", "Suspicious language detected" — MIRIX treats the failure as transient and retries it. Because 422 is content-driven, not a server problem, the retries never succeed. Two compounding effects:

1. **In-agent retry storm.** `_get_ai_reply` (agent.py:806) catches every `LLMError` and retries up to `MIRIX_LLM_RETRY_LIMIT=3` with exponential backoff, then falls back to a `second_try` re-attempt. Across ~7 sub-agents per Meta Agent step, one toxic input burns 20+ failed LLM calls and 25-90 minutes of wall time per attempt.
2. **Kafka rebalance redelivery.** The consumer is configured with `enable_auto_commit=True` and `max_poll_interval_ms=900000` (15 min). When a single message takes longer than 15 min to process, the consumer is evicted from the group, the offset is never committed, the message is redelivered, and processing starts over. This continues indefinitely.

The observed trace (`9e211b130acc4b8fbc6d036ba2c3c430`) shows one toxic input driving **85 separate Meta Agent runs over ~25 hours** and ~7,000 LLM error observations, all on the same payload.

The ticket also flags a secondary concern: that the toxic input lands in the agent's `messages` history, poisoning future writes for the same user. Verification confirmed retention write-back only happens on the success path (agent.py:1577-1590), so a fail-fast abort skips that step. The fix below preserves that property.

## Goals

- 422 from any LLM provider does **not** trigger retries. Fail in seconds, not minutes.
- The message in flight is **marked `processing_complete=True`** on the `memory_sources` row, so any Kafka redelivery short-circuits via the existing L2 idempotency check (agent.py:1424).
- 424 (Failed Dependency / dependency timeout) keeps its current retry behavior — these are genuinely transient.
- No change to the caller-facing API contract today (`POST /v1/memories` is async via Kafka — the caller has already received `{"status":"completed"}` long before the worker runs). A proper rejection contract on `GET /v1/memory-sources/{id}` is a separate future change.
- No new schema changes (no new columns).

## Non-Goals

- Adding a `rejection_reason` / `status` field to `memory_sources`. Useful, but out of scope for this fix — flagged as a follow-up.
- Surfacing a 4xx response back to the caller. The save path is queue-backed; the caller has already gotten a 200.
- Changing Kafka commit semantics, `max.poll.interval.ms`, or any consumer-level retry policy.
- Touching the 422 mapping itself (`LLMUnprocessableEntityError` already exists and is correctly produced by OpenAI/Anthropic clients).

## Approach

Two-site change ("B+A"):

### B — Fast-fail 422 inside `_get_ai_reply` (agent.py:806)

In the `except LLMError as llm_error:` block, before the retry/backoff logic runs, add an early-raise for `LLMUnprocessableEntityError`:

```python
except LLMError as llm_error:
    # 422 is content-driven and not retryable. Propagate immediately so
    # callers can mark the source complete and let the Kafka offset commit.
    if isinstance(llm_error, LLMUnprocessableEntityError):
        raise
    # ... existing retry / second_try logic for other LLMError subclasses
```

This eliminates the in-agent retry storm: a 422 propagates from the OpenAI client in roughly the time of one HTTP round-trip. 424 (`LLMServerError` with `DEPENDENCY_TIMEOUT`) is not `LLMUnprocessableEntityError`, so it continues to retry — desired.

The `second_try` fallback is also bypassed for 422, which is correct: re-running the LLM with only the last message won't make toxic content pass screening.

### A — Mark source complete on 422 in the meta-agent epilogue (agent.py:1602)

Wrap the meta-agent's outer sub-agent dispatch in a try/except for `LLMUnprocessableEntityError`. The cleanest spot is around the main step loop (the `while True` ending at line 1575) and the summary task await:

```python
try:
    # ... existing main loop + retention write-back + summary await
except LLMUnprocessableEntityError as e:
    logger.warning(
        "Source %s rejected by LLM provider (422); marking processing complete to suppress retries. error=%s",
        self.memory_source_id, e,
    )
    emit_idempotency_skip_span(
        name="Source Rejected: LLM 422",
        reason="llm-422-content-rejected",
        metadata={"memory_source_id": self.memory_source_id, "error": str(e)[:500]},
    )
    if self.agent_state.is_type(AgentType.meta_memory_agent) and self.memory_source_id:
        try:
            await self.memory_source_manager.mark_processing_complete(self.memory_source_id)
        except Exception as mark_err:
            logger.warning("Failed to mark source %s complete after 422: %s", self.memory_source_id, mark_err)
            raise
    return MirixUsageStatistics(step_count=0)
```

Behavior after the fix:

1. Toxic input arrives at the worker.
2. Meta-agent persists the `memory_source` row and the `source_messages` (existing behavior, line 1405).
3. First sub-agent calls OpenAI → 422.
4. `_get_ai_reply` propagates the `LLMUnprocessableEntityError` immediately (no retries).
5. Meta-agent's new `except` block logs, emits a skip span, calls `mark_processing_complete`, and returns clean `MirixUsageStatistics(step_count=0)`.
6. Worker's `_process_message_async` sees a successful return. Kafka commits the offset on the next `poll()`.
7. If any redelivery happens before commit, the L2 idempotency check at agent.py:1424 short-circuits with `processing-complete`.

## Components touched

- `mirix/agent/agent.py`
  - `_get_ai_reply`: one branch added in the `except LLMError` block.
  - `step()`: wrap the main step loop + retention write-back + summary-task await in the new try/except for `LLMUnprocessableEntityError`. The existing `mark_processing_complete` call at line 1602-1608 stays on the success path; the new except handles the 422 path.
- `mirix/errors.py`: no change — `LLMUnprocessableEntityError` already exists and is the right type.
- `mirix/llm_api/openai_client.py`, `mirix/llm_api/anthropic_client.py`: no change — they already map 422 to `LLMUnprocessableEntityError`.
- `mirix/observability/skip_spans.py`: reuse the existing `emit_idempotency_skip_span` helper with a new `reason` value (`"llm-422-content-rejected"`).
- No DB schema changes, no migrations.

## Edge cases

- **Summary agent 422.** The summary agent runs in parallel via `asyncio.create_task` (agent.py:1441) and is awaited after the main loop. A 422 there propagates from `await summary_task` (line 1597) inside the `try` we add — handled by the same `except`.
- **Sub-agent partial success.** If sub-agent A completes and writes a memory, then sub-agent B hits 422, agent A's memory + citation are already persisted. That's fine — the new `mark_processing_complete` simply prevents redelivery. Partial state is acceptable; full success is not a precondition for completion.
- **Direct-write path** (agent.py:1446-1449) does not go through the LLM, so it cannot raise 422. Existing behavior unchanged.
- **422 outside the meta-agent flow** (e.g., direct calls from non-queue code paths). Today no such path exists in production; if one is added later, it will see the immediate raise from `_get_ai_reply` and can handle it as appropriate.
- **Source not yet persisted.** If `_persist_memory_source` itself fails before the main loop, no `memory_source_id` row exists to mark — but that path doesn't raise 422 either. The new `except` checks `self.memory_source_id` before calling `mark_processing_complete`, so it's safe.
- **Retention / message history.** Retention write-back (agent.py:1578) runs *after* the main loop, only on success. A 422 abort skips it, so toxic input does not leak into the agent's `messages` history. This addresses the ticket's "next message will also fail" concern.

## Testing

- **Unit test (`_get_ai_reply` fast-fail):** Mock the LLM client to raise `LLMUnprocessableEntityError`. Assert one call, no retries, exception propagates. Confirm 424 (`LLMServerError(DEPENDENCY_TIMEOUT)`) still retries.
- **Unit test (meta-agent epilogue):** Patch sub-agent dispatch to raise `LLMUnprocessableEntityError`. Assert `mark_processing_complete` is called with the right `memory_source_id` and `MirixUsageStatistics(step_count=0)` is returned (no exception propagates).
- **Integration test:** End-to-end through the worker — enqueue a message with a forced 422 (mock the OpenAI client), assert the `memory_sources` row is `processing_complete=True` and no exception escapes `_process_message_async`. Re-enqueue the same message and assert the L2 idempotency skip fires.
- **Trace re-check:** After deploy, replay the original payload from trace `9e211b130acc4b8fbc6d036ba2c3c430` in a non-prod environment and confirm a single Meta Agent run, single 422 observation, source marked complete.

## Rollout

- Single PR against MIRIX_Intuit `re-org` (per memory: `re-org` is the base branch).
- No env-var flag — the change is strictly more defensive. If we want to be extra cautious, gate behind `MIRIX_FAIL_FAST_ON_422=true` (default off) for one deploy, then flip on. Recommendation: skip the flag; the change is small and any regression is easy to revert.

## Future work (not in scope)

- Add `rejection_status` / `rejection_reason` columns to `memory_sources` so `GET /v1/memory-sources/{id}` can expose rejection to callers.
- Same treatment for other non-retryable 4xx — 400, 401, 403, 404 — once we've confirmed they don't legitimately appear in steady state.
- Lower `MIRIX_LLM_RETRY_LIMIT` or shorten `MIRIX_LLM_RETRY_MAX_DELAY` to reduce wall time for retryable errors; orthogonal to this fix but related to the Kafka rebalance risk.
