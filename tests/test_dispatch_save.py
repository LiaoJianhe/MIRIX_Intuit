"""Tests for the unified save dispatcher (`dispatch_save`).

After VEPAGE-1251, all three run modes (numaflow, kafka, in-memory) route
through `error_policy.dispatch_save`. The dispatcher:

1. Runs the save under `process_with_policy` (classify + bounded retry).
2. Routes the verdict to the single finalize chokepoint
   (`MemorySourceManager.finalize_source`).

step() does NOT finalize internally anymore — the dispatcher handles ALL
outcomes including SUCCESS (Option B).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.errors import (
    LLMChainingExhaustedError,
    LLMUnprocessableEntityError,
    LLMRateLimitError,
)
from mirix.queue.error_policy import SaveOutcome, dispatch_save


@pytest.mark.asyncio
async def test_dispatch_save_success_finalizes_with_success():
    """A clean save returns SaveOutcome.SUCCESS and the dispatcher calls
    finalize_source(SUCCESS)."""
    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        return None

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-ok")

    assert outcome.kind is SaveOutcome.SUCCESS
    finalize.assert_awaited_once_with("src-ok", SaveOutcome.SUCCESS)


@pytest.mark.asyncio
async def test_dispatch_save_permanent_finalizes_with_permanent():
    """A Permanent classification routes through finalize_source(PERMANENT_FAILURE)."""
    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        raise LLMUnprocessableEntityError("422 rejected")

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-perm")

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    finalize.assert_awaited_once_with("src-perm", SaveOutcome.PERMANENT_FAILURE)


@pytest.mark.asyncio
async def test_dispatch_save_llm_chaining_exhausted_is_permanent(monkeypatch):
    """LLMChainingExhaustedError (raised by step() when function_failed
    exhausts chaining) classifies as PERMANENT and finalizes accordingly.
    This is the case where the meta-agent LLM produced malformed tool calls
    past its budget — retrying the whole step won't help."""
    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        raise LLMChainingExhaustedError("LLM gave up")

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-chain")

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    finalize.assert_awaited_once_with("src-chain", SaveOutcome.PERMANENT_FAILURE)


@pytest.mark.asyncio
async def test_dispatch_save_transient_exhausted_finalizes_with_transient(monkeypatch):
    """A transient that exhausts the retry budget returns
    TRANSIENT_EXHAUSTED and the dispatcher finalizes. No conscious
    redelivery — all 3 modes treat exhausted-transient as dead-letter."""
    from mirix.queue import error_policy as ep

    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)

    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        raise LLMRateLimitError("429 always")

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-trans")

    assert outcome.kind is SaveOutcome.TRANSIENT_EXHAUSTED
    finalize.assert_awaited_once_with("src-trans", SaveOutcome.TRANSIENT_EXHAUSTED)


@pytest.mark.asyncio
async def test_dispatch_save_without_source_id_skips_finalize():
    """No memory_source_id → finalize is not called (nothing to finalize)."""
    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        return None

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id=None)

    assert outcome.kind is SaveOutcome.SUCCESS
    finalize.assert_not_awaited()
