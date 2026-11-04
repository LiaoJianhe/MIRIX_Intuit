"""Tests for the post-save hook.

``dispatch_save`` is the one chokepoint every run mode (numaflow, internal
kafka worker, in-memory worker) passes through. After a save is finalized it
tells each registered function what happened. The hook only notifies: it never
changes the save's outcome or delays its return.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.errors import LLMUnprocessableEntityError
from mirix.observability.trace_attrs import (
    get_written_memories,
    record_written_memory,
    reset_save_written,
    set_save_written,
)
from mirix.queue import post_save
from mirix.queue.error_policy import SaveOutcome, dispatch_save
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.post_save import PostSaveEvent, notify_post_save, register_post_save_hook


@pytest.fixture(autouse=True)
def _clean_hooks():
    post_save.clear_post_save_hooks()
    yield
    post_save.clear_post_save_hooks()
    reset_save_written(None)


def _event(**overrides) -> PostSaveEvent:
    base = dict(
        memory_source_id="src-1",
        user_id="u-1",
        client_id="c-1",
        agent_id="a-1",
        outcome=SaveOutcome.SUCCESS,
        written=(("episodic", "ep-1"),),
    )
    base.update(overrides)
    return PostSaveEvent(**base)


def _message() -> QueueMessage:
    msg = QueueMessage()
    msg.client_id = "c-1"
    msg.agent_id = "a-1"
    msg.user_id = "u-1"
    return msg


# --- written-memory record ------------------------------------------------


def test_record_is_a_silent_no_op_without_an_active_save():
    record_written_memory("episodic", "ep-1")
    assert get_written_memories() == ()


def test_record_keeps_type_and_id_pairs_in_order():
    token = set_save_written()
    try:
        record_written_memory("episodic", "ep-1")
        record_written_memory("core", "blk-1")
        assert get_written_memories() == (("episodic", "ep-1"), ("core", "blk-1"))
    finally:
        reset_save_written(token)


def test_a_new_save_starts_empty():
    first = set_save_written()
    record_written_memory("semantic", "sem-1")
    reset_save_written(first)
    second = set_save_written()
    try:
        assert get_written_memories() == ()
    finally:
        reset_save_written(second)


def test_the_same_memory_is_listed_once():
    token = set_save_written()
    try:
        record_written_memory("episodic", "ep-1")
        record_written_memory("episodic", "ep-1")
        assert get_written_memories() == (("episodic", "ep-1"),)
    finally:
        reset_save_written(token)


# --- registry and notify --------------------------------------------------


@pytest.mark.asyncio
async def test_no_registered_hook_is_a_no_op():
    notify_post_save(_event())
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_registered_hook_receives_the_event():
    seen = []

    async def hook(event):
        seen.append(event)

    register_post_save_hook(hook)
    event = _event()
    notify_post_save(event)
    await asyncio.sleep(0.01)
    assert seen == [event]


@pytest.mark.asyncio
async def test_registering_twice_runs_the_hook_once():
    calls = []

    async def hook(event):
        calls.append(event)

    register_post_save_hook(hook)
    register_post_save_hook(hook)
    notify_post_save(_event())
    await asyncio.sleep(0.01)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_raising_hook_does_not_stop_the_others_or_escape():
    seen = []

    async def bad(event):
        raise RuntimeError("boom")

    async def good(event):
        seen.append(event)

    register_post_save_hook(bad)
    register_post_save_hook(good)
    notify_post_save(_event())  # must not raise
    await asyncio.sleep(0.01)
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_slow_hook_does_not_delay_notify():
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(event):
        started.set()
        await release.wait()

    register_post_save_hook(slow)
    notify_post_save(_event())  # returns at once, does not await the hook
    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await asyncio.sleep(0)


def test_notify_without_a_running_loop_does_not_raise():
    async def hook(event):  # pragma: no cover - never scheduled
        raise AssertionError("should not run")

    register_post_save_hook(hook)
    notify_post_save(_event())


# --- dispatch_save integration -------------------------------------------


def _manager_patch():
    return patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=MagicMock(finalize_source=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_dispatch_save_notifies_after_finalize_with_written_ids():
    order = []
    seen = []

    async def hook(event):
        seen.append(event)

    register_post_save_hook(hook)

    async def finalize(source_id, kind):
        order.append("finalize")

    mgr = MagicMock(finalize_source=AsyncMock(side_effect=finalize))

    async def run_step():
        record_written_memory("episodic", "ep-9")

    real_notify = post_save.notify_post_save

    def spy(event):
        order.append("notify")
        real_notify(event)

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=mgr),
        patch("mirix.queue.error_policy.notify_post_save", side_effect=spy),
    ):
        outcome = await dispatch_save(run_step, memory_source_id="src-9", message=_message())
        await asyncio.sleep(0.01)

    assert outcome.kind is SaveOutcome.SUCCESS
    assert order == ["finalize", "notify"]
    assert seen == [
        PostSaveEvent(
            memory_source_id="src-9",
            user_id="u-1",
            client_id="c-1",
            agent_id="a-1",
            outcome=SaveOutcome.SUCCESS,
            written=(("episodic", "ep-9"),),
        )
    ]


@pytest.mark.asyncio
async def test_dispatch_save_reports_a_permanent_failure_outcome():
    seen = []

    async def hook(event):
        seen.append(event)

    register_post_save_hook(hook)

    async def run_step():
        raise LLMUnprocessableEntityError("bad input")

    with _manager_patch():
        outcome = await dispatch_save(run_step, memory_source_id="src-p", message=_message())
        await asyncio.sleep(0.01)

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    assert [e.outcome for e in seen] == [SaveOutcome.PERMANENT_FAILURE]
    assert seen[0].written == ()


@pytest.mark.asyncio
async def test_a_raising_hook_does_not_change_the_save_outcome():
    async def bad(event):
        raise RuntimeError("boom")

    register_post_save_hook(bad)

    async def run_step():
        return None

    with _manager_patch():
        outcome = await dispatch_save(run_step, memory_source_id="src-ok", message=_message())
        await asyncio.sleep(0.01)

    assert outcome.kind is SaveOutcome.SUCCESS


@pytest.mark.asyncio
async def test_written_ids_do_not_leak_into_the_next_save():
    seen = []

    async def hook(event):
        seen.append(event.written)

    register_post_save_hook(hook)

    async def first():
        record_written_memory("semantic", "sem-1")

    async def second():
        return None

    with _manager_patch():
        await dispatch_save(first, memory_source_id="src-a", message=_message())
        await dispatch_save(second, memory_source_id="src-b", message=_message())
        await asyncio.sleep(0.01)

    assert seen == [(("semantic", "sem-1"),), ()]


@pytest.mark.asyncio
async def test_dispatch_save_without_a_message_still_works():
    async def run_step():
        return None

    with _manager_patch():
        outcome = await dispatch_save(run_step, memory_source_id=None)

    assert outcome.kind is SaveOutcome.SUCCESS
