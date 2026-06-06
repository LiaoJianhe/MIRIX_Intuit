"""Span-tree correctness for the ``timed_span`` helper.

Spans in this codebase resolve their parent from the ``current_observation_id``
ContextVar (``mirix/observability/context.py``); each span-creation site reads
``get_trace_context()["observation_id"]`` as its parent. ``timed_span`` opens a
Langfuse child span but must ALSO publish that span's id into the ContextVar for
the duration of the wrapped block, so any span opened inside the block (including
a nested ``timed_span``) nests under it rather than under its parent.

Concrete case (the ticket's named regression): ``_step`` opens
``timed_span("Load Agent")`` and inside it calls ``load_agent`` which opens
``timed_span("Load Agent State")``. ``Load Agent State`` must render as a CHILD of
``Load Agent``, not a sibling.

After the block exits, the prior observation id must be restored (including a
prior of ``None``) so the next sibling span parents back to the original parent.
"""

from unittest.mock import MagicMock, patch

import pytest

from mirix.observability import context as obs_context
from mirix.observability.timed_spans import timed_span


def _make_langfuse_with_span(span_id):
    """Fake Langfuse whose span exposes ``.id == span_id`` and acts as a sync
    context manager (mirrors real start_as_current_observation usage)."""
    span = MagicMock()
    span.id = span_id
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    langfuse = MagicMock()
    langfuse.start_as_current_observation.return_value = cm
    return langfuse, span


@pytest.mark.asyncio
async def test_block_runs_under_timed_span_as_parent():
    """While the wrapped block runs, the current observation id is the timed_span's
    own span id (so spans opened inside parent to it)."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="outer-obs")
    langfuse, _span = _make_langfuse_with_span("timed-span-obs")

    captured = {}
    with (
        patch("mirix.observability.timed_spans.get_langfuse_client", return_value=langfuse),
        patch("mirix.observability.timed_spans.mark_observation_as_child"),
    ):
        async with timed_span("Load Agent"):
            captured["observation_id"] = obs_context.get_trace_context().get("observation_id")

    assert captured["observation_id"] == "timed-span-obs"


@pytest.mark.asyncio
async def test_nested_timed_span_parents_under_outer_timed_span():
    """A timed_span opened inside another timed_span sees the OUTER span's id as
    its parent (the Load Agent -> Load Agent State case)."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="root-obs")

    outer_langfuse, _outer = _make_langfuse_with_span("load-agent-obs")

    captured = {}
    with (
        patch("mirix.observability.timed_spans.get_langfuse_client", return_value=outer_langfuse),
        patch("mirix.observability.timed_spans.mark_observation_as_child"),
    ):
        async with timed_span("Load Agent"):
            # The inner timed_span must build its trace context with the OUTER
            # span as its parent. Capture the parent_span_id passed to Langfuse.
            async with timed_span("Load Agent State"):
                pass
            # The second start_as_current_observation call is the inner span.
            inner_call = outer_langfuse.start_as_current_observation.call_args_list[1]
            captured["inner_parent"] = inner_call.kwargs["trace_context"].get("parent_span_id")

    assert captured["inner_parent"] == "load-agent-obs"


@pytest.mark.asyncio
async def test_observation_id_restored_after_block():
    """After the block exits, the prior observation id is restored so the next
    sibling span parents to the original parent."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="outer-obs")
    langfuse, _span = _make_langfuse_with_span("timed-span-obs")

    with (
        patch("mirix.observability.timed_spans.get_langfuse_client", return_value=langfuse),
        patch("mirix.observability.timed_spans.mark_observation_as_child"),
    ):
        async with timed_span("Load Agent"):
            pass

    assert obs_context.get_trace_context().get("observation_id") == "outer-obs"


@pytest.mark.asyncio
async def test_observation_id_restored_to_none_when_no_prior_parent():
    """When there was no prior observation (span sits directly under the trace
    root), the observation id must be cleared back to None afterward — not left
    pointing at the closed timed_span."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1")  # trace_id but no observation_id
    langfuse, _span = _make_langfuse_with_span("timed-span-obs")

    with (
        patch("mirix.observability.timed_spans.get_langfuse_client", return_value=langfuse),
        patch("mirix.observability.timed_spans.mark_observation_as_child"),
    ):
        async with timed_span("Load Agent"):
            pass

    assert obs_context.get_trace_context().get("observation_id") is None


@pytest.mark.asyncio
async def test_no_op_when_langfuse_disabled_leaves_context_untouched():
    """When tracing is unavailable the block still runs and the ContextVar is
    left untouched."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="outer-obs")

    ran = {}
    with patch("mirix.observability.timed_spans.get_langfuse_client", return_value=None):
        async with timed_span("Load Agent"):
            ran["did_run"] = True
            ran["observation_id"] = obs_context.get_trace_context().get("observation_id")

    assert ran["did_run"] is True
    assert ran["observation_id"] == "outer-obs"
    assert obs_context.get_trace_context().get("observation_id") == "outer-obs"
