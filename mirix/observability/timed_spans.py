"""Reusable child-span context manager for timing in-pipeline work.

Several pieces of the meta-agent pipeline (source persistence, citation writes,
sub-agent dispatch) do real wall-clock work but emit no Langfuse span, so they
show up as unaccounted gaps inside the parent agent span. ``timed_span`` wraps a
block in a child observation so its duration is attributed in the trace tree.

It mirrors the guard/error handling of ``emit_idempotency_skip_span``: attaches
under the current trace context as a child, and is a clean no-op when Langfuse
is disabled or no trace context is active (so it is always safe to wrap a block,
including in unit tests with no tracing configured).
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, cast

from mirix.log import get_logger
from mirix.observability.context import (
    current_observation_id,
    get_tid,
    get_trace_context,
    mark_observation_as_child,
)
from mirix.observability.langfuse_client import get_langfuse_client

logger = get_logger(__name__)


@asynccontextmanager
async def timed_span(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[None]:
    """Wrap an ``await`` block in a child Langfuse span for duration attribution.

    Args:
        name: Span name shown in the trace (e.g. "Persist Memory Source").
        metadata: Extra fields merged into span metadata for inspection.

    No-op (still runs the wrapped block) when Langfuse is disabled or no trace
    context is active. Tracing failures never propagate to the wrapped work.
    """
    langfuse = get_langfuse_client()
    trace_context = get_trace_context()
    trace_id = trace_context.get("trace_id") if trace_context else None
    parent_span_id = trace_context.get("observation_id") if trace_context else None

    if not (langfuse and trace_id):
        # Tracing unavailable: run the block untouched.
        yield
        return

    from langfuse.types import TraceContext

    trace_context_dict: Dict[str, Any] = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    # Stamp the TID into span metadata so the Langfuse OTel export emits
    # ``langfuse.observation.metadata.tid``. Consumers that filter spans by TID
    # (the full-stack-test span capture) would otherwise drop every nested
    # worker span — only the root spans that already stamp the tid (HTTP-entry
    # trace, worker "Meta Agent" observation) would survive. Mirrors the worker's
    # Meta Agent span metadata. Omitted when there's no active TID so we don't
    # write a misleading ``tid=None``.
    span_metadata: Dict[str, Any] = dict(metadata or {})
    tid = get_tid()
    if tid:
        span_metadata.setdefault("tid", tid)

    try:
        cm = langfuse.start_as_current_observation(
            name=name,
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            metadata=span_metadata,
        )
    except Exception as e:
        # If span creation itself fails, don't lose the work.
        logger.warning("timed_span(%s) failed to start: %s", name, e)
        yield
        return

    with cm as span:
        try:
            mark_observation_as_child(span)
        except Exception as e:
            logger.warning("timed_span(%s) failed to mark child: %s", name, e)

        # Publish this span as the current observation while the wrapped block
        # runs so any span opened inside it (including a nested timed_span)
        # nests under THIS span rather than under its parent. Restore the prior
        # observation id afterward so the next sibling span parents back to the
        # original parent.
        span_observation_id = getattr(span, "id", None)
        prior_observation_id = parent_span_id
        if span_observation_id:
            # set_trace_context ignores a falsy observation_id, so set the
            # ContextVar directly (and restore directly below) to also handle
            # the None / no-parent case correctly.
            current_observation_id.set(span_observation_id)
        try:
            yield
        finally:
            if span_observation_id:
                current_observation_id.set(prior_observation_id)
