"""Trace-level attribute helpers: the tag accumulator and the per-save write counts.

Langfuse replaces a trace's tag list wholesale on every trace-level tag write
(the ``langfuse.trace.tags`` span attribute), and the memory-save trace is written
from multiple independent call sites (the HTTP-entry decorator, the worker's
Meta Agent open, the finalize-time outcome emitter) — so a naive per-site
write silently clobbers whatever the previous site wrote (last writer wins).

:func:`update_trace_attributes` kills that structurally instead of by
discipline: new tags merge into a per-context set and every write sends the
FULL sorted set, so whatever the flush order, the trace converges to the
union. The set lives in a ContextVar with the same lifecycle as the trace
context (``clear_trace_context()`` resets it via :func:`reset_trace_tags`).

This module also hosts the per-save write-count accumulator: a ContextVar
dict set by ``dispatch_save`` and bumped in ``_write_citation`` (the single
funnel every memory write passes through), so the worker root span can close
with ``writes_by_memory_type`` without any extra I/O on the save path.
"""

import json
from contextvars import ContextVar, Token
from typing import Any, Dict, List, Optional, Set

from opentelemetry.trace import get_current_span

from mirix.log import get_logger
from mirix.observability.langfuse_client import get_langfuse_client

logger = get_logger(__name__)

# Langfuse v4 removed ``Langfuse.update_current_trace``: trace-level fields are
# plain attributes on a span, and the backend upserts the trace from whichever
# span carried them last. These are the wire names the v4 SDK itself writes
# (``langfuse._client.attributes.LangfuseOtelSpanAttributes``); they are
# duplicated here, not imported, to avoid depending on a private module.
_ATTR_TRACE_NAME = "langfuse.trace.name"
_ATTR_TRACE_USER_ID = "user.id"
_ATTR_TRACE_SESSION_ID = "session.id"
_ATTR_TRACE_TAGS = "langfuse.trace.tags"
_ATTR_TRACE_METADATA = "langfuse.trace.metadata"

# The accumulated tag set for the current trace. ``None`` means "no tags
# accumulated yet" — distinct from an empty set only in that we lazily create
# the set on first use.
_trace_tags: ContextVar[Optional[Set[str]]] = ContextVar("trace_tags", default=None)


def write_trace_attributes(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write trace-level attributes onto the CURRENT span (Langfuse v4 replacement
    for ``update_current_trace``).

    ``tags`` replaces the trace's tag list wholesale — callers that need merge
    semantics go through :func:`update_trace_attributes`. ``metadata`` is written
    one ``langfuse.trace.metadata.<key>`` attribute per key; ``None`` values are
    skipped and non-primitive values are JSON-encoded, since OTel attributes only
    carry primitives. No-op when there is no recording span.
    """
    span = get_current_span()
    if span is None or not span.is_recording():
        return

    if name:
        span.set_attribute(_ATTR_TRACE_NAME, name)
    if user_id:
        span.set_attribute(_ATTR_TRACE_USER_ID, user_id)
    if session_id:
        span.set_attribute(_ATTR_TRACE_SESSION_ID, session_id)
    if tags:
        span.set_attribute(_ATTR_TRACE_TAGS, list(tags))
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if not isinstance(value, (str, bool, int, float)):
            value = json.dumps(value, default=str)
        span.set_attribute(f"{_ATTR_TRACE_METADATA}.{key}", value)


def update_trace_attributes(
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge ``tags`` into the per-context tag set and write the FULL set.

    Args:
        tags: NEW tags to add, e.g. ``["client:aqna-client"]``. Tag strings are
            used verbatim (``f"{prefix}:{value}"``; values may themselves
            contain ``:`` — tags are exact-match strings).
        metadata: Trace metadata for this write. Langfuse merges metadata by
            key server-side, so it is passed through per call (no client-side
            accumulation needed).

    No-op when Langfuse is disabled or no trace_id is active. Never raises —
    instrumentation must not break the save path.
    """
    try:
        langfuse = get_langfuse_client()
        if langfuse is None:
            return

        # Local import to avoid a module cycle (context.clear_trace_context
        # calls back into reset_trace_tags below).
        from mirix.observability.context import get_trace_context

        trace_context = get_trace_context()
        if not (trace_context and trace_context.get("trace_id")):
            return

        current = _trace_tags.get()
        if current is None:
            current = set()
            _trace_tags.set(current)
        if tags:
            current.update(tags)

        kwargs: Dict[str, Any] = {}
        if current:
            kwargs["tags"] = sorted(current)
        if metadata is not None:
            kwargs["metadata"] = metadata
        if not kwargs:
            return

        write_trace_attributes(**kwargs)
    except Exception as e:  # noqa: BLE001 - instrumentation never raises
        logger.debug("update_trace_attributes failed (ignored): %s", e)


def reset_trace_tags() -> None:
    """Empty the per-context tag set.

    Called from ``clear_trace_context()`` so the tag lifecycle matches the
    trace-context lifecycle: one save's tags can never leak into the next
    message processed on the same (reused) worker task.
    """
    _trace_tags.set(None)


# --------------------------------------------------------------------------- #
# Per-save write-count accumulator (R3: Meta Agent root-span output).
#
# Same lifecycle pattern as the fault-injection active-source scope:
# dispatch_save sets a fresh dict per save and resets it in its finally;
# tasks spawned under asyncio.gather copy the context but share the parent's
# MUTABLE dict object, so concurrent sub-agents all bump the same counts.
# --------------------------------------------------------------------------- #

_save_write_counts: ContextVar[Optional[Dict[str, int]]] = ContextVar("save_write_counts", default=None)


def set_save_write_counts() -> Token:
    """Publish a fresh, empty write-count dict for the current save.

    Returns the ContextVar token so :func:`reset_save_write_counts` can
    restore the prior state in ``dispatch_save``'s ``finally``.
    """
    return _save_write_counts.set({})


def reset_save_write_counts(token: Optional[Token]) -> None:
    """Restore the write-count state that preceded :func:`set_save_write_counts`."""
    if token is not None:
        try:
            _save_write_counts.reset(token)
        except Exception:  # noqa: BLE001 - cross-context reset is a no-op
            _save_write_counts.set(None)


def bump_write_count(memory_type: str) -> None:
    """Count one memory write of ``memory_type`` for the active save.

    Silent no-op when no save is active (e.g. unit tests exercising a tool
    function directly) — instrumentation must never gate the write path.
    """
    counts = _save_write_counts.get()
    if counts is None:
        return
    counts[memory_type] = counts.get(memory_type, 0) + 1


def get_write_counts() -> Dict[str, int]:
    """Return a snapshot of the active save's write counts (``{}`` when unset)."""
    counts = _save_write_counts.get()
    return dict(counts) if counts else {}
