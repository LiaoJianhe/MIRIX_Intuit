"""Direct-write handlers for the `direct_writes` field on QueueMessage.

Lives outside ``mirix/functions/function_sets/`` because these handlers are
not agent-callable tools. Placing them under ``function_sets/`` would cause
``tool_manager.upsert_base_tools`` to scan them via ``load_function_set``
(which requires tool-style docstrings, no ``*,`` separators, etc.) and fail.

The handlers are thin shims that delegate to the existing LLM-facing memory
tool functions (e.g. ``episodic_memory_insert``) so citation-writing, filter
tag injection, and manager interaction live in exactly one place.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict

from mirix.functions.function_sets.memory_tools import episodic_memory_insert

if TYPE_CHECKING:
    from mirix.agent.agent import Agent


async def direct_write_episodic(agent: "Agent", **payload: Any) -> None:
    """Direct-write one episodic memory by delegating to ``episodic_memory_insert``.

    Accepts the same per-item fields the LLM tool does:
    ``event_type``, ``summary``, ``details``, ``event_actor`` (aliased to ``actor``
    for the tool), ``occurred_at``. Wraps the single item in the list-shaped
    argument the tool expects. Missing ``occurred_at`` defaults to now().
    """
    occurred_at = payload.get("occurred_at") or getattr(agent, "occurred_at", None)
    if occurred_at is None:
        occurred_at = datetime.now(timezone.utc).isoformat()

    item = {
        "event_type": payload["event_type"],
        "summary": payload["summary"],
        "details": payload["details"],
        "actor": payload["event_actor"],
        "occurred_at": occurred_at,
    }
    await episodic_memory_insert(agent, items=[item])


DIRECT_WRITE_HANDLERS: Dict[str, Any] = {
    "episodic": direct_write_episodic,
}
