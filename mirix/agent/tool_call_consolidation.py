"""Consolidate multi-tool-call responses from memory agents.

Memory agents are prompted to emit exactly one tool call per step. When an LLM
returns more than one, we reconcile it here:

  * If every tool call is the same batched insert tool (e.g. all
    ``episodic_memory_insert``), merge their ``items`` arrays into a single
    call. Inserts are additive, the tool schema already accepts batched
    ``items``, and this recovers data that would otherwise be dropped.
  * Otherwise — mixed tool names, or any tool whose semantics are not safely
    combinable (updates, deletes, replaces with ``old_ids``) — keep the first
    tool call and drop the rest, with a structured warning for monitoring.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Iterable

from mirix.schemas.agent import AgentType

if TYPE_CHECKING:
    from mirix.schemas.openai.chat_completion_response import ToolCall


MEMORY_AGENT_TYPES: frozenset[AgentType] = frozenset(
    {
        AgentType.core_memory_agent,
        AgentType.episodic_memory_agent,
        AgentType.procedural_memory_agent,
        AgentType.resource_memory_agent,
        AgentType.knowledge_vault_memory_agent,
        AgentType.semantic_memory_agent,
    }
)

# Tools that (a) accept an ``items: [...]`` batch parameter and (b) are purely
# additive, so merging items across calls is semantically equivalent to the
# LLM having emitted a single batched call.
COMBINABLE_INSERT_TOOLS: frozenset[str] = frozenset(
    {
        "episodic_memory_insert",
        "semantic_memory_insert",
        "resource_memory_insert",
        "procedural_memory_insert",
        "knowledge_vault_insert",
    }
)


def consolidate_memory_agent_tool_calls(
    tool_calls: list["ToolCall"],
    agent_type: AgentType,
    logger: logging.Logger,
) -> list["ToolCall"]:
    """Return a single-tool-call list consolidated from possibly-many tool calls.

    For non-memory agents, or when zero/one tool calls are provided, the input
    is returned unchanged. Otherwise the rules above apply.
    """
    if agent_type not in MEMORY_AGENT_TYPES:
        return tool_calls

    if len(tool_calls) <= 1:
        return tool_calls

    combined = _try_combine_same_insert(tool_calls)
    if combined is not None:
        logger.info(
            "Combined %d memory-agent tool call(s) into one batched call "
            "(agent_type=%s, tool=%s, total_items=%d)",
            len(tool_calls),
            agent_type,
            combined.function.name,
            _item_count(combined),
        )
        return [combined]

    # Fallback: truncate to first and emit a structured warning so we can
    # monitor the rate of non-combinable multi-tool-call responses.
    kept = tool_calls[0]
    dropped = tool_calls[1:]
    dropped_desc = [
        f"{tc.function.name}:{tc.id}" for tc in dropped if tc and tc.function
    ]
    logger.warning(
        "Truncating %d extra tool call(s) for memory agent %s "
        "(keeping %s:%s, dropping %s)",
        len(dropped),
        agent_type,
        kept.function.name if kept and kept.function else None,
        kept.id if kept else None,
        dropped_desc,
    )
    return [kept]


def _try_combine_same_insert(
    tool_calls: Iterable["ToolCall"],
) -> "ToolCall | None":
    """If all calls are the same combinable insert with parseable ``items``,
    return one merged ToolCall. Otherwise return None."""
    tool_calls = list(tool_calls)
    if not tool_calls:
        return None

    first = tool_calls[0]
    if first is None or first.function is None:
        return None
    name = first.function.name
    if name not in COMBINABLE_INSERT_TOOLS:
        return None

    all_items: list = []
    # Keep the first call's args so we preserve any non-items keys the caller
    # may have set (none are expected today, but future-proof).
    try:
        base_args = json.loads(first.function.arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(base_args, dict) or "items" not in base_args:
        return None
    if not isinstance(base_args["items"], list):
        return None
    all_items.extend(base_args["items"])

    for tc in tool_calls[1:]:
        if tc is None or tc.function is None:
            return None
        if tc.function.name != name:
            return None
        try:
            args = json.loads(tc.function.arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(args, dict) or "items" not in args:
            return None
        if not isinstance(args["items"], list):
            return None
        all_items.extend(args["items"])

    merged = first.model_copy(deep=True)
    merged_args = dict(base_args)
    merged_args["items"] = all_items
    merged.function.arguments = json.dumps(merged_args)
    return merged


def _item_count(tool_call: "ToolCall") -> int:
    try:
        args = json.loads(tool_call.function.arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return -1
    items = args.get("items") if isinstance(args, dict) else None
    return len(items) if isinstance(items, list) else -1
