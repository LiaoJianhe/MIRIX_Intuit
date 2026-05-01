"""Direct-write handler registry for the ``direct_writes`` field on QueueMessage.

Maps ``memory_type`` to the existing LLM-facing memory tool function. The
meta-agent's ``_apply_direct_write`` unpacks the payload as kwargs directly
into the tool call, so the payload shape matches the tool signature
(e.g. ``{"items": [{...}]}`` for ``episodic_memory_insert``). This keeps
filter-tag injection, manager coordination, and citation writing in one
place — the existing tool — without a shim layer.

Lives outside ``mirix/functions/function_sets/`` because
``tool_manager.upsert_base_tools`` walks every public function in that
directory via ``load_function_set`` (which requires tool-style docstrings).
This registry is not a function; it only imports the tool callables.
"""

from typing import Any, Callable, Dict

from mirix.functions.function_sets.memory_tools import (
    episodic_memory_insert,
    knowledge_vault_insert,
    procedural_memory_insert,
    resource_memory_insert,
    semantic_memory_insert,
)

DIRECT_WRITE_HANDLERS: Dict[str, Callable[..., Any]] = {
    "episodic": episodic_memory_insert,
    "semantic": semantic_memory_insert,
    "procedural": procedural_memory_insert,
    "resource": resource_memory_insert,
    "knowledge_vault": knowledge_vault_insert,
}
