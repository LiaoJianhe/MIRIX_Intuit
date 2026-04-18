"""Direct-write handlers for the `direct_writes` field on QueueMessage.

Lives outside ``mirix/functions/function_sets/`` because these handlers are
not agent-callable tools. Placing them under ``function_sets/`` would cause
``tool_manager.upsert_base_tools`` to scan them via ``load_function_set``
(which requires tool-style docstrings, no ``*,`` separators, etc.) and fail.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from mirix.functions.function_sets.memory_tools import _write_citation
from mirix.schemas.agent import AgentType
from mirix.schemas.episodic_memory import EpisodicEvent as PydanticEpisodicEvent

if TYPE_CHECKING:
    from mirix.agent.agent import Agent


async def direct_write_episodic(
    agent: "Agent",
    *,
    event_type: str,
    summary: str,
    details: str,
    event_actor: str,
    occurred_at: Optional[str] = None,
) -> None:
    """Direct-insert an episodic memory row + citation without LLM dispatch.

    Reads memory_source_id / external_thread_id / filter_tags off the agent
    (hydrated by AsyncServer._step). Uses insert_event when the agent has
    a meta_memory_agent state, else create_episodic_memory. Then calls the
    shared _write_citation helper for the citation row.
    """
    effective_occurred_at = occurred_at if occurred_at is not None else getattr(agent, "occurred_at", None)
    if effective_occurred_at:
        if isinstance(effective_occurred_at, str):
            occurred_dt = datetime.fromisoformat(effective_occurred_at.replace("Z", "+00:00"))
        else:
            occurred_dt = effective_occurred_at
        if occurred_dt.tzinfo is not None:
            occurred_dt = occurred_dt.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        occurred_dt = datetime.now(timezone.utc).replace(tzinfo=None)

    client = agent.actor
    user_id = getattr(agent, "user_id", None)
    org_id = str(client.organization_id)

    existing_tags = getattr(agent, "filter_tags", None)
    filter_tags = dict(existing_tags) if existing_tags else {}
    if "scope" not in filter_tags and getattr(client, "write_scope", None):
        filter_tags["scope"] = client.write_scope

    if agent.agent_state.is_type(AgentType.meta_memory_agent):
        agent_id = agent.agent_state.parent_id or agent.agent_state.id
        event_result = await agent.episodic_memory_manager.insert_event(
            actor=client,
            agent_state=agent.agent_state,
            agent_id=agent_id,
            event_type=event_type,
            timestamp=occurred_dt,
            event_actor=event_actor,
            details=details,
            summary=summary,
            organization_id=org_id,
            filter_tags=filter_tags or None,
            client_id=client.id,
            user_id=user_id,
            use_cache=getattr(agent, "use_cache", True),
        )
    else:
        event_model = PydanticEpisodicEvent(
            occurred_at=occurred_dt,
            event_type=event_type,
            client_id=client.id,
            user_id=user_id,
            agent_id=None,
            actor=event_actor,
            summary=summary,
            details=details,
            organization_id=org_id,
            summary_embedding=None,
            details_embedding=None,
            embedding_config=None,
            filter_tags=filter_tags or None,
            last_modify={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": "created",
            },
        )
        event_result = await agent.episodic_memory_manager.create_episodic_memory(
            event_model,
            actor=client,
            client_id=client.id,
            user_id=user_id,
            use_cache=getattr(agent, "use_cache", True),
        )

    await _write_citation(agent, memory_type="episodic", memory_id=event_result.id, citation_type="created")


DIRECT_WRITE_HANDLERS: Dict[str, Any] = {
    "episodic": direct_write_episodic,
}
