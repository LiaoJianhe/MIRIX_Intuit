"""Tests for create_episodic_memory_direct and the direct-memory helpers in rest_api.py."""

import pytest


def test_memory_source_input_defaults():
    from mirix.server.rest_api import MemorySourceInput

    src = MemorySourceInput()
    assert src.external_id is None
    assert src.external_thread_id is None
    assert src.source_type == "conversation"
    assert src.source_system is None
    assert src.source_metadata is None
    assert src.occurred_at is None
    assert src.messages is None


def test_memory_source_input_accepts_messages():
    from mirix.server.rest_api import MemorySourceInput, SourceMessageInput

    msgs = [
        SourceMessageInput(role="user", content="hi"),
        SourceMessageInput(role="assistant", content="hello"),
    ]
    src = MemorySourceInput(external_id="ext-1", messages=msgs)
    assert src.external_id == "ext-1"
    assert len(src.messages) == 2
    assert src.messages[0].role == "user"


def test_source_message_input_fields():
    from mirix.server.rest_api import SourceMessageInput

    msg = SourceMessageInput(
        role="system",
        content="x",
        external_message_id="m-1",
        occurred_at="2026-04-17T00:00:00Z",
    )
    assert msg.role == "system"
    assert msg.content == "x"
    assert msg.external_message_id == "m-1"
    assert msg.occurred_at == "2026-04-17T00:00:00Z"


def test_create_episodic_memory_direct_request_requires_source():
    """CreateEpisodicMemoryDirectRequest rejects construction without source."""
    import pydantic

    from mirix.server.rest_api import CreateEpisodicMemoryDirectRequest

    with pytest.raises(pydantic.ValidationError):
        CreateEpisodicMemoryDirectRequest(
            user_id="u-1",
            event_type="evt",
            summary="s",
            details="d",
            event_actor="system",
        )


def test_create_episodic_memory_direct_request_happy():
    from mirix.server.rest_api import (
        CreateEpisodicMemoryDirectRequest,
        MemorySourceInput,
    )

    req = CreateEpisodicMemoryDirectRequest(
        user_id="u-1",
        event_type="evt",
        summary="s",
        details="d",
        event_actor="system",
        source=MemorySourceInput(external_id="ext-a"),
    )
    assert req.user_id == "u-1"
    assert req.filter_tags is None
    assert req.occurred_at is None
    assert req.source.external_id == "ext-a"
    assert req.source.source_type == "conversation"
