"""Tests for direct_writes plumbing through put_messages into QueueMessage."""

import json

import pytest


@pytest.mark.asyncio
async def test_put_messages_serializes_direct_writes(monkeypatch):
    """direct_writes arg → QueueMessage.direct_writes populated with memory_type + payload_json."""
    from mirix.queue.message_pb2 import QueueMessage
    from mirix.queue.queue_util import put_messages

    captured = {}

    async def fake_save(message: QueueMessage) -> None:
        captured["msg"] = message

    monkeypatch.setattr("mirix.queue.queue_util.queue.save", fake_save)

    class _Actor:
        id = "client-1"

    await put_messages(
        actor=_Actor(),
        agent_id="agent-1",
        input_messages=[],
        chaining=None,
        user_id="user-1",
        verbose=False,
        filter_tags={"scope": "s"},
        block_filter_tags=None,
        block_filter_tags_update_mode="merge",
        use_cache=True,
        occurred_at="2026-04-17T10:00:00Z",
        memory_source_id="src-abcdef12",
        external_id="ext-1",
        external_thread_id=None,
        source_type="engagement",
        source_system="test-system",
        source_metadata={"display_id": "D1"},
        summary=None,
        summarize=False,
        source_messages=None,
        direct_writes=[
            {
                "memory_type": "episodic",
                "payload": {
                    "event_type": "e",
                    "summary": "s",
                    "details": "d",
                    "event_actor": "system",
                    "occurred_at": "2026-04-17T10:00:00Z",
                },
            },
        ],
    )

    msg = captured["msg"]
    assert len(msg.direct_writes) == 1
    assert msg.direct_writes[0].memory_type == "episodic"
    payload = json.loads(msg.direct_writes[0].payload_json)
    assert payload["event_type"] == "e"
    assert payload["event_actor"] == "system"
