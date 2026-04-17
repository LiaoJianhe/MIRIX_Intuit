"""Tests for the private type-agnostic helpers in rest_api.py."""

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio(loop_scope="module")

TEST_ORG_ID = "direct-helpers-org"
TEST_CLIENT_ID = "direct-helpers-client"
TEST_USER_ID = "direct-helpers-user"


def _unique(prefix: str = "src") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _provision_org_client_and_user():
    """Create the org, client, and user needed by all tests in this module."""
    from conftest import _create_client_and_key

    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    await _create_client_and_key(TEST_CLIENT_ID, TEST_ORG_ID, org_name="Direct Helpers Test Org")

    user_mgr = UserManager()
    try:
        await user_mgr.get_user_by_id(TEST_USER_ID)
    except Exception:
        await user_mgr.create_user(
            PydanticUser(
                id=TEST_USER_ID,
                name="Direct Helpers Test User",
                organization_id=TEST_ORG_ID,
                timezone="UTC",
            )
        )
    yield


async def _get_actor():
    """Fetch the test client as a PydanticClient for use as actor."""
    from mirix.services.client_manager import ClientManager

    client_mgr = ClientManager()
    return await client_mgr.get_client_by_id(TEST_CLIENT_ID)


async def test_persist_source_with_messages_happy_path():
    """Source + N messages persisted; returns (source, deduped=False)."""
    from mirix.server.rest_api import (
        MemorySourceInput,
        SourceMessageInput,
        _persist_source_with_messages,
    )
    from mirix.services.source_message_manager import SourceMessageManager

    actor = await _get_actor()
    ext_id = _unique("ext")
    memory_source_id = _unique("src")

    source_input = MemorySourceInput(
        external_id=ext_id,
        source_type="conversation",
        source_system="unit-test",
        messages=[
            SourceMessageInput(role="user", content="hi"),
            SourceMessageInput(role="assistant", content="hello"),
        ],
    )
    source, deduped = await _persist_source_with_messages(
        memory_source_id=memory_source_id,
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=source_input,
        fallback_occurred_at=None,
        filter_tags=None,
    )
    assert source is not None
    assert source.id == memory_source_id
    assert source.external_id == ext_id
    assert source.filter_tags.get("scope") == actor.write_scope
    assert deduped is False

    msgs_page = await SourceMessageManager().get_messages_by_source_id(source.id)
    assert len(msgs_page.items) == 2
    assert msgs_page.items[0].role == "user"
    assert msgs_page.items[1].role == "assistant"


async def test_persist_source_dedup_on_replay():
    """Second call with same external_id returns the original source and deduped=True."""
    from mirix.server.rest_api import MemorySourceInput, _persist_source_with_messages

    actor = await _get_actor()
    ext_id = _unique("ext-dedup")
    first_id = _unique("src")
    second_id = _unique("src")

    src_in = MemorySourceInput(external_id=ext_id, source_type="conversation")

    src1, dedup1 = await _persist_source_with_messages(
        memory_source_id=first_id,
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=src_in,
        fallback_occurred_at=None,
        filter_tags=None,
    )
    src2, dedup2 = await _persist_source_with_messages(
        memory_source_id=second_id,
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=src_in,
        fallback_occurred_at=None,
        filter_tags=None,
    )

    assert dedup1 is False
    assert dedup2 is True
    assert src1.id == src2.id  # second call returns the existing source
    assert src2.id == first_id


async def test_persist_source_no_messages():
    """When source.messages is None, no source_messages rows written."""
    from mirix.server.rest_api import MemorySourceInput, _persist_source_with_messages
    from mirix.services.source_message_manager import SourceMessageManager

    actor = await _get_actor()
    memory_source_id = _unique("src")

    src_in = MemorySourceInput(external_id=_unique("ext-nomsg"), messages=None)
    source, deduped = await _persist_source_with_messages(
        memory_source_id=memory_source_id,
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=src_in,
        fallback_occurred_at=None,
        filter_tags=None,
    )
    assert source is not None
    assert deduped is False
    msgs_page = await SourceMessageManager().get_messages_by_source_id(source.id)
    assert len(msgs_page.items) == 0


async def test_persist_source_occurred_at_fallback():
    """source.occurred_at absent -> use fallback_occurred_at."""
    from mirix.server.rest_api import MemorySourceInput, _persist_source_with_messages

    actor = await _get_actor()
    memory_source_id = _unique("src")

    src_in = MemorySourceInput(external_id=_unique("ext-fb"), occurred_at=None)
    source, _ = await _persist_source_with_messages(
        memory_source_id=memory_source_id,
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=src_in,
        fallback_occurred_at="2026-01-01T00:00:00Z",
        filter_tags=None,
    )
    assert source.occurred_at is not None
    assert source.occurred_at.year == 2026
    assert source.occurred_at.month == 1
    assert source.occurred_at.day == 1


async def test_write_citation_for_memory_happy_path():
    """Creates a citation row for (memory_type, memory_id)."""
    from mirix.server.rest_api import (
        MemorySourceInput,
        _persist_source_with_messages,
        _write_citation_for_memory,
    )

    actor = await _get_actor()
    source, _ = await _persist_source_with_messages(
        memory_source_id=_unique("src"),
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=MemorySourceInput(external_id=_unique("ext-cit")),
        fallback_occurred_at=None,
        filter_tags=None,
    )
    memory_id = _unique("episodic-fake")
    citation = await _write_citation_for_memory(
        memory_source_id=source.id,
        memory_type="episodic",
        memory_id=memory_id,
        citation_type="created",
        external_thread_id=None,
        occurred_at=None,
        created_by_id=actor.id,
    )
    assert citation is not None
    assert citation.memory_type == "episodic"
    assert citation.memory_id == memory_id
    assert citation.citation_type == "created"
    assert citation.memory_source_id == source.id


async def test_write_citation_type_agnostic():
    """Helper accepts arbitrary memory_type with no episodic-specific branching."""
    from mirix.server.rest_api import (
        MemorySourceInput,
        _persist_source_with_messages,
        _write_citation_for_memory,
    )

    actor = await _get_actor()
    source, _ = await _persist_source_with_messages(
        memory_source_id=_unique("src"),
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=MemorySourceInput(external_id=_unique("ext-cit-agnostic")),
        fallback_occurred_at=None,
        filter_tags=None,
    )
    for mtype in ("semantic", "procedural", "resource", "knowledge_vault", "core"):
        memory_id = _unique(f"{mtype}-fake")
        citation = await _write_citation_for_memory(
            memory_source_id=source.id,
            memory_type=mtype,
            memory_id=memory_id,
            citation_type="created",
            external_thread_id=None,
            occurred_at=None,
            created_by_id=actor.id,
        )
        assert citation is not None
        assert citation.memory_type == mtype
        assert citation.memory_id == memory_id


async def test_should_apply_update_no_prior_citations():
    from mirix.server.rest_api import _should_apply_update

    ok = await _should_apply_update(
        memory_type="episodic",
        memory_id=_unique("never-seen"),
        occurred_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    assert ok is True


async def test_should_apply_update_none_timestamp_allows():
    from mirix.server.rest_api import _should_apply_update

    ok = await _should_apply_update(
        memory_type="episodic",
        memory_id="anything",
        occurred_at=None,
    )
    assert ok is True


async def test_should_apply_update_newer_allows():
    from mirix.server.rest_api import (
        MemorySourceInput,
        _persist_source_with_messages,
        _should_apply_update,
        _write_citation_for_memory,
    )

    actor = await _get_actor()
    source, _ = await _persist_source_with_messages(
        memory_source_id=_unique("src"),
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=MemorySourceInput(external_id=_unique("ext-guard-newer")),
        fallback_occurred_at=None,
        filter_tags=None,
    )
    memory_id = _unique("ep-guard-newer")
    old = datetime(2025, 1, 1, tzinfo=timezone.utc)
    await _write_citation_for_memory(
        memory_source_id=source.id,
        memory_type="episodic",
        memory_id=memory_id,
        citation_type="created",
        external_thread_id=None,
        occurred_at=old,
        created_by_id=actor.id,
    )

    newer = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert (
        await _should_apply_update(memory_type="episodic", memory_id=memory_id, occurred_at=newer)
        is True
    )


async def test_should_apply_update_older_blocks():
    from mirix.server.rest_api import (
        MemorySourceInput,
        _persist_source_with_messages,
        _should_apply_update,
        _write_citation_for_memory,
    )

    actor = await _get_actor()
    source, _ = await _persist_source_with_messages(
        memory_source_id=_unique("src"),
        actor=actor,
        user_id=TEST_USER_ID,
        organization_id=TEST_ORG_ID,
        source_input=MemorySourceInput(external_id=_unique("ext-guard-older")),
        fallback_occurred_at=None,
        filter_tags=None,
    )
    memory_id = _unique("ep-guard-older")
    current = datetime(2026, 6, 1, tzinfo=timezone.utc)
    await _write_citation_for_memory(
        memory_source_id=source.id,
        memory_type="episodic",
        memory_id=memory_id,
        citation_type="created",
        external_thread_id=None,
        occurred_at=current,
        created_by_id=actor.id,
    )

    stale = datetime(2025, 12, 1, tzinfo=timezone.utc)
    assert (
        await _should_apply_update(memory_type="episodic", memory_id=memory_id, occurred_at=stale)
        is False
    )
