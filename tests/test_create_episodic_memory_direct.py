"""Tests for create_episodic_memory_direct and the direct-memory helpers in rest_api.py."""

import uuid

import pytest
import pytest_asyncio


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def tmp_org():
    from mirix.schemas.organization import Organization as PydanticOrganization
    from mirix.services.organization_manager import OrganizationManager

    org_mgr = OrganizationManager()
    org_id = _unique("org")
    return await org_mgr.create_organization(
        PydanticOrganization(id=org_id, name="Direct Episodic Test Org")
    )


@pytest_asyncio.fixture
async def tmp_user(tmp_org):
    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    user_mgr = UserManager()
    return await user_mgr.create_user(
        PydanticUser(
            id=_unique("user"),
            name="Direct Episodic Test User",
            organization_id=tmp_org.id,
            timezone="UTC",
            status="active",
        )
    )


@pytest_asyncio.fixture
async def tmp_client_no_meta(tmp_org):
    """Create a fresh client with a write_scope and no meta agent registered."""
    from mirix.schemas.client import Client as PydanticClient
    from mirix.services.client_manager import ClientManager

    client_mgr = ClientManager()
    client_id = _unique("client")
    return await client_mgr.create_client(
        PydanticClient(
            id=client_id,
            name=f"No-Meta Test Client {client_id}",
            organization_id=tmp_org.id,
            write_scope="test-scope-a",
            read_scopes=["test-scope-a"],
        )
    )


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


@pytest.mark.asyncio
async def test_create_episodic_memory_direct_no_meta_agent(tmp_client_no_meta, tmp_user, tmp_org):
    """Happy path without a meta agent: uses create_episodic_memory.

    Asserts: exactly one episodic row, one source, zero source_messages, one citation.
    """
    from mirix.server.rest_api import (
        CreateEpisodicMemoryDirectRequest,
        MemorySourceInput,
        create_episodic_memory_direct,
    )
    from mirix.services.memory_citation_manager import MemoryCitationManager
    from mirix.services.memory_source_manager import MemorySourceManager
    from mirix.services.source_message_manager import SourceMessageManager

    req = CreateEpisodicMemoryDirectRequest(
        user_id=tmp_user.id,
        event_type="test_event",
        summary="summary text",
        details="details text",
        event_actor="system",
        filter_tags={"environment": "test"},
        occurred_at="2026-04-17T10:00:00Z",
        source=MemorySourceInput(external_id=_unique("ext-happy-no-meta")),
    )

    result = await create_episodic_memory_direct(
        client_id=tmp_client_no_meta.id,
        request=req,
    )

    assert result["success"] is True
    assert result["memory_source_id"].startswith("src-")
    assert result["citation_id"] is not None
    assert result["memory"]["id"] is not None
    assert result["memory"]["event_type"] == "test_event"

    source = await MemorySourceManager().get_by_id(result["memory_source_id"])
    assert source is not None
    assert source.external_id == req.source.external_id
    assert source.filter_tags.get("scope") == tmp_client_no_meta.write_scope

    msgs_page = await SourceMessageManager().get_messages_by_source_id(source.id)
    assert len(msgs_page.items) == 0

    citations = await MemoryCitationManager().get_citations_for_memory(
        memory_type="episodic", memory_id=result["memory"]["id"]
    )
    assert len(citations) == 1
    assert citations[0].memory_source_id == source.id
    assert citations[0].citation_type == "created"


@pytest.mark.asyncio
async def test_create_episodic_with_messages(tmp_client_no_meta, tmp_user, tmp_org):
    from mirix.server.rest_api import (
        CreateEpisodicMemoryDirectRequest,
        MemorySourceInput,
        SourceMessageInput,
        create_episodic_memory_direct,
    )
    from mirix.services.source_message_manager import SourceMessageManager

    req = CreateEpisodicMemoryDirectRequest(
        user_id=tmp_user.id,
        event_type="chat",
        summary="s",
        details="d",
        event_actor="user",
        occurred_at="2026-04-17T10:00:00Z",
        source=MemorySourceInput(
            external_id=_unique("ext-with-msgs"),
            messages=[
                SourceMessageInput(role="user", content="hi"),
                SourceMessageInput(role="assistant", content="hello"),
            ],
        ),
    )
    result = await create_episodic_memory_direct(client_id=tmp_client_no_meta.id, request=req)
    msgs_page = await SourceMessageManager().get_messages_by_source_id(result["memory_source_id"])
    assert len(msgs_page.items) == 2


@pytest.mark.asyncio
async def test_create_episodic_dedup_returns_same_source(tmp_client_no_meta, tmp_user, tmp_org):
    from mirix.server.rest_api import (
        CreateEpisodicMemoryDirectRequest,
        MemorySourceInput,
        create_episodic_memory_direct,
    )

    ext = _unique("ext-dedup")
    req = CreateEpisodicMemoryDirectRequest(
        user_id=tmp_user.id,
        event_type="e",
        summary="s",
        details="d",
        event_actor="system",
        occurred_at="2026-04-17T10:00:00Z",
        source=MemorySourceInput(external_id=ext),
    )
    r1 = await create_episodic_memory_direct(client_id=tmp_client_no_meta.id, request=req)
    r2 = await create_episodic_memory_direct(client_id=tmp_client_no_meta.id, request=req)
    assert r1["memory_source_id"] == r2["memory_source_id"]
    assert r2["deduped"] is True
    assert r2["memory"] is None
    assert r2["citation_id"] is None


@pytest.mark.asyncio
async def test_create_episodic_scope_injection_overrides_client_tags(tmp_client_no_meta, tmp_user, tmp_org):
    from mirix.server.rest_api import (
        CreateEpisodicMemoryDirectRequest,
        MemorySourceInput,
        create_episodic_memory_direct,
    )
    from mirix.services.memory_source_manager import MemorySourceManager

    req = CreateEpisodicMemoryDirectRequest(
        user_id=tmp_user.id,
        event_type="e",
        summary="s",
        details="d",
        event_actor="system",
        filter_tags={"scope": "attacker-scope", "other": "ok"},
        occurred_at="2026-04-17T10:00:00Z",
        source=MemorySourceInput(external_id=_unique("ext-scope")),
    )
    r = await create_episodic_memory_direct(client_id=tmp_client_no_meta.id, request=req)
    src = await MemorySourceManager().get_by_id(r["memory_source_id"])
    assert src.filter_tags["scope"] == tmp_client_no_meta.write_scope
    assert src.filter_tags["scope"] != "attacker-scope"
    assert src.filter_tags.get("other") == "ok"


@pytest.mark.asyncio
async def test_create_episodic_source_type_default(tmp_client_no_meta, tmp_user, tmp_org):
    from mirix.server.rest_api import (
        CreateEpisodicMemoryDirectRequest,
        MemorySourceInput,
        create_episodic_memory_direct,
    )
    from mirix.services.memory_source_manager import MemorySourceManager

    req = CreateEpisodicMemoryDirectRequest(
        user_id=tmp_user.id,
        event_type="e",
        summary="s",
        details="d",
        event_actor="system",
        source=MemorySourceInput(external_id=_unique("ext-default-type")),
        occurred_at="2026-04-17T10:00:00Z",
    )
    r = await create_episodic_memory_direct(client_id=tmp_client_no_meta.id, request=req)
    src = await MemorySourceManager().get_by_id(r["memory_source_id"])
    assert src.source_type == "conversation"


@pytest.mark.asyncio
async def test_create_episodic_source_occurred_at_fallback(tmp_client_no_meta, tmp_user, tmp_org):
    from mirix.server.rest_api import (
        CreateEpisodicMemoryDirectRequest,
        MemorySourceInput,
        create_episodic_memory_direct,
    )
    from mirix.services.memory_source_manager import MemorySourceManager

    req = CreateEpisodicMemoryDirectRequest(
        user_id=tmp_user.id,
        event_type="e",
        summary="s",
        details="d",
        event_actor="system",
        occurred_at="2026-02-14T00:00:00Z",
        source=MemorySourceInput(external_id=_unique("ext-fallback")),
    )
    r = await create_episodic_memory_direct(client_id=tmp_client_no_meta.id, request=req)
    src = await MemorySourceManager().get_by_id(r["memory_source_id"])
    assert src.occurred_at is not None
    assert src.occurred_at.year == 2026
    assert src.occurred_at.month == 2


@pytest.mark.asyncio
async def test_create_episodic_no_write_scope_rejects(tmp_org, tmp_user):
    from fastapi import HTTPException

    from mirix.schemas.client import Client as PydanticClient
    from mirix.server.rest_api import (
        CreateEpisodicMemoryDirectRequest,
        MemorySourceInput,
        create_episodic_memory_direct,
    )
    from mirix.services.client_manager import ClientManager

    client = await ClientManager().create_client(
        PydanticClient(
            id=_unique("client-no-scope"),
            name="No Scope Client",
            organization_id=tmp_org.id,
            write_scope=None,
            read_scopes=[],
        )
    )

    req = CreateEpisodicMemoryDirectRequest(
        user_id=tmp_user.id,
        event_type="e",
        summary="s",
        details="d",
        event_actor="system",
        source=MemorySourceInput(external_id=_unique("ext-no-scope")),
        occurred_at="2026-04-17T00:00:00Z",
    )
    with pytest.raises(HTTPException) as exc:
        await create_episodic_memory_direct(client_id=client.id, request=req)
    assert exc.value.status_code == 403
