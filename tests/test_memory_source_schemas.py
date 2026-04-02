"""Unit tests for memory source schemas, enums, and ORM models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mirix.schemas.enums import CitationType, MemoryType, SummarySource


class TestEnums:
    def test_summary_source_values(self):
        assert SummarySource.client == "client"
        assert SummarySource.generated == "generated"
        assert len(SummarySource) == 2

    def test_citation_type_values(self):
        assert CitationType.created == "created"
        assert CitationType.updated == "updated"
        assert len(CitationType) == 2

    def test_memory_type_values(self):
        assert MemoryType.episodic == "episodic"
        assert MemoryType.semantic == "semantic"
        assert MemoryType.procedural == "procedural"
        assert MemoryType.resource == "resource"
        assert MemoryType.knowledge_vault == "knowledge_vault"
        assert MemoryType.core == "core"
        assert len(MemoryType) == 6

    def test_enums_are_string_enums(self):
        """All new enums should be str subclasses for JSON serialization."""
        assert isinstance(SummarySource.client, str)
        assert isinstance(CitationType.created, str)
        assert isinstance(MemoryType.episodic, str)


# --- MemorySource schema tests ---

from mirix.schemas.memory_source import MemorySource, MemorySourceUpdate


class TestMemorySourceSchema:
    def test_id_prefix(self):
        ms = MemorySource(
            id="src-00000000-0000-0000-0000-000000000000",
            client_id="client-123",
            user_id="user-123",
            organization_id="org-123",
            source_type="conversation",
        )
        assert ms.id.startswith("src-")

    def test_defaults(self):
        ms = MemorySource(
            id="src-00000000-0000-0000-0000-000000000000",
            client_id="client-123",
            user_id="user-123",
            organization_id="org-123",
        )
        assert ms.source_type == "conversation"
        assert ms.processing_complete is False
        assert ms.external_id is None
        assert ms.external_thread_id is None
        assert ms.source_system is None
        assert ms.source_metadata is None
        assert ms.summary is None
        assert ms.summary_source is None
        assert ms.batch_hash is None
        assert ms.occurred_at is None

    def test_all_fields(self):
        now = datetime.now(timezone.utc)
        ms = MemorySource(
            id="src-00000000-0000-0000-0000-000000000000",
            client_id="client-123",
            user_id="user-123",
            organization_id="org-123",
            external_id="ext-abc",
            external_thread_id="thread-xyz",
            source_type="domain_event",
            source_system="slack",
            source_metadata={"channel": "#general"},
            occurred_at=now,
            summary="Test summary",
            summary_source="client",
            processing_complete=True,
            batch_hash="abc123hash",
        )
        assert ms.external_id == "ext-abc"
        assert ms.source_metadata == {"channel": "#general"}
        assert ms.summary_source == "client"
        assert ms.processing_complete is True

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            MemorySource(
                id="src-00000000-0000-0000-0000-000000000000",
                client_id="client-123",
                user_id="user-123",
                organization_id="org-123",
                not_a_field="bad",
            )

    def test_generate_id(self):
        generated = MemorySource._generate_id()
        assert generated.startswith("src-")

    def test_update_schema_optional_fields(self):
        update = MemorySourceUpdate(id="src-00000000-0000-0000-0000-000000000000")
        assert update.id.startswith("src-")
        assert update.summary is None
        assert update.processing_complete is None


# --- SourceMessage schema tests ---

from mirix.schemas.source_message import SourceMessage as SourceMessageSchema


class TestSourceMessageSchema:
    def test_id_prefix(self):
        sm = SourceMessageSchema(
            id="smsg-00000000-0000-0000-0000-000000000000",
            memory_source_id="src-123",
            role="user",
            content={"text": "hello"},
            sequence_num=0,
            content_hash="abc123",
        )
        assert sm.id.startswith("smsg-")

    def test_defaults(self):
        sm = SourceMessageSchema(
            id="smsg-00000000-0000-0000-0000-000000000000",
            memory_source_id="src-123",
            role="user",
            content={"text": "hello"},
            sequence_num=0,
            content_hash="abc123",
        )
        assert sm.external_thread_id is None
        assert sm.external_message_id is None
        assert sm.occurred_at is None

    def test_all_fields(self):
        now = datetime.now(timezone.utc)
        sm = SourceMessageSchema(
            id="smsg-00000000-0000-0000-0000-000000000000",
            memory_source_id="src-123",
            external_thread_id="thread-xyz",
            external_message_id="ext-msg-1",
            role="assistant",
            content={"text": "response"},
            occurred_at=now,
            sequence_num=1,
            content_hash="def456",
        )
        assert sm.external_message_id == "ext-msg-1"
        assert sm.sequence_num == 1

    def test_generate_id(self):
        generated = SourceMessageSchema._generate_id()
        assert generated.startswith("smsg-")


# --- MemoryCitation schema tests ---

from mirix.schemas.memory_citation import MemoryCitation as MemoryCitationSchema


class TestMemoryCitationSchema:
    def test_id_prefix(self):
        mc = MemoryCitationSchema(
            id="cit-00000000-0000-0000-0000-000000000000",
            memory_type="episodic",
            memory_id="ep_mem-123",
            memory_source_id="src-123",
            citation_type="created",
        )
        assert mc.id.startswith("cit-")

    def test_defaults(self):
        mc = MemoryCitationSchema(
            id="cit-00000000-0000-0000-0000-000000000000",
            memory_type="episodic",
            memory_id="ep_mem-123",
            memory_source_id="src-123",
            citation_type="created",
        )
        assert mc.external_thread_id is None
        assert mc.occurred_at is None
        assert mc.message_ids is None

    def test_all_fields(self):
        now = datetime.now(timezone.utc)
        mc = MemoryCitationSchema(
            id="cit-00000000-0000-0000-0000-000000000000",
            memory_type="semantic",
            memory_id="sem-456",
            memory_source_id="src-789",
            external_thread_id="thread-abc",
            occurred_at=now,
            citation_type="updated",
            message_ids=["smsg-1", "smsg-2"],
        )
        assert mc.memory_type == "semantic"
        assert mc.citation_type == "updated"
        assert mc.message_ids == ["smsg-1", "smsg-2"]

    def test_generate_id(self):
        generated = MemoryCitationSchema._generate_id()
        assert generated.startswith("cit-")

    def test_memory_type_validates_all_types(self):
        """Ensure all 6 memory types can be used."""
        for mt in ["episodic", "semantic", "procedural", "resource", "knowledge_vault", "core"]:
            mc = MemoryCitationSchema(
                id="cit-00000000-0000-0000-0000-000000000000",
                memory_type=mt,
                memory_id="mem-123",
                memory_source_id="src-123",
                citation_type="created",
            )
            assert mc.memory_type == mt


# --- ORM model tests ---

from mirix.orm.memory_citation import MemoryCitation as MemoryCitationORM
from mirix.orm.memory_source import MemorySource as MemorySourceORM
from mirix.orm.source_message import SourceMessage as SourceMessageORM


class TestORMModels:
    def test_memory_source_tablename(self):
        assert MemorySourceORM.__tablename__ == "memory_sources"

    def test_source_message_tablename(self):
        assert SourceMessageORM.__tablename__ == "source_messages"

    def test_memory_citation_tablename(self):
        assert MemoryCitationORM.__tablename__ == "memory_citations"

    def test_memory_source_pydantic_model(self):
        from mirix.schemas.memory_source import MemorySource as PydanticMemorySource

        assert MemorySourceORM.__pydantic_model__ is PydanticMemorySource

    def test_source_message_pydantic_model(self):
        from mirix.schemas.source_message import SourceMessage as PydanticSourceMessage

        assert SourceMessageORM.__pydantic_model__ is PydanticSourceMessage

    def test_memory_citation_pydantic_model(self):
        from mirix.schemas.memory_citation import MemoryCitation as PydanticMemoryCitation

        assert MemoryCitationORM.__pydantic_model__ is PydanticMemoryCitation

    def test_memory_source_has_org_and_user_mixins(self):
        """MemorySource should have organization_id and user_id from mixins."""
        assert hasattr(MemorySourceORM, "organization_id")
        assert hasattr(MemorySourceORM, "user_id")

    def test_source_message_has_memory_source_id(self):
        """SourceMessage is scoped through parent MemorySource."""
        assert hasattr(SourceMessageORM, "created_at")
        assert hasattr(SourceMessageORM, "memory_source_id")

    def test_memory_citation_has_memory_source_id(self):
        """MemoryCitation is scoped through parent MemorySource."""
        assert hasattr(MemoryCitationORM, "created_at")
        assert hasattr(MemoryCitationORM, "memory_source_id")

    def test_all_models_registered(self):
        """All 3 new models should be importable from mirix.orm."""
        from mirix.orm import MemoryCitation, MemorySource, SourceMessage

        assert MemorySource.__tablename__ == "memory_sources"
        assert SourceMessage.__tablename__ == "source_messages"
        assert MemoryCitation.__tablename__ == "memory_citations"
