"""Naming consistency for failure-event spans (VEPAGE-1245).

Two failure-event spans previously used sentence case ("Citation write failed",
"Source message write failed") while sibling skip events use the
"Category: detail" convention (e.g. "Idempotency Skip: processing complete").
These tests pin the failure-event spans to the aligned names:

- "Citation Write: failed"
- "Source Message Write: failed"

They drive the real transient-error branch in each manager (provider create()
raising a transient error that survives retries) and capture the name passed to
``emit_idempotency_skip_span``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.services.memory_citation_manager import MemoryCitationManager
from mirix.services.source_message_manager import SourceMessageManager


class _Transient(Exception):
    """Stand-in transient DB error."""


@pytest.mark.asyncio
async def test_citation_write_failed_span_uses_aligned_name():
    manager = object.__new__(MemoryCitationManager)

    provider = MagicMock()
    provider.create = AsyncMock(side_effect=_Transient("boom"))

    captured = {}

    def _capture(name, reason, metadata=None):
        captured["name"] = name

    async def _retry(fn, op=None):
        # Exhaust retries: invoke the op once and let the transient error escape.
        return await fn()

    with (
        patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=provider,
        ),
        patch("mirix.database.provider_write_retry.retry_transient", side_effect=_retry),
        patch("mirix.database.provider_write_retry.is_conflict", return_value=False),
        patch("mirix.database.provider_write_retry.is_transient", return_value=True),
        patch(
            "mirix.observability.skip_spans.emit_idempotency_skip_span",
            side_effect=_capture,
        ),
    ):
        result = await manager.create(
            memory_source_id="ms-1",
            memory_type="episodic",
            memory_id="mem-1",
            citation_type="created",
        )

    assert result is None  # write must stand; failure swallowed
    assert captured["name"] == "Citation Write: failed"


@pytest.mark.asyncio
async def test_source_message_write_failed_span_uses_aligned_name():
    manager = object.__new__(SourceMessageManager)

    provider = MagicMock()
    provider.create = AsyncMock(side_effect=_Transient("boom"))

    captured = {}

    def _capture(name, reason, metadata=None):
        captured["name"] = name

    async def _retry(fn, op=None):
        return await fn()

    with (
        patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=provider,
        ),
        patch("mirix.database.provider_write_retry.retry_transient", side_effect=_retry),
        patch("mirix.database.provider_write_retry.is_conflict", return_value=False),
        patch("mirix.database.provider_write_retry.is_transient", return_value=True),
        patch(
            "mirix.observability.skip_spans.emit_idempotency_skip_span",
            side_effect=_capture,
        ),
    ):
        inserted = await manager.bulk_insert(
            messages=[{"role": "user", "content": "hello"}],
            memory_source_id="ms-1",
        )

    assert inserted == 0  # transient row skipped, not raised
    assert captured["name"] == "Source Message Write: failed"
