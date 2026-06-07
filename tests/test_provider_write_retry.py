"""Unit tests for the MIRIX provider write retry helper.

After VEPAGE-1251 §5.6 the helper is pure isinstance against MIRIX-owned
types — the ECMS provider boundary translates SDK exceptions into
`ProviderTransientError` / `ProviderPermanentError` / `ProviderConflictError`
before they reach this code. No structural detection lives here anymore.

Tests that exercised the legacy structural detection (status_code parsing,
`uq_*` constraint-name parsing, `BadRequestError` ambiguity) now live in the
ECMS-side `app/tests/unit/ipsr/test_sdk_exception_translation.py` against
the translator that produces these types.

Run:
    pytest tests/test_provider_write_retry.py -v
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from mirix.database.provider_write_retry import (
    is_conflict,
    is_transient,
    retry_transient,
)
from mirix.errors import (
    ProviderConflictError,
    ProviderPermanentError,
    ProviderTransientError,
)


class TestIsConflict:
    """is_conflict is pure isinstance against ProviderConflictError."""

    def test_provider_conflict_error_is_conflict(self):
        assert is_conflict(ProviderConflictError("uq_email")) is True

    def test_provider_permanent_error_is_not_conflict(self):
        assert is_conflict(ProviderPermanentError("400 bad shape")) is False

    def test_provider_transient_error_is_not_conflict(self):
        assert is_conflict(ProviderTransientError("503")) is False

    def test_arbitrary_exception_is_not_conflict(self):
        assert is_conflict(Exception("unique constraint violation")) is False

    def test_arbitrary_exception_with_409_is_not_conflict(self):
        """The legacy structural-409 detection is gone. The provider
        boundary translates 409 to ProviderConflictError; raw status codes
        no longer match."""

        class _StatusError(Exception):
            status_code = 409

        assert is_conflict(_StatusError()) is False


class TestIsTransient:
    """is_transient is pure isinstance against ProviderTransientError."""

    def test_provider_transient_error_is_transient(self):
        assert is_transient(ProviderTransientError("503 from IPS-R")) is True

    def test_provider_permanent_error_is_not_transient(self):
        assert is_transient(ProviderPermanentError("400")) is False

    def test_provider_conflict_error_is_not_transient(self):
        """Conflict short-circuits to False — must not be retried."""
        assert is_transient(ProviderConflictError("uq_email")) is False

    def test_arbitrary_status_503_is_not_transient(self):
        """Legacy structural detection is gone."""

        class _StatusError(Exception):
            status_code = 503

        assert is_transient(_StatusError()) is False

    def test_asyncio_timeout_is_not_transient(self):
        """Legacy asyncio.TimeoutError → transient is gone — the provider
        boundary translates timeouts into ProviderTransientError."""
        assert is_transient(asyncio.TimeoutError()) is False

    def test_provider_conflict_takes_precedence_over_transient_check(self):
        """A ProviderConflictError must NOT be retried even if its message
        contains words that would otherwise look transient."""
        exc = ProviderConflictError("Timeout while writing duplicate row")
        assert is_conflict(exc) is True
        assert is_transient(exc) is False


class TestRetryTransient:
    @pytest.mark.asyncio
    async def test_returns_value_on_success(self):
        coro = AsyncMock(return_value="ok")
        result = await retry_transient(coro, op="test")
        assert result == "ok"
        assert coro.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        coro = AsyncMock(side_effect=[ProviderTransientError("503"), "ok"])
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await retry_transient(coro, op="test", max_attempts=3)
        assert result == "ok"
        assert coro.await_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_raises(self):
        coro = AsyncMock(side_effect=ProviderTransientError("503"))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ProviderTransientError):
                await retry_transient(coro, op="test", max_attempts=3)
        assert coro.await_count == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_on_conflict(self):
        coro = AsyncMock(side_effect=ProviderConflictError("uq_email"))
        with pytest.raises(ProviderConflictError):
            await retry_transient(coro, op="test", max_attempts=3)
        assert coro.await_count == 1

    @pytest.mark.asyncio
    async def test_does_not_retry_on_permanent_error(self):
        coro = AsyncMock(side_effect=ProviderPermanentError("401"))
        with pytest.raises(ProviderPermanentError):
            await retry_transient(coro, op="test", max_attempts=3)
        assert coro.await_count == 1


class TestInnerExhaustedMarker:
    """retry_transient tags the propagated exception when its budget is
    exhausted so process_with_policy skips the whole-step retry."""

    @pytest.mark.asyncio
    async def test_exhausted_transient_is_tagged_inner_exhausted(self):
        from mirix.queue.error_policy import is_inner_exhausted

        coro = AsyncMock(side_effect=ProviderTransientError("503"))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ProviderTransientError) as excinfo:
                await retry_transient(coro, op="test", max_attempts=3)

        assert coro.await_count == 3
        assert is_inner_exhausted(excinfo.value) is True, (
            "exhausted transient must be tagged so the whole-step loop "
            "doesn't add another 3 retries on top"
        )

    @pytest.mark.asyncio
    async def test_permanent_error_not_tagged(self):
        """Permanent errors don't go through the retry loop, so no marker
        is applied (and none is needed — permanent short-circuits in
        process_with_policy regardless)."""
        from mirix.queue.error_policy import is_inner_exhausted

        coro = AsyncMock(side_effect=ProviderPermanentError("401"))
        with pytest.raises(ProviderPermanentError) as excinfo:
            await retry_transient(coro, op="test", max_attempts=3)
        assert is_inner_exhausted(excinfo.value) is False

    @pytest.mark.asyncio
    async def test_pre_tagged_transient_short_circuits_retry(self):
        """If the exception arrives already inner-exhausted (e.g. from a
        deeper retry tier like the provider's own _retry_transient), this
        helper does NOT retry — it propagates immediately. Prevents the
        N×M stacking multiplication when retry tiers stack."""
        from mirix.queue.error_policy import is_inner_exhausted, mark_inner_exhausted

        pre_tagged = ProviderTransientError("already exhausted by inner tier")
        mark_inner_exhausted(pre_tagged)

        coro = AsyncMock(side_effect=pre_tagged)
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ProviderTransientError) as excinfo:
                await retry_transient(coro, op="test", max_attempts=3)

        # Single attempt — no stacking.
        assert coro.await_count == 1
        assert is_inner_exhausted(excinfo.value) is True
