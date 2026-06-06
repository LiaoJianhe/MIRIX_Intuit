"""Unit tests for the structural exception classifier + retry helper.

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


class _StatusCodeError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


class _ConnectionTimeoutError(Exception):
    """Class-name-based detection target."""


class _ThrottleError(Exception):
    pass


class _UnauthorizedError(Exception):
    def __init__(self):
        super().__init__("HTTP 401")
        self.status_code = 401


class _IpsrBadRequestError(Exception):
    """Mirrors ipsrclientsdkpython.exceptions.BadRequestError: carries an
    ``error_code`` and message but NO ``status_code`` attribute."""

    def __init__(self, message: str, error_code: str = "DATABASE_CONSTRAINT_VIOLATION"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class TestIsConflict:
    def test_409_is_conflict(self):
        assert is_conflict(_StatusCodeError(409))

    def test_message_with_unique_is_conflict(self):
        assert is_conflict(Exception("unique constraint violation"))

    def test_message_with_duplicate_is_conflict(self):
        assert is_conflict(Exception("duplicate key value"))

    def test_500_without_hint_is_not_conflict(self):
        assert not is_conflict(_StatusCodeError(500, "internal server error"))

    def test_400_without_hint_is_not_conflict(self):
        assert not is_conflict(_StatusCodeError(400, "bad request"))

    def test_ipsr_unique_constraint_violation_is_conflict(self):
        # Real provider unique-index violation: a BadRequestError with no
        # status_code, an ambiguous DATABASE_CONSTRAINT_VIOLATION error_code,
        # and the constraint name in the message. Must classify as a conflict so
        # the caller dedups instead of crashing.
        exc = _IpsrBadRequestError("Client data violates a database constraint:  uq_memory_sources_ext_id")
        assert is_conflict(exc)

    def test_ipsr_batch_hash_constraint_violation_is_conflict(self):
        exc = _IpsrBadRequestError("Client data violates a database constraint:  uq_memory_sources_batch")
        assert is_conflict(exc)

    def test_ipsr_source_message_constraint_violation_is_conflict(self):
        exc = _IpsrBadRequestError("Client data violates a database constraint:  uq_source_messages_ext_id")
        assert is_conflict(exc)

    def test_ipsr_primary_key_violation_is_conflict(self):
        # Primary-key collision. The provider names PK constraints
        # ``<table>_pkey`` (e.g. users_pkey), so the uq_-only matcher missed it
        # and unhandled BadRequestError crashed admin-user creation on startup.
        exc = _IpsrBadRequestError("Client data violates a database constraint:  users_pkey")
        assert is_conflict(exc)

    def test_ipsr_column_shape_mismatch_is_not_conflict(self):
        # Same DATABASE_CONSTRAINT_VIOLATION error_code is also raised for a
        # column-shape mismatch, which is a permanent error, NOT a conflict. It
        # carries no uq_ index name, so it must NOT be classified as a conflict.
        exc = _IpsrBadRequestError("Client data violates a database constraint:  1, number of columns: 0")
        assert not is_conflict(exc)


class TestIsTransient:
    def test_503_is_transient(self):
        assert is_transient(_StatusCodeError(503))

    def test_429_is_transient(self):
        assert is_transient(_StatusCodeError(429))

    def test_connection_timeout_class_is_transient(self):
        assert is_transient(_ConnectionTimeoutError())

    def test_throttle_class_is_transient(self):
        assert is_transient(_ThrottleError())

    def test_asyncio_timeout_is_transient(self):
        assert is_transient(asyncio.TimeoutError())

    def test_401_is_not_transient(self):
        assert not is_transient(_UnauthorizedError())

    def test_400_is_not_transient(self):
        assert not is_transient(_StatusCodeError(400))

    def test_409_is_not_transient(self):
        assert not is_transient(_StatusCodeError(409))


class TestRetryTransient:
    @pytest.mark.asyncio
    async def test_returns_value_on_success(self):
        coro = AsyncMock(return_value="ok")
        result = await retry_transient(coro, op="test")
        assert result == "ok"
        assert coro.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        coro = AsyncMock(side_effect=[_StatusCodeError(503), "ok"])
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await retry_transient(coro, op="test", max_attempts=3)
        assert result == "ok"
        assert coro.await_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_raises(self):
        coro = AsyncMock(side_effect=_StatusCodeError(503))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(_StatusCodeError):
                await retry_transient(coro, op="test", max_attempts=3)
        assert coro.await_count == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_on_conflict(self):
        coro = AsyncMock(side_effect=Exception("unique constraint violation"))
        with pytest.raises(Exception, match="unique"):
            await retry_transient(coro, op="test", max_attempts=3)
        assert coro.await_count == 1

    @pytest.mark.asyncio
    async def test_does_not_retry_on_permanent_error(self):
        coro = AsyncMock(side_effect=_UnauthorizedError())
        with pytest.raises(_UnauthorizedError):
            await retry_transient(coro, op="test", max_attempts=3)
        assert coro.await_count == 1


class TestProviderTypeIntegration:
    """S3 of VEPAGE-1251: Provider* types from the ECMS boundary translation
    are classified by isinstance (preferred path)."""

    def test_provider_conflict_error_is_conflict(self):
        from mirix.errors import ProviderConflictError

        assert is_conflict(ProviderConflictError("uq_email")) is True

    def test_provider_transient_error_is_transient(self):
        from mirix.errors import ProviderTransientError

        assert is_transient(ProviderTransientError("503 from IPS-R")) is True

    def test_provider_permanent_error_is_not_transient_or_conflict(self):
        from mirix.errors import ProviderPermanentError

        exc = ProviderPermanentError("400 bad shape")
        assert is_transient(exc) is False
        assert is_conflict(exc) is False

    def test_provider_conflict_takes_precedence_over_transient_check(self):
        """A ProviderConflictError must NOT be retried even if its message
        contains words that would otherwise look transient."""
        from mirix.errors import ProviderConflictError

        exc = ProviderConflictError("Timeout while writing duplicate row")
        assert is_conflict(exc) is True
        # is_transient() returns False because Conflict short-circuits in
        # the isinstance branch.
        assert is_transient(exc) is False


class TestInnerExhaustedMarker:
    """S3 of VEPAGE-1251: retry_transient tags the propagated exception
    when its budget is exhausted so process_with_policy skips the
    whole-step retry."""

    @pytest.mark.asyncio
    async def test_exhausted_transient_is_tagged_inner_exhausted(self):
        from mirix.errors import ProviderTransientError
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
    async def test_legacy_transient_also_tagged_inner_exhausted(self):
        """The marker must also be applied to structurally-detected
        transients (during the boundary-translation transition)."""
        from mirix.queue.error_policy import is_inner_exhausted

        coro = AsyncMock(side_effect=_StatusCodeError(503))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(_StatusCodeError) as excinfo:
                await retry_transient(coro, op="test", max_attempts=3)

        assert is_inner_exhausted(excinfo.value) is True

    @pytest.mark.asyncio
    async def test_permanent_error_not_tagged(self):
        """Permanent errors don't go through the retry loop, so no marker
        is applied (and none is needed — permanent short-circuits in
        process_with_policy regardless)."""
        from mirix.queue.error_policy import is_inner_exhausted

        coro = AsyncMock(side_effect=_UnauthorizedError())
        with pytest.raises(_UnauthorizedError) as excinfo:
            await retry_transient(coro, op="test", max_attempts=3)
        assert is_inner_exhausted(excinfo.value) is False
