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
