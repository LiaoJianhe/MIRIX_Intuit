"""
Tests for mirix.database.call_context (call origin + hybrid window).

Usage:
    pytest tests/test_call_context.py -v
"""

import asyncio
import contextvars

import pytest

from mirix.database import call_context
from mirix.database.call_context import (
    CALL_ORIGIN_CLIENT_API,
    CALL_ORIGIN_ENGINE,
    get_call_origin,
    get_hybrid_window_seconds,
    set_call_origin,
    set_hybrid_window_seconds,
)


@pytest.fixture(autouse=True)
def reset_call_origin():
    set_call_origin(CALL_ORIGIN_ENGINE)
    yield
    set_call_origin(CALL_ORIGIN_ENGINE)


@pytest.fixture(autouse=True)
def restore_hybrid_window():
    original = get_hybrid_window_seconds()
    yield
    set_hybrid_window_seconds(original)


class TestCallOrigin:
    def test_default_call_origin_is_engine(self):
        """ContextVar default is CALL_ORIGIN_ENGINE when no value is set in a context."""

        def read_default():
            return call_context.call_origin_var.get()

        assert contextvars.Context().run(read_default) == CALL_ORIGIN_ENGINE

    def test_set_and_get_call_origin(self):
        set_call_origin(CALL_ORIGIN_CLIENT_API)
        assert get_call_origin() == CALL_ORIGIN_CLIENT_API


class TestHybridWindowSeconds:
    def test_set_and_get_hybrid_window_seconds(self):
        set_hybrid_window_seconds(42)
        assert get_hybrid_window_seconds() == 42


class TestContextVarTaskIsolation:
    @pytest.mark.asyncio
    async def test_call_origin_isolated_between_tasks(self):
        set_call_origin(CALL_ORIGIN_ENGINE)

        async def child():
            set_call_origin(CALL_ORIGIN_CLIENT_API)
            assert get_call_origin() == CALL_ORIGIN_CLIENT_API

        await asyncio.create_task(child())
        assert get_call_origin() == CALL_ORIGIN_ENGINE
