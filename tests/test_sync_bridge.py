"""
Tests for sync_bridge run_sync and run_sync_or_default.

Covers deadlock guard, timeout with future cancellation, and normal operation.
"""

import asyncio
import threading
import time

import pytest

from mirix.database.sync_bridge import (
    clear_sync_bridge,
    get_event_loop,
    run_sync,
    run_sync_or_default,
    set_event_loop_for_sync_bridge,
)


def test_run_sync_no_loop_raises():
    """run_sync raises RuntimeError when no event loop is stored."""
    clear_sync_bridge()
    with pytest.raises(RuntimeError, match="No event loop available"):
        run_sync(asyncio.sleep(0), timeout=1)


def test_run_sync_on_event_loop_thread_raises():
    """run_sync raises RuntimeError when called from the event loop thread (deadlock guard)."""
    result = {"error": None}

    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def set_bridge():
            set_event_loop_for_sync_bridge()

        def from_loop_thread():
            try:
                run_sync(asyncio.sleep(0), timeout=1)
            except RuntimeError as e:
                result["error"] = e
            finally:
                loop.stop()

        loop.call_soon(set_bridge)
        loop.call_soon(from_loop_thread)
        try:
            loop.run_forever()
        finally:
            loop.close()
            clear_sync_bridge()

    thread = threading.Thread(target=run_loop)
    thread.start()
    thread.join(timeout=2)
    assert result["error"] is not None
    assert "event loop thread" in str(result["error"])


def test_run_sync_normal_operation():
    """run_sync returns coroutine result when called from a different thread."""
    result = {"ready": False}

    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def set_bridge():
            set_event_loop_for_sync_bridge()
            result["ready"] = True

        loop.call_soon(set_bridge)
        try:
            loop.run_forever()
        finally:
            loop.close()
            clear_sync_bridge()

    thread = threading.Thread(target=run_loop)
    thread.start()
    while not result["ready"]:
        time.sleep(0.05)
    try:
        async def forty_two():
            return 42

        out = run_sync(forty_two(), timeout=5)
        assert out == 42
    finally:
        get_event_loop().call_soon_threadsafe(get_event_loop().stop)
        thread.join(timeout=2)


def test_run_sync_or_default_no_loop_returns_default():
    """run_sync_or_default returns default when no loop is stored."""
    clear_sync_bridge()
    out = run_sync_or_default(asyncio.sleep(0), default="fallback", timeout=1)
    assert out == "fallback"


def test_run_sync_or_default_on_event_loop_thread_returns_default():
    """run_sync_or_default returns default when called from event loop thread."""
    result = {"out": None}

    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def set_bridge():
            set_event_loop_for_sync_bridge()

        def from_loop_thread():
            result["out"] = run_sync_or_default(asyncio.sleep(0), default=99, timeout=1)
            loop.stop()

        loop.call_soon(set_bridge)
        loop.call_soon(from_loop_thread)
        try:
            loop.run_forever()
        finally:
            loop.close()
            clear_sync_bridge()

    thread = threading.Thread(target=run_loop)
    thread.start()
    thread.join(timeout=2)
    assert result["out"] == 99


def test_run_sync_timeout_cancels_future():
    """run_sync raises TimeoutError and cancels the future on timeout."""
    result = {"ready": False}

    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def set_bridge():
            set_event_loop_for_sync_bridge()
            result["ready"] = True

        loop.call_soon(set_bridge)
        try:
            loop.run_forever()
        finally:
            loop.close()
            clear_sync_bridge()

    thread = threading.Thread(target=run_loop)
    thread.start()
    while not result["ready"]:
        time.sleep(0.05)
    try:
        async def slow():
            await asyncio.sleep(10)

        with pytest.raises(TimeoutError):
            run_sync(slow(), timeout=0.1)
    finally:
        get_event_loop().call_soon_threadsafe(get_event_loop().stop)
        thread.join(timeout=2)


def test_run_sync_or_default_timeout_returns_default():
    """run_sync_or_default returns default on timeout."""
    result = {"ready": False}

    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def set_bridge():
            set_event_loop_for_sync_bridge()
            result["ready"] = True

        loop.call_soon(set_bridge)
        try:
            loop.run_forever()
        finally:
            loop.close()
            clear_sync_bridge()

    thread = threading.Thread(target=run_loop)
    thread.start()
    while not result["ready"]:
        time.sleep(0.05)
    try:
        async def slow():
            await asyncio.sleep(10)

        out = run_sync_or_default(slow(), default="timeout_default", timeout=0.1)
        assert out == "timeout_default"
    finally:
        get_event_loop().call_soon_threadsafe(get_event_loop().stop)
        thread.join(timeout=2)
