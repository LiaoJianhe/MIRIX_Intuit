"""
Sync-to-async bridge for Mirix.

Stores the application event loop and its thread so sync callers (e.g. queue
worker, ORM lifecycle) can run async code via run_coroutine_threadsafe without
depending on Redis. Used by cache provider sync wrappers and agent/memory
manager _sync_* methods.
"""

import asyncio
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Event loop and thread for sync callers using run_coroutine_threadsafe
_event_loop: Optional[asyncio.AbstractEventLoop] = None
# Thread that owns the event loop; sync bridge must not block on this thread
_event_loop_thread_id: Optional[int] = None


def get_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get the stored event loop (for sync callers using run_coroutine_threadsafe)."""
    return _event_loop


def get_event_loop_thread_id() -> Optional[int]:
    """Thread ID that owns the event loop; sync bridge must not block there."""
    return _event_loop_thread_id


def set_event_loop_for_sync_bridge(force: bool = False) -> None:
    """
    Store the current event loop and thread for sync bridges.

    Call this from async context during app startup (e.g. REST server
    initialize()). Sync wrappers (_sync_get_agent_by_id, _sync_list_*)
    need a loop to run async code; storing it here ensures they work
    with any registered cache provider, not only when Redis is enabled.

    When force=False, does not overwrite an already-stored loop so the
    queue worker and Redis stay on the same loop when a second init runs
    (e.g. rest_api.initialize() called again). When force=True (e.g. from
    initialize_redis_client() when creating the client), always set so the
    stored loop matches the loop the Redis client is created on.
    """
    global _event_loop, _event_loop_thread_id

    if not force and _event_loop is not None:
        return

    _event_loop = asyncio.get_running_loop()
    _event_loop_thread_id = threading.current_thread().ident


def run_sync(coro: Any, timeout: float = 1) -> Any:
    """
    Run a coroutine from sync context via the stored event loop.

    For use by manager _sync_* methods. Raises RuntimeError if no event
    loop is available or if called from the event loop thread (deadlock).
    Cancels the scheduled future on timeout and re-raises.

    Args:
        coro: Coroutine to run (e.g. from get_agent_by_id, list_agents).
        timeout: Max seconds to wait. Default 1.

    Returns:
        The coroutine result.

    Raises:
        RuntimeError: No loop, or called from event loop thread.
        TimeoutError: Coroutine did not complete within timeout.
    """
    loop = get_event_loop()
    if loop is None:
        raise RuntimeError("No event loop available for sync bridge")
    if (
        _event_loop_thread_id is not None
        and threading.current_thread().ident == _event_loop_thread_id
    ):
        raise RuntimeError(
            "run_sync() called from event loop thread — would deadlock"
        )
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except (TimeoutError, FuturesTimeoutError):
        fut.cancel()
        logger.warning(
            "run_sync timed out after %ss; future cancelled", timeout
        )
        raise


def run_sync_or_default(coro: Any, default: Any, timeout: float = 1) -> Any:
    """
    Run a coroutine from sync context; return default on deadlock/timeout.

    For use by cache provider and best-effort operations (e.g. cache
    invalidation). Does not raise on deadlock or timeout.

    Args:
        coro: Coroutine to run.
        default: Value to return on deadlock, timeout, or exception.
        timeout: Max seconds to wait. Default 1.

    Returns:
        The coroutine result, or default on failure.
    """
    loop = get_event_loop()
    if loop is None:
        return default
    if (
        _event_loop_thread_id is not None
        and threading.current_thread().ident == _event_loop_thread_id
    ):
        logger.warning(
            "run_sync_or_default skipped: called from event loop thread "
            "(would deadlock). Returning default."
        )
        return default
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except Exception as e:
        fut.cancel()
        logger.warning("run_sync_or_default failed after %ss: %s", timeout, e)
        return default


def clear_sync_bridge() -> None:
    """Clear the stored event loop and thread (e.g. on app shutdown)."""
    global _event_loop, _event_loop_thread_id
    _event_loop = None
    _event_loop_thread_id = None
