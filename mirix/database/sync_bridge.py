"""
Sync-to-async bridge for Mirix.

Stores the application event loop and its thread so sync callers (e.g. queue
worker, ORM lifecycle) can run async code via run_coroutine_threadsafe without
depending on Redis. Used by cache provider sync wrappers and agent/memory
manager _sync_* methods.
"""

import asyncio
from typing import Optional

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


def set_event_loop_for_sync_bridge() -> None:
    """
    Store the current event loop and thread for sync bridges.

    Call this from async context during app startup (e.g. REST server
    initialize()). Sync wrappers (_sync_get_agent_by_id, _sync_list_*)
    need a loop to run async code; storing it here ensures they work
    with any registered cache provider, not only when Redis is enabled.
    """
    global _event_loop, _event_loop_thread_id
    import threading

    _event_loop = asyncio.get_running_loop()
    _event_loop_thread_id = threading.current_thread().ident


def clear_sync_bridge() -> None:
    """Clear the stored event loop and thread (e.g. on app shutdown)."""
    global _event_loop, _event_loop_thread_id
    _event_loop = None
    _event_loop_thread_id = None
