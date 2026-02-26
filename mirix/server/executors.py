"""
Named thread pool executors for REST API and queue workloads.

Fast executor: light CRUD (get_client, list_tools, block operations).
Slow executor: heavy operations (agent step, memory extraction) when invoked directly.
Default event loop executor is set to the fast pool so asyncio.to_thread uses it.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from mirix.log import get_logger
from mirix.settings import settings

logger = get_logger(__name__)

_fast_executor: Optional[ThreadPoolExecutor] = None
_slow_executor: Optional[ThreadPoolExecutor] = None


def get_fast_executor() -> ThreadPoolExecutor:
    """Return the fast I/O executor (light CRUD)."""
    if _fast_executor is None:
        raise RuntimeError("Executors not initialized; call initialize_executors() first")
    return _fast_executor


def get_slow_executor() -> ThreadPoolExecutor:
    """Return the slow I/O executor (agent step, memory extraction)."""
    if _slow_executor is None:
        raise RuntimeError("Executors not initialized; call initialize_executors() first")
    return _slow_executor


def initialize_executors() -> None:
    """Create and register the fast and slow executors."""
    global _fast_executor, _slow_executor
    _fast_executor = ThreadPoolExecutor(
        max_workers=settings.executor_fast_io_workers,
        thread_name_prefix="fast_io",
    )
    _slow_executor = ThreadPoolExecutor(
        max_workers=settings.executor_slow_io_workers,
        thread_name_prefix="slow_io",
    )
    logger.info(
        "Executors initialized: fast_io=%s, slow_io=%s",
        settings.executor_fast_io_workers,
        settings.executor_slow_io_workers,
    )


def shutdown_executors() -> None:
    """Shut down executors (e.g. on app cleanup)."""
    global _fast_executor, _slow_executor
    if _fast_executor is not None:
        _fast_executor.shutdown(wait=True)
        _fast_executor = None
    if _slow_executor is not None:
        _slow_executor.shutdown(wait=True)
        _slow_executor = None
    logger.debug("Executors shut down")
