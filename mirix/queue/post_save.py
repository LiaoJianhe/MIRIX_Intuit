"""Post-save hook: tell registered functions that a save has finished.

``dispatch_save`` is the one place every run mode (numaflow consumer, internal
kafka worker, in-memory worker) passes through. After a save is finalized it
calls :func:`notify_post_save` with what happened. The hook only notifies:

* hooks are started as background tasks and never awaited, so a slow hook cannot
  delay the save;
* every exception is caught and logged, so a failing hook cannot change the
  save's outcome;
* with no hook registered it does nothing.

The event carries only MIRIX's own facts. What a consumer does with them is the
consumer's business.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional, Set, Tuple

from mirix.log import get_logger

if TYPE_CHECKING:  # error_policy imports this module; avoid the cycle at runtime
    from mirix.queue.error_policy import SaveOutcome

logger = get_logger(__name__)


@dataclass(frozen=True)
class PostSaveEvent:
    """What a finished save looked like."""

    memory_source_id: Optional[str]
    user_id: Optional[str]
    client_id: Optional[str]
    agent_id: Optional[str]
    outcome: SaveOutcome
    # (memory_type, memory_id) for each memory the save wrote.
    written: Tuple[Tuple[str, str], ...] = ()


PostSaveHook = Callable[[PostSaveEvent], Awaitable[None]]

_HOOKS: List[PostSaveHook] = []
_PENDING: Set["asyncio.Task[None]"] = set()


def register_post_save_hook(hook: PostSaveHook) -> None:
    """Register ``hook`` to be called after every save. Registering twice is a no-op."""
    if hook not in _HOOKS:
        _HOOKS.append(hook)
        logger.info("Registered post-save hook: %s", getattr(hook, "__name__", repr(hook)))


def clear_post_save_hooks() -> None:
    """Remove every registered hook. For tests."""
    _HOOKS.clear()


async def _run_hook(hook: PostSaveHook, event: PostSaveEvent) -> None:
    try:
        await hook(event)
    except Exception:  # noqa: BLE001 - a hook must never affect the save
        logger.exception(
            "post-save hook %s failed (ignored) memory_source=%s",
            getattr(hook, "__name__", repr(hook)),
            event.memory_source_id,
        )


def notify_post_save(event: PostSaveEvent) -> None:
    """Start every registered hook in the background. Never raises, never waits."""
    if not _HOOKS:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("post-save notify skipped: no running event loop")
        return
    for hook in list(_HOOKS):
        try:
            task = loop.create_task(_run_hook(hook, event))
        except Exception:  # noqa: BLE001
            logger.exception("could not schedule post-save hook (ignored)")
            continue
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)
