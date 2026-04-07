"""
Call-origin context propagation for hybrid read strategy.

Uses Python contextvars.ContextVar to distinguish between client API
requests (Search-only) and engine operations (hybrid Search + IPS
Relational recent-window merge). ContextVar values propagate across
awaits automatically in async code.

Constants:
    CALL_ORIGIN_CLIENT_API: For user-facing memory read/search paths
        (e.g., POST /v1/search, GET /v1/memories/raw/{id}).
    CALL_ORIGIN_ENGINE: For MIRIX engine internals (queue workers,
        build_system_prompt_with_memories, agent inner_step).

The default is CALL_ORIGIN_ENGINE so the safer hybrid path is used
when the origin is unknown.

Usage:
    # In ECMS routes (before calling MIRIX)
    from mirix.database.call_context import set_call_origin, CALL_ORIGIN_CLIENT_API
    set_call_origin(CALL_ORIGIN_CLIENT_API)

    # In MIRIX managers (checking context)
    from mirix.database.call_context import get_call_origin, CALL_ORIGIN_CLIENT_API
    if get_call_origin() == CALL_ORIGIN_CLIENT_API:
        # Search-only path
    else:
        # Hybrid path
"""

from contextvars import ContextVar

CALL_ORIGIN_CLIENT_API = "client_api"
CALL_ORIGIN_ENGINE = "engine_operation"

call_origin_var: ContextVar[str] = ContextVar("call_origin", default=CALL_ORIGIN_ENGINE)

_hybrid_window_seconds: int = 5


def set_call_origin(origin: str) -> None:
    """Set the call origin for the current async context."""
    call_origin_var.set(origin)


def get_call_origin() -> str:
    """Get the call origin for the current async context."""
    return call_origin_var.get()


def set_hybrid_window_seconds(seconds: int) -> None:
    """
    Set the hybrid recent-window duration.

    Called once at ECMS startup via register_ips_providers() after reading
    config.ips_hybrid_read_window_seconds. MIRIX reads this value via
    get_hybrid_window_seconds() without importing ECMS SvcSettings.
    """
    global _hybrid_window_seconds
    _hybrid_window_seconds = seconds


def get_hybrid_window_seconds() -> int:
    """Get the configured hybrid recent-window duration in seconds."""
    return _hybrid_window_seconds
