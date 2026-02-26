"""
Tests for mirix.server.executors lifecycle.
"""

import pytest

from mirix.server.executors import (
    get_fast_executor,
    get_slow_executor,
    initialize_executors,
    shutdown_executors,
)


def test_initialize_and_get_executors():
    """initialize_executors() creates executors; get_* return them."""
    shutdown_executors()
    initialize_executors()
    try:
        fast = get_fast_executor()
        slow = get_slow_executor()
        assert fast is not None
        assert slow is not None
        assert fast != slow
    finally:
        shutdown_executors()


def test_get_executor_before_init_raises():
    """get_fast_executor raises if executors not initialized."""
    shutdown_executors()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_fast_executor()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_slow_executor()


def test_shutdown_clears_executors():
    """shutdown_executors() clears executors; get_* raises after."""
    initialize_executors()
    shutdown_executors()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_fast_executor()
