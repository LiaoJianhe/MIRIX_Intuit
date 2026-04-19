"""Tests for the DIRECT_WRITE_HANDLERS registry.

The registry maps memory_type to the corresponding LLM-facing *_insert tool
in memory_tools.py directly — there is no shim. Verify every expected
memory_type resolves to the right callable.
"""

import pytest


@pytest.mark.asyncio
async def test_direct_write_handlers_registry_has_all_expected_types():
    from mirix.functions.direct_write_handlers import DIRECT_WRITE_HANDLERS
    from mirix.functions.function_sets.memory_tools import (
        episodic_memory_insert,
        knowledge_vault_insert,
        procedural_memory_insert,
        resource_memory_insert,
        semantic_memory_insert,
    )

    assert DIRECT_WRITE_HANDLERS["episodic"] is episodic_memory_insert
    assert DIRECT_WRITE_HANDLERS["semantic"] is semantic_memory_insert
    assert DIRECT_WRITE_HANDLERS["procedural"] is procedural_memory_insert
    assert DIRECT_WRITE_HANDLERS["resource"] is resource_memory_insert
    assert DIRECT_WRITE_HANDLERS["knowledge_vault"] is knowledge_vault_insert


@pytest.mark.asyncio
async def test_direct_write_handlers_registry_values_are_callables():
    from mirix.functions.direct_write_handlers import DIRECT_WRITE_HANDLERS

    for memory_type, handler in DIRECT_WRITE_HANDLERS.items():
        assert callable(handler), f"Handler for {memory_type!r} is not callable"
