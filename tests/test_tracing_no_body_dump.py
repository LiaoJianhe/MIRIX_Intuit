"""Verify update_trace_attributes does not capture raw body content."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_update_trace_attributes_omits_raw_body_values(monkeypatch):
    from mirix import tracing

    monkeypatch.setattr(tracing, "_is_tracing_initialized", True)

    captured: dict[str, Any] = {}
    fake_span = MagicMock()
    fake_span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
    fake_span.update_name = MagicMock()
    monkeypatch.setattr(tracing.trace, "get_current_span", lambda: fake_span)

    fake_request = MagicMock()
    fake_request.scope = {"route": MagicMock(path="/test")}
    fake_request.method = "POST"
    fake_request.url = "http://x/test"
    fake_request.path_params = {}
    fake_request.json = AsyncMock(
        return_value={
            "messages": [{"role": "user", "content": "my SSN is 123-45-6789"}],
            "user_id": "u1",
            "query": "my SSN is 123-45-6789",
        }
    )

    await tracing.update_trace_attributes(fake_request)

    # No span attribute should contain the raw PII content.
    for k, v in captured.items():
        assert "123-45-6789" not in str(v), f"PII leaked into span attr {k}={v}"

    # Structural metadata + allowlisted ID is captured.
    assert captured.get("request.body.has_messages") is True
    assert captured.get("request.body.num_messages") == 1
    assert captured.get("request.body.has_query") is True
    assert captured.get("request.body.user_id") == "u1"


@pytest.mark.asyncio
async def test_update_trace_attributes_skipped_when_tracing_off(monkeypatch):
    from mirix import tracing

    monkeypatch.setattr(tracing, "_is_tracing_initialized", False)
    fake_request = MagicMock()
    # No exceptions raised when tracing is disabled.
    await tracing.update_trace_attributes(fake_request)


@pytest.mark.asyncio
async def test_trace_method_does_not_capture_params_by_default(monkeypatch):
    from mirix import tracing

    monkeypatch.setattr(tracing, "_is_tracing_initialized", True)

    captured: dict[str, Any] = {}
    fake_span = MagicMock()
    fake_span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
    fake_span.set_status = MagicMock()

    class FakeContext:
        def __enter__(self):
            return fake_span

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        tracing.tracer, "start_as_current_span", lambda name: FakeContext()
    )

    @tracing.trace_method
    async def secret_op(messages, user_id):
        return "ok"

    await secret_op([{"content": "ssn 123-45-6789"}], "u1")

    # No `parameter.*` attribute should be set when no allowlist is provided.
    assert not any(k.startswith("parameter.") for k in captured.keys())


@pytest.mark.asyncio
async def test_trace_method_captures_only_allowlisted_params(monkeypatch):
    from mirix import tracing

    monkeypatch.setattr(tracing, "_is_tracing_initialized", True)

    captured: dict[str, Any] = {}
    fake_span = MagicMock()
    fake_span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
    fake_span.set_status = MagicMock()

    class FakeContext:
        def __enter__(self):
            return fake_span

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        tracing.tracer, "start_as_current_span", lambda name: FakeContext()
    )

    @tracing.trace_method(trace_params=("user_id",))
    async def op(messages, user_id):
        return "ok"

    await op([{"content": "ssn 123-45-6789"}], "u1")

    assert captured.get("parameter.user_id") == "u1"
    assert "parameter.messages" not in captured
