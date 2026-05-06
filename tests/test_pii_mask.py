"""Tests for mirix.observability.pii_mask.

The mask is the callback Langfuse fires per observation field; it must:
- Forward strings to ispy-pii.
- Walk dicts/lists/tuples recursively, masking leaf strings.
- Pass through non-string scalars unchanged.
- Never raise: ispy-pii failures degrade to REDACTED_PLACEHOLDER.
- Honor the MIRIX_LANGFUSE_MASK_ENABLED kill switch.
"""

from unittest.mock import patch

import httpx
import pytest

from mirix.observability.pii_mask import (
    REDACTED_PLACEHOLDER,
    build_langfuse_mask,
)


@pytest.fixture(autouse=True)
def _enable_mask(monkeypatch):
    monkeypatch.setenv("MIRIX_LANGFUSE_MASK_ENABLED", "true")
    yield


def _build_mask_with_transport(transport: httpx.MockTransport):
    """Construct a mask whose httpx.Client uses the given transport."""
    # Patch httpx.Client at module load so build_langfuse_mask picks it up.
    real_client = httpx.Client

    def _mock_client_factory(*args, **kwargs):
        return real_client(transport=transport, **kwargs)

    with patch("mirix.observability.pii_mask.httpx.Client", _mock_client_factory):
        return build_langfuse_mask(endpoint="http://ispy.test/v2/analyze")


def test_mask_redacts_string():
    def handler(request):
        return httpx.Response(200, json={"redactedText": "ssn ***-**-6789"})

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    assert mask(data="ssn 123-45-6789") == "ssn ***-**-6789"


def test_mask_passes_through_non_string_scalars():
    def handler(request):
        return httpx.Response(200, json={"redactedText": "should-not-be-called"})

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    assert mask(data=42) == 42
    assert mask(data=3.14) == 3.14
    assert mask(data=True) is True
    assert mask(data=None) is None


def test_mask_walks_nested_dict():
    """Every leaf string in a nested dict is forwarded through ispy-pii."""
    inputs_seen: list[str] = []

    def handler(request):
        body = request.read().decode()
        inputs_seen.append(body)
        # Tag the response so we can verify each call routed correctly.
        return httpx.Response(
            200, json={"redactedText": f"masked({len(body)})"}
        )

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    result = mask(
        data={
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        }
    )
    # Each leaf string was forwarded through the mask.
    assert result["messages"][0]["role"].startswith("masked(")
    assert result["messages"][0]["content"].startswith("masked(")
    assert result["messages"][1]["role"].startswith("masked(")
    assert result["messages"][1]["content"].startswith("masked(")
    # Four leaf strings → at most four ispy-pii calls (cache may collapse
    # duplicates; here all four leaves are distinct).
    assert len(inputs_seen) == 4


def test_mask_walks_lists_and_tuples():
    def handler(request):
        return httpx.Response(200, json={"redactedText": "X"})

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    assert mask(data=["a", "b", "c"]) == ["X", "X", "X"]
    assert mask(data=("a", "b")) == ("X", "X")
    assert mask(data=[{"k": "v"}]) == [{"k": "X"}]


def test_mask_caches_repeated_strings():
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"redactedText": "redacted"})

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    mask(data="repeated string")
    mask(data="repeated string")
    mask(data="repeated string")
    assert call_count == 1


def test_mask_returns_placeholder_on_http_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    assert mask(data="ssn 123-45-6789") == REDACTED_PLACEHOLDER


def test_mask_returns_placeholder_on_timeout():
    def handler(request):
        raise httpx.TimeoutException("slow")

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    assert mask(data="ssn 123-45-6789") == REDACTED_PLACEHOLDER


def test_mask_returns_placeholder_on_non_dict_response():
    def handler(request):
        return httpx.Response(200, json=[{"redactedText": "ignored"}])

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    assert mask(data="anything") == REDACTED_PLACEHOLDER


def test_mask_returns_placeholder_on_arbitrary_exception():
    def handler(request):
        raise OSError("connection reset by peer")

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    assert mask(data="anything") == REDACTED_PLACEHOLDER


def test_mask_passthrough_when_disabled(monkeypatch):
    monkeypatch.setenv("MIRIX_LANGFUSE_MASK_ENABLED", "false")

    def handler(request):
        return httpx.Response(200, json={"redactedText": "should-not-be-called"})

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    # Strings, dicts, lists all pass through.
    assert mask(data="user francis@example.com") == "user francis@example.com"
    assert mask(data={"k": "v"}) == {"k": "v"}
    assert mask(data=["a", "b"]) == ["a", "b"]


def test_mask_handles_empty_string():
    def handler(request):
        return httpx.Response(200, json={"redactedText": "should-not-be-called"})

    mask = _build_mask_with_transport(httpx.MockTransport(handler))
    # Empty strings short-circuit; ispy-pii would reject empty input.
    assert mask(data="") == ""
