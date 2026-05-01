from unittest.mock import patch

import httpx
import pytest

from mirix.pii import REDACTED_PLACEHOLDER, mask


@pytest.fixture(autouse=True)
def _enable_pii(monkeypatch):
    monkeypatch.setenv("MIRIX_ISPY_PII_ENABLED", "true")
    monkeypatch.setenv("MIRIX_ISPY_PII_ENDPOINT", "https://ispy.test/v2/analyze")
    monkeypatch.setenv("MIRIX_ISPY_PII_TIMEOUT_MS", "200")
    yield


def test_mask_returns_redacted_text_on_success():
    def handler(request):
        return httpx.Response(200, json={"redactedText": "My SSN is [REDACTED]"})

    with patch(
        "mirix.pii._client", httpx.Client(transport=httpx.MockTransport(handler))
    ):
        assert mask("My SSN is 123-45-6789") == "My SSN is [REDACTED]"


def test_mask_returns_passthrough_when_disabled(monkeypatch):
    monkeypatch.setenv("MIRIX_ISPY_PII_ENABLED", "false")
    assert mask("anything") == "anything"


def test_mask_returns_placeholder_on_http_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    with patch(
        "mirix.pii._client", httpx.Client(transport=httpx.MockTransport(handler))
    ):
        assert mask("My SSN is 123-45-6789") == REDACTED_PLACEHOLDER


def test_mask_returns_placeholder_on_timeout():
    def handler(request):
        raise httpx.TimeoutException("slow")

    with patch(
        "mirix.pii._client", httpx.Client(transport=httpx.MockTransport(handler))
    ):
        assert mask("My SSN is 123-45-6789") == REDACTED_PLACEHOLDER


def test_mask_handles_empty_string():
    assert mask("") == ""


def test_mask_returns_placeholder_when_response_missing_field():
    def handler(request):
        return httpx.Response(200, json={"unrelated": "field"})

    with patch(
        "mirix.pii._client", httpx.Client(transport=httpx.MockTransport(handler))
    ):
        assert mask("hi") == REDACTED_PLACEHOLDER
