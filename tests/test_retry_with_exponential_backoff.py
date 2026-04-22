"""Tests for retry_with_exponential_backoff.

The retry wrapper protects memory writes from transient upstream failures
(rate limits, 5xx). It must catch the error types actually raised by the
clients we use — httpx.HTTPStatusError from direct httpx paths, and
openai.APIStatusError (and subclasses) from the OpenAI SDK.
"""

import httpx
import openai
import pytest

from mirix.llm_api.llm_api_tools import retry_with_exponential_backoff


def _make_openai_status_error(cls, status_code: int):
    """Build a real openai.APIStatusError-family exception for tests."""
    response = httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "http://example/embeddings"),
    )
    return cls(message=f"status {status_code}", response=response, body=None)


def _make_httpx_status_error(status_code: int):
    response = httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "http://example/embeddings"),
    )
    return httpx.HTTPStatusError(f"status {status_code}", request=response.request, response=response)


@pytest.mark.asyncio
async def test_retries_on_openai_rate_limit_then_succeeds():
    """A 429 from the OpenAI SDK should trigger the retry loop, not bubble out."""
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_openai_status_error(openai.RateLimitError, 429)
        return "ok"

    wrapped = retry_with_exponential_backoff(flaky, initial_delay=0, max_retries=3)
    assert await wrapped() == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retries_on_openai_5xx_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _make_openai_status_error(openai.InternalServerError, 503)
        return "ok"

    wrapped = retry_with_exponential_backoff(
        flaky, initial_delay=0, max_retries=5, error_codes=(429, 500, 502, 503, 504)
    )
    assert await wrapped() == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_still_retries_on_httpx_status_error():
    """Existing httpx path must keep working."""
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_httpx_status_error(429)
        return "ok"

    wrapped = retry_with_exponential_backoff(flaky, initial_delay=0, max_retries=3)
    assert await wrapped() == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_does_not_retry_on_non_retryable_openai_error():
    """A 400 must not be retried — it is not in error_codes."""
    calls = {"n": 0}

    async def bad_request():
        calls["n"] += 1
        raise _make_openai_status_error(openai.BadRequestError, 400)

    wrapped = retry_with_exponential_backoff(bad_request, initial_delay=0, max_retries=3)
    with pytest.raises(openai.BadRequestError):
        await wrapped()
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_gives_up_after_max_retries_on_openai_429():
    from mirix.errors import RateLimitExceededError

    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        raise _make_openai_status_error(openai.RateLimitError, 429)

    wrapped = retry_with_exponential_backoff(always_fail, initial_delay=0, max_retries=2)
    with pytest.raises(RateLimitExceededError):
        await wrapped()
    # Initial call + 2 retries = 3 invocations
    assert calls["n"] == 3
