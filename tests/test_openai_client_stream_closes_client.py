"""Regression coverage for OpenAIClient.stream()'s client lifecycle.

Background: AsyncOpenAI's underlying httpx client only cleans up via
AsyncHttpxClientWrapper.__del__, which fires an unawaited
asyncio.create_task(self.aclose()) at GC time. On a per-call client that
never gets explicitly closed, an exception in that orphaned cleanup task is
never retrieved and surfaces later as an unrelated-looking "Future exception
was never retrieved" from asyncio's default handler — see the identical fix
+ rationale in OpenAIClient.request() and mirix.embeddings.

stream() can't `async with`-manage its client like request() does, because
the caller consumes the returned stream AFTER the method returns — closing
the client immediately would sever the stream mid-flight. Instead,
_ClientOwningAsyncStream ties the client's lifetime to the stream's own:
this file verifies the client is actually closed on every exit path
(normal completion, an early break, and an exception raised mid-iteration),
plus the case where the request itself fails before any stream exists.
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.llm_api.openai_client import OpenAIClient
from mirix.schemas.llm_config import LLMConfig


def _llm_config():
    return LLMConfig(
        model="gpt-4.1",
        model_endpoint_type="openai",
        model_endpoint="https://example.invalid/v1",
        context_window=8192,
    )


class _FakeAsyncStream:
    """Minimal stand-in for openai.AsyncStream: yields fixed chunks, tracks close()."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        self.closed = True


def _mock_openai_client(fake_stream=None, create_side_effect=None):
    """Patch AsyncOpenAI so client.chat.completions.create(...) returns
    fake_stream (or raises create_side_effect), and client.close() is an
    AsyncMock we can assert on.
    """
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    if create_side_effect is not None:
        mock_client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    else:
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream)
    return patch("mirix.llm_api.openai_client.AsyncOpenAI", return_value=mock_client), mock_client


@pytest.mark.asyncio
async def test_stream_closes_client_on_normal_completion():
    """Fully consuming the stream closes both the stream and the client."""
    fake_stream = _FakeAsyncStream([1, 2, 3])
    patcher, mock_client = _mock_openai_client(fake_stream=fake_stream)

    client = OpenAIClient(llm_config=_llm_config())
    with patcher:
        owning_stream = await client.stream({"model": "gpt-4.1", "messages": []})
        seen = [chunk async for chunk in owning_stream]

    assert seen == [1, 2, 3]
    assert fake_stream.closed is True
    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_closes_client_on_early_break_under_async_with():
    """Breaking out of iteration early, under `async with`, still closes both.

    A bare `break` alone does NOT guarantee cleanup here — Python only runs
    an async generator's `finally` on an explicit `aclose()`, not simply on
    the `async for` loop ending (same real behavior as openai.AsyncStream
    itself). `async with` (or an explicit `close()` call) is what callers
    need for a cleanup guarantee on early exit.
    """
    fake_stream = _FakeAsyncStream([1, 2, 3])
    patcher, mock_client = _mock_openai_client(fake_stream=fake_stream)

    client = OpenAIClient(llm_config=_llm_config())
    with patcher:
        async with await client.stream({"model": "gpt-4.1", "messages": []}) as owning_stream:
            async for chunk in owning_stream:
                if chunk == 1:
                    break

    assert fake_stream.closed is True
    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_closes_client_on_exception_during_iteration():
    """An exception raised while iterating still closes the stream and client."""

    class _BoomStream(_FakeAsyncStream):
        async def _iter(self):
            yield 1
            raise RuntimeError("boom mid-stream")

    fake_stream = _BoomStream([])
    patcher, mock_client = _mock_openai_client(fake_stream=fake_stream)

    client = OpenAIClient(llm_config=_llm_config())
    with patcher:
        owning_stream = await client.stream({"model": "gpt-4.1", "messages": []})
        with pytest.raises(RuntimeError, match="boom mid-stream"):
            async for _chunk in owning_stream:
                pass

    assert fake_stream.closed is True
    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_closes_client_when_create_call_itself_fails():
    """If client.chat.completions.create() raises before any stream exists,
    the client is still closed (no stream to delegate close() to)."""
    patcher, mock_client = _mock_openai_client(create_side_effect=RuntimeError("connect failed"))

    client = OpenAIClient(llm_config=_llm_config())
    with patcher:
        with pytest.raises(RuntimeError, match="connect failed"):
            await client.stream({"model": "gpt-4.1", "messages": []})

    mock_client.close.assert_awaited_once()
