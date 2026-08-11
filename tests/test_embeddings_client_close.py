"""Regression coverage for OpenAIEmbeddingWithCustomAuth's client lifecycle.

Background: this call site is wrapped in a retry loop (embedding_with_retry /
traced_embedding_with_retry), so each attempt creates a fresh AsyncOpenAI
client. An abandoned client (on a transient failure that gets retried) only
cleans up via its httpx wrapper's __del__, which schedules an unawaited
asyncio.create_task(self.aclose()) at GC time — if that eventually raises,
the exception is never retrieved and surfaces later as an unrelated "Future
exception was never retrieved" (same root cause fixed in
OpenAIClient.request() and .stream()). `async with AsyncOpenAI(...)`
guarantees an explicit, awaited close on every exit path instead.
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.embeddings import OpenAIEmbeddingWithCustomAuth
from mirix.llm_api.auth_provider import (
    AuthProvider,
    list_auth_providers,
    register_auth_provider,
    unregister_auth_provider,
)
from mirix.schemas.embedding_config import EmbeddingConfig

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def cleanup_registry():
    for name in list_auth_providers():
        unregister_auth_provider(name)
    yield
    for name in list_auth_providers():
        unregister_auth_provider(name)


class _GoodProvider(AuthProvider):
    def get_auth_headers(self):
        return {"Authorization": "Intuit_IAM_Authentication real-token"}

    async def get_auth_headers_async(self):
        return {"Authorization": "Intuit_IAM_Authentication real-token"}


def _config():
    return EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_endpoint="https://example.invalid/v1",
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
        auth_provider="good_provider",
    )


def _mock_openai_client(embedding=None, create_side_effect=None):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    if create_side_effect is not None:
        mock_client.embeddings.create = AsyncMock(side_effect=create_side_effect)
    else:
        response = AsyncMock()
        response.data = [AsyncMock(embedding=embedding)]
        mock_client.embeddings.create = AsyncMock(return_value=response)
    return patch("openai.AsyncOpenAI", return_value=mock_client), mock_client


async def test_get_text_embedding_closes_client_on_success():
    """A successful embedding call still closes the per-call client via `async with`."""
    register_auth_provider("good_provider", _GoodProvider())
    patcher, mock_client = _mock_openai_client(embedding=[0.1, 0.2, 0.3])

    embedder = OpenAIEmbeddingWithCustomAuth(config=_config(), auth_provider="good_provider")
    with patcher:
        result = await embedder.get_text_embedding("hello world")

    assert result == [0.1, 0.2, 0.3]
    mock_client.__aexit__.assert_awaited_once()


async def test_get_text_embedding_closes_client_on_api_error():
    """A non-retryable API error still closes the client (async with covers the raise path)."""
    register_auth_provider("good_provider", _GoodProvider())
    patcher, mock_client = _mock_openai_client(create_side_effect=ValueError("bad request"))

    embedder = OpenAIEmbeddingWithCustomAuth(config=_config(), auth_provider="good_provider")
    with patcher:
        with pytest.raises(ValueError, match="bad request"):
            await embedder.get_text_embedding("hello world")

    mock_client.__aexit__.assert_awaited_once()
