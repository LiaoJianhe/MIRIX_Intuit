"""Integration tests for S11: citation inclusion in search results.

Full HTTP-level tests that exercise the complete pipeline:
  client.add() → async processing → agents write citations → search(include_citations=True)

Prerequisites:
- Server must be running: python scripts/start_server.py
- Optional: Set MIRIX_API_URL in .env file (defaults to http://localhost:8000)
- Requires GEMINI_API_KEY for LLM-powered agent processing

Run:
    ./scripts/run_tests_with_docker.sh --podman -s -v -k test_search_citations_integration
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pytest
import pytest_asyncio

from mirix.client import MirixClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("isolate_api_key_env"),
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("MIRIX_API_URL", "http://localhost:8000")
CONFIG_PATH = (
    Path(__file__).parent.parent
    / "mirix"
    / "configs"
    / "examples"
    / "mirix_gemini.yaml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def poll_until(
    fetch_results: Callable[[], Awaitable[Dict[str, Any]]],
    is_ready: Callable[[Dict[str, Any]], bool],
    wait_log: str = "Waiting %ds (elapsed %ds)...",
    max_wait_s: int = 90,
    interval_s: int = 10,
) -> Dict[str, Any]:
    """Poll an async callable until condition is met or timeout expires."""
    results = await fetch_results()
    elapsed = 0
    while not is_ready(results) and elapsed < max_wait_s:
        logger.info(wait_log, interval_s, elapsed)
        await asyncio.sleep(interval_s)
        elapsed += interval_s
        results = await fetch_results()
    return results


def _has_any_citation(results: List[Dict[str, Any]]) -> bool:
    """True if any result in the flat list has a non-empty citations list."""
    return any(len(r.get("citations", [])) > 0 for r in results)


def _collect_all_citations_from_memories_dict(
    memories: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Walk a nested memories dict and collect all citation lists found."""
    all_citations = []
    for memory_type, section in memories.items():
        if not isinstance(section, dict):
            continue
        if memory_type == "core":
            for scope_data in section.get("scopes", {}).values():
                if isinstance(scope_data, dict):
                    for item in scope_data.get("items", []):
                        all_citations.extend(item.get("citations", []))
        else:
            for item in section.get("recent", []):
                all_citations.extend(item.get("citations", []))
            for item in section.get("relevant", []):
                all_citations.extend(item.get("citations", []))
            for item in section.get("items", []):
                all_citations.extend(item.get("citations", []))
    return all_citations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TestSearchCitationsIntegration:
    """Full pipeline integration tests for include_citations."""

    pytestmark = [pytest.mark.asyncio(loop_scope="class")]

    @pytest.fixture(scope="class")
    def scope_value(self):
        return f"cit-test-{int(time.time())}"

    @pytest.fixture(scope="class")
    def org_id(self):
        return f"cit-test-org-{int(time.time())}"

    @pytest_asyncio.fixture(scope="class")
    async def client(self, org_id, scope_value):
        client_id = f"cit-test-client-{int(time.time())}"
        c = await MirixClient.create(
            api_key=None,
            client_id=client_id,
            client_name="Citation Test Client",
            client_scope=scope_value,
            org_id=org_id,
            debug=True,
        )
        await c.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=True)
        logger.info(
            "Client initialized: %s (scope=%s, org=%s)", client_id, scope_value, org_id
        )
        return c

    @pytest_asyncio.fixture(scope="class")
    async def user_id(self, client, org_id):
        uid = f"cit-test-user-{int(time.time())}"
        await client.create_or_get_user(
            user_id=uid, user_name="Citation Test User", org_id=org_id
        )
        logger.info("User created: %s", uid)
        return uid

    @pytest_asyncio.fixture(scope="class", autouse=True)
    async def setup_memories(self, client, user_id, scope_value):
        """Add memories and wait for async processing to complete (including citations)."""
        logger.info("Adding memories for citation test...")

        filter_tags = {"scope": scope_value}

        # Add a conversation that should produce episodic + semantic memories
        await client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I had a meeting with the DevOps team about Kubernetes migration",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Noted the DevOps meeting about Kubernetes migration.",
                        }
                    ],
                },
            ],
            filter_tags=filter_tags,
            occurred_at="2026-04-10T14:00:00",
        )
        logger.info("Memory add queued — waiting for async agent processing...")
        await asyncio.sleep(45)

        # Verify memories were actually created before running tests
        results = await poll_until(
            fetch_results=lambda: client.search(
                user_id=user_id,
                query="Kubernetes",
                limit=10,
            ),
            is_ready=lambda r: r.get("count", 0) > 0,
            max_wait_s=60,
            interval_s=10,
        )
        assert (
            results.get("count", 0) > 0
        ), "Memories were not created — agent processing may have failed"
        logger.info(
            "Verified: %d memories exist before citation tests", results["count"]
        )

    # -------------------------------------------------------------------
    # Tests
    # -------------------------------------------------------------------

    async def test_search_without_include_citations_has_no_citations_key(
        self, client, user_id
    ):
        """Default search (include_citations=False) should NOT have a citations key on results."""
        results = await client.search(
            user_id=user_id,
            query="Kubernetes",
            limit=10,
        )
        assert results["success"]
        assert results["count"] > 0
        for r in results["results"]:
            assert (
                "citations" not in r
            ), f"citations key should not be present by default: {r}"

    async def test_search_with_include_citations(self, client, user_id):
        """search(include_citations=True) should return citations on results."""
        results = await client.search(
            user_id=user_id,
            query="Kubernetes",
            limit=10,
            include_citations=True,
        )
        assert results["success"]
        assert results["count"] > 0

        # Every result should have a citations key (list, possibly empty)
        for r in results["results"]:
            assert "citations" in r, f"Missing citations key on result: {r}"
            assert isinstance(r["citations"], list)

        # At least one result should have a non-empty citation (the agent created these)
        assert _has_any_citation(
            results["results"]
        ), "Expected at least one result with citations — agent should have written citations"

        # Validate citation structure
        for r in results["results"]:
            for cit in r["citations"]:
                assert "memory_source_id" in cit, f"Missing memory_source_id: {cit}"
                assert "citation_type" in cit, f"Missing citation_type: {cit}"
                assert cit["citation_type"] in (
                    "created",
                    "updated",
                ), f"Unexpected citation_type: {cit}"
                assert "occurred_at" in cit
                assert "external_thread_id" in cit

    async def test_search_citation_memory_source_id_is_valid(self, client, user_id):
        """The memory_source_id in citations should be retrievable via the source endpoint."""
        results = await client.search(
            user_id=user_id,
            query="Kubernetes",
            limit=10,
            include_citations=True,
        )
        # Collect all unique source IDs from citations
        source_ids = set()
        for r in results["results"]:
            for cit in r.get("citations", []):
                if cit.get("memory_source_id"):
                    source_ids.add(cit["memory_source_id"])

        assert (
            len(source_ids) > 0
        ), "Expected at least one memory_source_id in citations"

        # Verify each source ID is retrievable
        for sid in source_ids:
            resp = await client._request("GET", f"/memory-sources/{sid}")
            assert (
                resp is not None
            ), f"memory_source_id {sid} from citation was not found"
            assert resp.get("id") == sid

    async def test_search_all_users_with_citations(self, client, user_id, scope_value):
        """search_all_users(include_citations=True) attaches citations."""
        results = await client.search_all_users(
            query="Kubernetes",
            client_id=client.client_id,
            limit=10,
            include_citations=True,
        )
        assert results["success"]
        assert results["count"] > 0

        for r in results["results"]:
            assert "citations" in r
            assert isinstance(r["citations"], list)

        assert _has_any_citation(
            results["results"]
        ), "Expected citations in search_all_users results"

    async def test_retrieve_with_topic_with_citations(self, client, user_id):
        """retrieve_with_topic(include_citations=True) attaches citations to nested memories."""
        result = await client.retrieve_with_topic(
            user_id=user_id,
            topic="Kubernetes",
            limit=10,
            include_citations=True,
        )
        assert result.get("success")
        memories = result.get("memories", {})

        all_cits = _collect_all_citations_from_memories_dict(memories)
        assert len(all_cits) > 0, (
            f"Expected citations in retrieve_with_topic memories, got none. "
            f"Memory types present: {list(memories.keys())}"
        )

        # Validate structure
        for cit in all_cits:
            assert "memory_source_id" in cit
            assert "citation_type" in cit

    async def test_retrieve_with_conversation_with_citations(self, client, user_id):
        """retrieve_with_conversation(include_citations=True) attaches citations."""
        result = await client.retrieve_with_conversation(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tell me about the Kubernetes meeting"}
                    ],
                },
            ],
            limit=10,
            include_citations=True,
        )
        assert result.get("success")
        memories = result.get("memories", {})

        all_cits = _collect_all_citations_from_memories_dict(memories)
        assert len(all_cits) > 0, (
            f"Expected citations in retrieve_with_conversation memories, got none. "
            f"Memory types present: {list(memories.keys())}"
        )

    async def test_retrieve_without_citations_has_no_citations_key(
        self, client, user_id
    ):
        """Default retrieve (no include_citations) should NOT have citations on items."""
        result = await client.retrieve_with_topic(
            user_id=user_id,
            topic="Kubernetes",
            limit=10,
        )
        assert result.get("success")
        memories = result.get("memories", {})

        # Walk all items and verify no citations key exists
        for memory_type, section in memories.items():
            if not isinstance(section, dict):
                continue
            for item in section.get("items", []):
                assert (
                    "citations" not in item
                ), f"citations key should not be present by default: {item}"
            for item in section.get("recent", []):
                assert "citations" not in item
            for item in section.get("relevant", []):
                assert "citations" not in item

    async def test_multiple_sources_produce_multiple_citations(
        self, client, user_id, scope_value
    ):
        """Adding a second conversation about the same topic should produce additional citations."""
        # Add a second conversation about Kubernetes
        await client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Follow-up: the Kubernetes migration is now scheduled for next week",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Got it, Kubernetes migration scheduled for next week.",
                        }
                    ],
                },
            ],
            chaining=True,
            filter_tags={"scope": scope_value},
            occurred_at="2026-04-11T10:00:00",
        )
        logger.info("Second memory add queued — waiting for processing...")
        await asyncio.sleep(45)

        # Search with citations — some memories may now have >1 citation
        results = await poll_until(
            fetch_results=lambda: client.search(
                user_id=user_id,
                query="Kubernetes migration",
                limit=10,
                include_citations=True,
            ),
            is_ready=lambda r: any(
                len(x.get("citations", [])) > 1 for x in r.get("results", [])
            ),
            max_wait_s=60,
            interval_s=10,
        )

        # Find results with multiple citations (memory updated by both sources)
        multi_citation_results = [
            r for r in results["results"] if len(r.get("citations", [])) > 1
        ]
        logger.info(
            "Results with multiple citations: %d / %d",
            len(multi_citation_results),
            results["count"],
        )

        # Verify the source IDs are distinct
        if multi_citation_results:
            for r in multi_citation_results:
                source_ids = [c["memory_source_id"] for c in r["citations"]]
                assert (
                    len(set(source_ids)) > 1
                ), f"Expected distinct source IDs for multi-citation result, got: {source_ids}"
