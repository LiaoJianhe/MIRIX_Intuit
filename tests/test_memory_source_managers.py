"""Unit tests for memory source manager services and content hash computation."""

import hashlib
import json
from datetime import datetime, timezone

from mirix.services.source_message_manager import compute_content_hash


class TestComputeContentHash:
    """Test the content hash function used for source message dedup."""

    def test_deterministic(self):
        """Same inputs produce same hash."""
        h1 = compute_content_hash("user", {"text": "hello"})
        h2 = compute_content_hash("user", {"text": "hello"})
        assert h1 == h2

    def test_different_role(self):
        """Different roles produce different hashes."""
        h1 = compute_content_hash("user", {"text": "hello"})
        h2 = compute_content_hash("assistant", {"text": "hello"})
        assert h1 != h2

    def test_different_content(self):
        """Different content produces different hashes."""
        h1 = compute_content_hash("user", {"text": "hello"})
        h2 = compute_content_hash("user", {"text": "world"})
        assert h1 != h2

    def test_returns_hex_string(self):
        """Hash is a valid hex string of expected length (SHA-256 = 64 hex chars)."""
        h = compute_content_hash("user", {"text": "test"})
        assert len(h) == 64
        int(h, 16)  # Should not raise — valid hex

    def test_length_prefix_prevents_boundary_collision(self):
        """Length-prefixed encoding prevents 'ab' + 'c' == 'a' + 'bc' collisions."""
        h1 = compute_content_hash("ab", {"text": "c"})
        h2 = compute_content_hash("a", {"text": "bc"})
        assert h1 != h2

    def test_dict_key_order_irrelevant(self):
        """JSON sort_keys ensures consistent hashing regardless of dict key order."""
        h1 = compute_content_hash("user", {"a": 1, "b": 2})
        h2 = compute_content_hash("user", {"b": 2, "a": 1})
        assert h1 == h2

    def test_nested_content(self):
        """Nested content structures hash correctly."""
        content = {"blocks": [{"type": "text", "text": "hello"}]}
        h = compute_content_hash("user", content)
        assert len(h) == 64


class TestManagerImports:
    """Verify all manager classes can be imported without errors."""

    def test_memory_source_manager_imports(self):
        from mirix.services.memory_source_manager import MemorySourceManager
        assert MemorySourceManager is not None

    def test_source_message_manager_imports(self):
        from mirix.services.source_message_manager import SourceMessageManager
        assert SourceMessageManager is not None

    def test_memory_citation_manager_imports(self):
        from mirix.services.memory_citation_manager import MemoryCitationManager
        assert MemoryCitationManager is not None


class TestManagerMethodSignatures:
    """Verify manager methods exist with expected signatures."""

    def test_memory_source_manager_methods(self):
        from mirix.services.memory_source_manager import MemorySourceManager
        mgr = MemorySourceManager.__new__(MemorySourceManager)
        assert callable(getattr(mgr, "create", None))
        assert callable(getattr(mgr, "get_by_id", None))
        assert callable(getattr(mgr, "mark_processing_complete", None))

    def test_source_message_manager_methods(self):
        from mirix.services.source_message_manager import SourceMessageManager
        mgr = SourceMessageManager.__new__(SourceMessageManager)
        assert callable(getattr(mgr, "bulk_insert", None))

    def test_memory_citation_manager_methods(self):
        from mirix.services.memory_citation_manager import MemoryCitationManager
        mgr = MemoryCitationManager.__new__(MemoryCitationManager)
        assert callable(getattr(mgr, "create", None))
        assert callable(getattr(mgr, "check_exists", None))
        assert callable(getattr(mgr, "get_max_occurred_at", None))
