"""Tests for Layer 1 idempotency: batch_hash computation and external_id auto-derivation.

Covers S3 acceptance criteria:
- Same external_id submitted twice: second INSERT is silently skipped (integration)
- Same content, same thread, same time: batch_hash catches duplicate
- Same content, different thread: batch_hash differs (no false positive)
- All external_message_ids provided: external_id auto-derived from sorted hash
- Duplicate messages within a source: content_hash catches duplicate (covered by S1 tests)
- Length-prefixed hashing prevents boundary collisions
"""

from mirix.services.source_message_manager import (
    compute_batch_hash,
    compute_content_hash,
    derive_external_id_from_message_ids,
    normalize_message,
)


class TestComputeBatchHash:
    """Test batch_hash computation for content-based dedup."""

    def _make_msg(self, role="user", text="hello"):
        return {"role": role, "content": {"text": text}}

    def test_deterministic(self):
        """Same inputs produce same hash."""
        msgs = [self._make_msg()]
        h1 = compute_batch_hash("thread-1", "2026-01-01T00:00:00Z", msgs)
        h2 = compute_batch_hash("thread-1", "2026-01-01T00:00:00Z", msgs)
        assert h1 == h2

    def test_returns_hex_string(self):
        """Hash is a valid 64-char hex SHA-256."""
        h = compute_batch_hash(None, None, [self._make_msg()])
        assert len(h) == 64
        int(h, 16)

    def test_same_content_same_thread_same_time(self):
        """Identical batch → identical hash (dedup succeeds)."""
        msgs = [self._make_msg("user", "hi"), self._make_msg("assistant", "hello")]
        h1 = compute_batch_hash("t1", "2026-01-01T00:00:00Z", msgs)
        h2 = compute_batch_hash("t1", "2026-01-01T00:00:00Z", msgs)
        assert h1 == h2

    def test_same_content_different_thread(self):
        """Same messages but different thread → different hash (no false positive)."""
        msgs = [self._make_msg()]
        h1 = compute_batch_hash("thread-A", None, msgs)
        h2 = compute_batch_hash("thread-B", None, msgs)
        assert h1 != h2

    def test_same_content_different_time(self):
        """Same messages but different occurred_at → different hash."""
        msgs = [self._make_msg()]
        h1 = compute_batch_hash(None, "2026-01-01T00:00:00Z", msgs)
        h2 = compute_batch_hash(None, "2026-01-02T00:00:00Z", msgs)
        assert h1 != h2

    def test_different_content_same_thread(self):
        """Different messages, same thread → different hash."""
        h1 = compute_batch_hash("t1", None, [self._make_msg("user", "hello")])
        h2 = compute_batch_hash("t1", None, [self._make_msg("user", "world")])
        assert h1 != h2

    def test_message_order_matters(self):
        """Reordering messages produces a different hash."""
        m1 = self._make_msg("user", "first")
        m2 = self._make_msg("assistant", "second")
        h1 = compute_batch_hash(None, None, [m1, m2])
        h2 = compute_batch_hash(None, None, [m2, m1])
        assert h1 != h2

    def test_none_thread_and_time(self):
        """None values are handled gracefully (empty string fallback)."""
        h = compute_batch_hash(None, None, [self._make_msg()])
        assert len(h) == 64

    def test_length_prefix_prevents_boundary_collision(self):
        """Length-prefixed encoding prevents thread='ab' + time='c' == thread='a' + time='bc'."""
        msgs = [self._make_msg()]
        h1 = compute_batch_hash("ab", "c", msgs)
        h2 = compute_batch_hash("a", "bc", msgs)
        assert h1 != h2

    def test_empty_messages(self):
        """Empty message list still produces a valid hash."""
        h = compute_batch_hash("t1", "2026-01-01T00:00:00Z", [])
        assert len(h) == 64

    def test_role_boundary_collision(self):
        """Length prefixing prevents role+content boundary collision across messages."""
        # "user" + "abc" vs "use" + "rabc" — different if length-prefixed
        m1 = self._make_msg("user", "abc")
        m2 = self._make_msg("use", "rabc")
        h1 = compute_batch_hash(None, None, [m1])
        h2 = compute_batch_hash(None, None, [m2])
        assert h1 != h2


class TestDeriveExternalIdFromMessageIds:
    """Test external_id auto-derivation from external_message_id values."""

    def test_deterministic(self):
        """Same message IDs produce same external_id."""
        ids = ["msg-1", "msg-2", "msg-3"]
        e1 = derive_external_id_from_message_ids(ids)
        e2 = derive_external_id_from_message_ids(ids)
        assert e1 == e2

    def test_prefix(self):
        """Auto-derived IDs have 'auto-' prefix."""
        eid = derive_external_id_from_message_ids(["msg-1"])
        assert eid.startswith("auto-")

    def test_hex_suffix(self):
        """Suffix is a valid 64-char hex SHA-256."""
        eid = derive_external_id_from_message_ids(["msg-1"])
        hex_part = eid[len("auto-"):]
        assert len(hex_part) == 64
        int(hex_part, 16)

    def test_order_independent(self):
        """Message IDs are sorted before hashing, so order doesn't matter."""
        e1 = derive_external_id_from_message_ids(["msg-3", "msg-1", "msg-2"])
        e2 = derive_external_id_from_message_ids(["msg-1", "msg-2", "msg-3"])
        assert e1 == e2

    def test_different_ids(self):
        """Different message IDs produce different external_ids."""
        e1 = derive_external_id_from_message_ids(["msg-1", "msg-2"])
        e2 = derive_external_id_from_message_ids(["msg-1", "msg-3"])
        assert e1 != e2

    def test_single_id(self):
        """Works with a single message ID."""
        eid = derive_external_id_from_message_ids(["msg-1"])
        assert eid.startswith("auto-")

    def test_length_prefix_prevents_collision(self):
        """Length-prefixed encoding prevents 'ab' + 'c' == 'a' + 'bc' collisions."""
        e1 = derive_external_id_from_message_ids(["ab", "c"])
        e2 = derive_external_id_from_message_ids(["a", "bc"])
        assert e1 != e2


class TestPersistMemorySourceIdempotencyLogic:
    """Test the idempotency decision logic used by _persist_memory_source.

    These are pure-logic tests that verify the decision tree:
    1. If external_id is provided → use it (no batch_hash)
    2. If all messages have external_message_ids → auto-derive external_id (no batch_hash)
    3. Otherwise → compute batch_hash
    """

    def test_client_external_id_takes_precedence(self):
        """When client provides external_id, no auto-derivation or batch_hash needed."""
        external_id = "client-provided-id"
        msgs = [{"role": "user", "content": {"text": "hi"}, "external_message_id": "m1"}]

        # Simulate the logic from _persist_memory_source
        batch_hash = None
        if not external_id and msgs:
            ext_msg_ids = [m["external_message_id"] for m in msgs if m.get("external_message_id")]
            if len(ext_msg_ids) == len(msgs):
                external_id = derive_external_id_from_message_ids(ext_msg_ids)
        if not external_id and msgs:
            batch_hash = compute_batch_hash(None, None, msgs)

        assert external_id == "client-provided-id"
        assert batch_hash is None

    def test_auto_derive_when_all_messages_have_ids(self):
        """When all messages have external_message_ids, auto-derive external_id."""
        external_id = None
        msgs = [
            {"role": "user", "content": {"text": "hi"}, "external_message_id": "m1"},
            {"role": "assistant", "content": {"text": "hello"}, "external_message_id": "m2"},
        ]

        batch_hash = None
        if not external_id and msgs:
            ext_msg_ids = [m["external_message_id"] for m in msgs if m.get("external_message_id")]
            if len(ext_msg_ids) == len(msgs):
                external_id = derive_external_id_from_message_ids(ext_msg_ids)
        if not external_id and msgs:
            batch_hash = compute_batch_hash(None, None, msgs)

        assert external_id is not None
        assert external_id.startswith("auto-")
        assert batch_hash is None

    def test_batch_hash_when_partial_message_ids(self):
        """When only some messages have external_message_ids, fall back to batch_hash."""
        external_id = None
        msgs = [
            {"role": "user", "content": {"text": "hi"}, "external_message_id": "m1"},
            {"role": "assistant", "content": {"text": "hello"}},  # No external_message_id
        ]

        batch_hash = None
        if not external_id and msgs:
            ext_msg_ids = [m["external_message_id"] for m in msgs if m.get("external_message_id")]
            if len(ext_msg_ids) == len(msgs):
                external_id = derive_external_id_from_message_ids(ext_msg_ids)
        if not external_id and msgs:
            batch_hash = compute_batch_hash(None, None, msgs)

        assert external_id is None
        assert batch_hash is not None
        assert len(batch_hash) == 64

    def test_batch_hash_when_no_message_ids(self):
        """When no messages have external_message_ids, fall back to batch_hash."""
        external_id = None
        msgs = [
            {"role": "user", "content": {"text": "hi"}},
            {"role": "assistant", "content": {"text": "hello"}},
        ]

        batch_hash = None
        if not external_id and msgs:
            ext_msg_ids = [m["external_message_id"] for m in msgs if m.get("external_message_id")]
            if len(ext_msg_ids) == len(msgs):
                external_id = derive_external_id_from_message_ids(ext_msg_ids)
        if not external_id and msgs:
            batch_hash = compute_batch_hash(None, None, msgs)

        assert external_id is None
        assert batch_hash is not None

    def test_empty_messages_no_hash(self):
        """When there are no messages, neither external_id nor batch_hash is computed."""
        external_id = None
        msgs = []

        batch_hash = None
        if not external_id and msgs:
            ext_msg_ids = [m["external_message_id"] for m in msgs if m.get("external_message_id")]
            if len(ext_msg_ids) == len(msgs):
                external_id = derive_external_id_from_message_ids(ext_msg_ids)
        if not external_id and msgs:
            batch_hash = compute_batch_hash(None, None, msgs)

        assert external_id is None
        assert batch_hash is None
