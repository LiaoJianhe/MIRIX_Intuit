"""
Tests for the recent/relevant union in the save-flow prompt builder
(VEPAGE-1178): ``Agent._merge_recent_into_relevant`` and
``Agent._recent_dedup_note``.

The owning sub-agent unions just-written ("recent", from the Relational
recent-window) rows into the ranked ("relevant", from Search) list, de-duped
by id, and surfaces an inline note when recent rows were appended so the LLM
checks for duplicates before creating new entries.

Usage:
    pytest tests/test_recent_relevant_union.py -v
"""

from types import SimpleNamespace

from mirix.agent.agent import Agent


def _item(id_):
    return SimpleNamespace(id=id_)


class TestMergeRecentIntoRelevant:
    def test_appends_recent_only_rows_after_relevant(self):
        relevant = [_item("a"), _item("b")]
        recent = [_item("c"), _item("d")]

        merged, appended = Agent._merge_recent_into_relevant(relevant, recent)

        assert [m.id for m in merged] == ["a", "b", "c", "d"]
        assert appended == 2

    def test_dedups_recent_rows_already_in_relevant(self):
        relevant = [_item("a"), _item("b")]
        recent = [_item("b"), _item("c")]  # "b" overlaps

        merged, appended = Agent._merge_recent_into_relevant(relevant, recent)

        assert [m.id for m in merged] == ["a", "b", "c"]
        assert appended == 1

    def test_preserves_relevant_ranking_order(self):
        relevant = [_item("c"), _item("a"), _item("b")]  # ranked, not sorted
        recent = []

        merged, appended = Agent._merge_recent_into_relevant(relevant, recent)

        assert [m.id for m in merged] == ["c", "a", "b"]
        assert appended == 0

    def test_empty_recent_returns_relevant_unchanged(self):
        relevant = [_item("a")]
        merged, appended = Agent._merge_recent_into_relevant(relevant, [])
        assert [m.id for m in merged] == ["a"]
        assert appended == 0

    def test_empty_relevant_returns_all_recent(self):
        recent = [_item("x"), _item("y")]
        merged, appended = Agent._merge_recent_into_relevant([], recent)
        assert [m.id for m in merged] == ["x", "y"]
        assert appended == 2

    def test_all_recent_overlap_appends_nothing(self):
        relevant = [_item("a"), _item("b")]
        recent = [_item("a"), _item("b")]
        merged, appended = Agent._merge_recent_into_relevant(relevant, recent)
        assert [m.id for m in merged] == ["a", "b"]
        assert appended == 0


class TestRecentDedupNote:
    def test_note_present_when_recent_rows_appended(self):
        note = Agent._recent_dedup_note({"recent_appended": 3})
        assert note != ""
        assert "recently-written" in note
        assert "duplicates" in note

    def test_no_note_when_no_recent_rows_appended(self):
        assert Agent._recent_dedup_note({"recent_appended": 0}) == ""

    def test_no_note_when_bucket_missing(self):
        assert Agent._recent_dedup_note(None) == ""

    def test_no_note_when_key_absent(self):
        assert Agent._recent_dedup_note({"current_count": 5}) == ""
