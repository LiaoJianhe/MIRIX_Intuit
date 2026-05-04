import pytest

from mirix.pii import get_redactor, mask, set_redactor


@pytest.fixture(autouse=True)
def _reset_redactor():
    """Restore the passthrough default after every test."""
    yield
    set_redactor(None)


def test_default_is_passthrough():
    assert mask("My SSN is 123-45-6789") == "My SSN is 123-45-6789"


def test_set_redactor_replaces_default():
    set_redactor(lambda s: s.replace("123-45-6789", "[SSN]"))
    assert mask("My SSN is 123-45-6789") == "My SSN is [SSN]"


def test_set_redactor_none_resets_to_passthrough():
    set_redactor(lambda s: "REDACTED")
    assert mask("anything") == "REDACTED"
    set_redactor(None)
    assert mask("anything") == "anything"


def test_get_redactor_returns_current():
    def fn(s):
        return s.upper()

    set_redactor(fn)
    assert get_redactor() is fn


def test_empty_string_short_circuits_redactor():
    """An empty string never invokes the redactor."""
    called = []

    def boom(s):
        called.append(s)
        return s

    set_redactor(boom)
    assert mask("") == ""
    assert called == []


def test_redactor_exception_falls_back_to_original_text():
    """A misbehaving redactor must not break logging."""

    def bad(s):
        raise RuntimeError("boom")

    set_redactor(bad)
    # Original text is returned unchanged.
    assert mask("My SSN is 123-45-6789") == "My SSN is 123-45-6789"
