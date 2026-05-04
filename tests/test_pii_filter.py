import logging
from unittest.mock import patch

from mirix.pii_filter import PIIRedactionFilter


def _record(level: int, msg: str, args=()) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_info_records_pass_through_unchanged():
    f = PIIRedactionFilter()
    rec = _record(logging.INFO, "user ssn=123-45-6789")
    with patch("mirix.pii_filter.mask") as m:
        assert f.filter(rec) is True
        m.assert_not_called()
    assert rec.getMessage() == "user ssn=123-45-6789"


def test_warning_records_are_masked():
    f = PIIRedactionFilter()
    rec = _record(logging.WARNING, "ssn %s", ("123-45-6789",))
    with patch("mirix.pii_filter.mask", return_value="ssn [REDACTED]") as m:
        assert f.filter(rec) is True
        assert rec.getMessage() == "ssn [REDACTED]"
        m.assert_called_once_with("ssn 123-45-6789")


def test_error_records_are_masked():
    f = PIIRedactionFilter()
    rec = _record(logging.ERROR, "boom %s", ("secret",))
    with patch("mirix.pii_filter.mask", return_value="boom [X]"):
        assert f.filter(rec) is True
        assert rec.getMessage() == "boom [X]"


def test_filter_returns_true_when_no_change():
    f = PIIRedactionFilter()
    rec = _record(logging.WARNING, "no pii here")
    with patch("mirix.pii_filter.mask", return_value="no pii here"):
        assert f.filter(rec) is True
        assert rec.getMessage() == "no pii here"


def test_filter_picks_up_redactor_swap():
    """Regression: a previous lru_cache made the filter ignore set_redactor."""
    from mirix.pii import set_redactor

    f = PIIRedactionFilter()
    set_redactor(lambda s: s.upper())
    try:
        rec = _record(logging.WARNING, "before %s", ("after",))
        f.filter(rec)
        assert rec.getMessage() == "BEFORE AFTER"

        set_redactor(lambda s: s.replace("after", "[redacted]"))
        rec2 = _record(logging.WARNING, "before %s", ("after",))
        f.filter(rec2)
        assert rec2.getMessage() == "before [redacted]"
    finally:
        set_redactor(None)
