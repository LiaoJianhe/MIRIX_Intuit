import logging
from unittest.mock import patch

import pytest

from mirix.pii_filter import PIIRedactionFilter, _masked


@pytest.fixture(autouse=True)
def _reset_cache():
    _masked.cache_clear()
    yield
    _masked.cache_clear()


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


def test_cache_avoids_repeat_calls():
    f = PIIRedactionFilter()
    rec1 = _record(logging.WARNING, "boom %s", ("x",))
    rec2 = _record(logging.WARNING, "boom %s", ("x",))
    with patch("mirix.pii_filter.mask", return_value="masked") as m:
        f.filter(rec1)
        f.filter(rec2)
        assert m.call_count == 1


def test_filter_returns_true_when_no_change():
    f = PIIRedactionFilter()
    rec = _record(logging.WARNING, "no pii here")
    with patch("mirix.pii_filter.mask", return_value="no pii here"):
        assert f.filter(rec) is True
        assert rec.getMessage() == "no pii here"
