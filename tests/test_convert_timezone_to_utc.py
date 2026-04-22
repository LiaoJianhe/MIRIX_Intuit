"""
Unit tests for convert_timezone_to_utc.

Covers the legacy space-separated formats plus ISO-8601 variants that LLMs
emit naturally (T separator, optional Z / offset), which previously crashed
the episodic memory insert path.
"""

from datetime import timezone as dt_timezone

import pytest

from mirix.utils import convert_timezone_to_utc


TZ = "America/Los_Angeles (UTC-08:00)"


class TestLegacyFormats:
    """The two formats the original implementation already accepted."""

    def test_space_separated_seconds(self):
        result = convert_timezone_to_utc("2026-04-22 08:00:00", TZ)
        # 08:00 America/Los_Angeles on this date (DST) == 15:00 UTC
        assert result.hour == 15
        assert result.tzinfo.utcoffset(result) == dt_timezone.utc.utcoffset(result)

    def test_space_separated_microseconds(self):
        result = convert_timezone_to_utc("2026-04-22 08:00:00.123456", TZ)
        assert result.hour == 15
        assert result.microsecond == 123456


class TestISO8601:
    """ISO-8601 strings the LLM emits for episodic_memory_insert items."""

    def test_iso_with_t_separator(self):
        # Repro of the reported Langfuse error
        result = convert_timezone_to_utc("2026-04-22T08:00:00", TZ)
        assert result.hour == 15
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 22

    def test_iso_with_t_and_microseconds(self):
        result = convert_timezone_to_utc("2026-04-22T08:00:00.123456", TZ)
        assert result.hour == 15
        assert result.microsecond == 123456

    def test_iso_with_z_suffix_is_utc(self):
        # If the string already carries a UTC marker, honor it — do not re-localize.
        result = convert_timezone_to_utc("2026-04-22T15:00:00Z", TZ)
        assert result.hour == 15
        assert result.utcoffset() == dt_timezone.utc.utcoffset(result)

    def test_iso_with_explicit_offset_is_honored(self):
        # +02:00 at 17:00 local == 15:00 UTC, regardless of the timezone arg.
        result = convert_timezone_to_utc("2026-04-22T17:00:00+02:00", TZ)
        assert result.hour == 15
        assert result.utcoffset() == dt_timezone.utc.utcoffset(result)


class TestInvalid:
    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            convert_timezone_to_utc("not a timestamp", TZ)
