"""S4 of VEPAGE-1251: classify() maps SQLAlchemy DB types + origin-split
for unknowns.

Before this story:
* classify() only mapped LLM types; SQLAlchemy/asyncio/Python builtins all
  fell into the unmapped default (TRANSIENT with one-shot warning).
* That default has the wrong shape for code bugs (AttributeError /
  KeyError / TypeError): retrying a deterministic Python bug burns cost
  and hides the bug under "retrying" noise — the VEPAGE-1228 shape.

After this story:
* SQLAlchemy types MIRIX already imports are mapped by isinstance:
  OperationalError / DBAPIError / asyncio.TimeoutError -> TRANSIENT;
  IntegrityError / DataError -> PERMANENT.
* For exceptions still unmapped, classify() splits by origin: a pure-Python
  bug (AttributeError, KeyError, TypeError, IndexError, NameError) with
  no provider/DB frame in the traceback -> PERMANENT (it'll never
  succeed on retry). Everything else stays TRANSIENT.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.exc import (
    DataError,
    IntegrityError,
    OperationalError,
)

from mirix.queue.error_policy import Bucket, classify


# ---------- SQLAlchemy mappings ----------


class TestSqlAlchemyClassification:
    def test_operational_error_is_transient(self):
        """DB connection blip — worth retrying."""
        exc = OperationalError("SELECT 1", {}, Exception("connection reset"))
        assert classify(exc) is Bucket.TRANSIENT

    def test_integrity_error_is_permanent(self):
        """Constraint violation — re-running won't change the outcome."""
        exc = IntegrityError(
            "INSERT INTO ...", {}, Exception("duplicate key value")
        )
        assert classify(exc) is Bucket.PERMANENT

    def test_data_error_is_permanent(self):
        """Bad data shape — input is the problem; retry won't fix it."""
        exc = DataError(
            "INSERT INTO ...", {}, Exception("invalid input syntax for type")
        )
        assert classify(exc) is Bucket.PERMANENT

    def test_asyncio_timeout_is_transient(self):
        """asyncio.TimeoutError — typical for stalled I/O; retry can recover."""
        assert classify(asyncio.TimeoutError()) is Bucket.TRANSIENT


# ---------- origin-split for unknowns ----------


class TestUnknownOriginSplit:
    """When classify() sees an exception with no isinstance match, it now
    inspects the traceback. Pure-Python bug shapes raised with NO provider
    or DB frame in the chain are reclassified PERMANENT — they will never
    succeed on retry. Everything else still defaults TRANSIENT (the
    historical behavior)."""

    def test_attribute_error_pure_python_is_permanent(self):
        """The VEPAGE-1228 shape: a Resolve Child Agents AttributeError
        with no provider frame. Must classify PERMANENT so the source
        fails fast instead of redelivering for hours."""
        try:
            obj = None
            obj.foo  # raises AttributeError
        except AttributeError as exc:
            assert classify(exc) is Bucket.PERMANENT

    def test_key_error_pure_python_is_permanent(self):
        try:
            d = {}
            d["missing"]
        except KeyError as exc:
            assert classify(exc) is Bucket.PERMANENT

    def test_type_error_pure_python_is_permanent(self):
        try:
            "string" + 1
        except TypeError as exc:
            assert classify(exc) is Bucket.PERMANENT

    def test_index_error_pure_python_is_permanent(self):
        try:
            [][0]
        except IndexError as exc:
            assert classify(exc) is Bucket.PERMANENT

    def test_name_error_pure_python_is_permanent(self):
        try:
            undefined_variable_xyz  # noqa: F821
        except NameError as exc:
            assert classify(exc) is Bucket.PERMANENT

    def test_unknown_non_bug_exception_still_transient(self):
        """A genuinely-unknown exception type (not a code-bug shape)
        keeps the historical TRANSIENT default — better a wasted
        redelivery than silently swallowing something we haven't
        catalogued yet."""

        class _MysteryError(Exception):
            pass

        assert classify(_MysteryError("???")) is Bucket.TRANSIENT

    def test_runtime_error_is_transient_not_permanent(self):
        """RuntimeError is too broad to call permanent — it shows up in
        provider/SDK code as a catch-all wrapper. Keep the default."""
        assert classify(RuntimeError("???")) is Bucket.TRANSIENT
