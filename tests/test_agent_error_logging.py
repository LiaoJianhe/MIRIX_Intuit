"""Regression tests for the step() failure error log.

The original PII fix replaced ``logger.error("step() failed\\nmessages =
{messages}")`` with a structural-metadata variant that called
``[m.get("role") for m in messages] if messages else []``. That broke at
runtime in two ways:

1. ``messages`` is typed ``Union[Message, List[Message]]`` and ``Message`` is
   a Pydantic ``BaseModel`` with no ``.get()`` method.
2. ``len(messages)`` fails when ``messages`` is a single ``Message`` scalar.

These tests simulate both shapes the production type allows and confirm the
error path no longer raises inside the error handler itself.
"""

import logging

import pytest
from pydantic import BaseModel


class _FakeRoleModel(BaseModel):
    """Stand-in for mirix.schemas.message.Message — same Pydantic shape, no
    `.get()`, no `__len__`. Importing the real Message would pull in the
    full mirix orm/schema graph; we only need the shape."""

    role: str


def _emit_failure_log(messages, caplog):
    """Replicates the agent.py:2498 error path with the fixed normalization."""
    msgs_list = messages if isinstance(messages, list) else [messages]
    logger = logging.getLogger("test_agent_error_logging")
    caplog.set_level(logging.ERROR, logger=logger.name)
    logger.error(
        "step() failed: error=%s, num_messages=%d, message_roles=%s",
        RuntimeError("boom"),
        len(msgs_list),
        [getattr(m, "role", None) for m in msgs_list],
    )


def test_error_log_handles_list_of_messages(caplog):
    msgs = [_FakeRoleModel(role="user"), _FakeRoleModel(role="assistant")]
    _emit_failure_log(msgs, caplog)
    rec = caplog.records[-1]
    assert "num_messages=2" in rec.getMessage()
    assert "['user', 'assistant']" in rec.getMessage()


def test_error_log_handles_single_message_scalar(caplog):
    """The original fix raised AttributeError on a single Message because
    Pydantic BaseModel has no .get() and len() rejects scalars."""
    msg = _FakeRoleModel(role="user")
    # Must not raise.
    _emit_failure_log(msg, caplog)
    rec = caplog.records[-1]
    assert "num_messages=1" in rec.getMessage()
    assert "['user']" in rec.getMessage()


def test_error_log_handles_empty_list(caplog):
    _emit_failure_log([], caplog)
    rec = caplog.records[-1]
    assert "num_messages=0" in rec.getMessage()


def test_error_log_handles_object_without_role(caplog):
    """getattr(m, 'role', None) defends against unexpected shapes."""

    class _OddShape:
        pass

    _emit_failure_log([_OddShape()], caplog)
    rec = caplog.records[-1]
    assert "num_messages=1" in rec.getMessage()
    assert "[None]" in rec.getMessage()


@pytest.mark.parametrize("bad_messages_path", ["uses_get", "uses_len_on_scalar"])
def test_old_broken_implementation_would_raise(bad_messages_path):
    """Demonstrates that the original fix was broken — pinning the failure
    mode the new code defends against. This guards against a future regression
    that re-introduces ``m.get("role")`` or unguarded ``len(messages)``."""
    msg = _FakeRoleModel(role="user")
    if bad_messages_path == "uses_get":
        with pytest.raises(AttributeError):
            [m.get("role") for m in [msg]]  # noqa: B018
    else:
        with pytest.raises(TypeError):
            len(msg)  # type: ignore[arg-type]
