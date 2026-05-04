import logging
import traceback
from typing import Optional

from mirix.settings import settings

selected_log_level = logging.DEBUG if settings.debug else logging.INFO


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    logger = logging.getLogger("Mirix")
    logger.setLevel(logging.INFO)
    return logger


def safe_traceback(exc: BaseException) -> str:
    """Render exc's frame stack with exception-message text stripped.

    Used at LLM-client and agent error sites where we want to log *where*
    a failure happened without echoing exception messages that may contain
    user-supplied content (LLM provider 400-class responses, for example,
    quote the offending request body in their error strings).

    Returns frames (file:line:function) followed by the exception type
    chain. The runtime exception messages — which is where PII typically
    leaks — are not rendered.
    """
    frames = "".join(traceback.format_tb(exc.__traceback__))
    chain = []
    cur: BaseException | None = exc
    while cur is not None:
        chain.append(type(cur).__name__)
        cur = cur.__cause__ or cur.__context__
    return frames + " -> ".join(chain)
