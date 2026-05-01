import logging
from typing import Optional

from mirix.pii_filter import PIIRedactionFilter
from mirix.settings import settings

selected_log_level = logging.DEBUG if settings.debug else logging.INFO
_PII_FILTER = PIIRedactionFilter()


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    logger = logging.getLogger("Mirix")
    logger.setLevel(logging.INFO)
    if not any(isinstance(f, PIIRedactionFilter) for f in logger.filters):
        logger.addFilter(_PII_FILTER)
    return logger
