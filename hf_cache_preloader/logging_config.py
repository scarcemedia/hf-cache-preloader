from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    level_name = log_level.upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"invalid LOG_LEVEL: {log_level}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.debug("logging configured", extra={"log_level": level_name})
