import json
import logging
from datetime import datetime, timezone
from typing import Any


LOGGER_NAME = "mtbank.agent"


def _create_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


logger = _create_logger()


def log_agent_event(
    *,
    agent: str,
    event: str,
    payload: Any,
    duration_ms: float | None = None,
) -> None:
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "mtbank-call-analytics",
        "agent": agent,
        "event": event,
        "payload": payload,
    }

    if duration_ms is not None:
        record["duration_ms"] = round(duration_ms, 2)

    logger.info(
        json.dumps(
            record,
            ensure_ascii=False,
            default=str,
        )
    )