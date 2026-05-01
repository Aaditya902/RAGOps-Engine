"""
core/logging.py — Structured JSON logging configuration.

12-Factor App Factor XI: Logs
  Logs are treated as an event stream — written to stdout as newline-delimited
  JSON. The execution environment (Docker, K8s, Datadog) routes them.
  We never open log files or manage rotation here.

Call configure_logging() once at application startup (main.py, cli.py).
All other modules get a logger via: log = structlog.get_logger()
"""
import logging
import structlog
from config import settings


def configure_logging() -> None:
    """Configure structlog for JSON output to stdout."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),   # Factor XI: machine-readable stream
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )