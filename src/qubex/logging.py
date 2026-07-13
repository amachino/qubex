"""Logging helpers for Qubex."""

from __future__ import annotations

import logging

# Avoid "No handler found" warnings
logging.getLogger("qubex").addHandler(logging.NullHandler())


class LogFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log records with color and exception context."""
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            if exc_type:
                # Append exception type and value to the message
                record.msg = f"{record.msg} ({exc_type.__name__}: {exc_value})"
                # Prevent caching effects if record is reused (unlikely but safe)
                record.message = ""

        log_fmt = "[%(name)s] %(levelname)s: %(message)s"
        if record.levelno == logging.INFO:
            fmt = "%(message)s"
        elif record.levelno == logging.WARNING:
            fmt = self.YELLOW + log_fmt + self.RESET
        elif record.levelno == logging.ERROR:
            fmt = self.RED + log_fmt + self.RESET
        elif record.levelno == logging.CRITICAL:
            fmt = self.BOLD_RED + log_fmt + self.RESET
        else:
            fmt = self.GREY + log_fmt + self.RESET

        formatter = logging.Formatter(fmt)
        return formatter.format(record)


def set_log_level(level: int | str = logging.INFO) -> None:
    """Set the logging level for the qubex package and ensure a console handler is present."""
    logger = logging.getLogger("qubex")
    logger.setLevel(level)

    # Ensure a StreamHandler is attached so the user sees output
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(LogFormatter())
        logger.addHandler(handler)
