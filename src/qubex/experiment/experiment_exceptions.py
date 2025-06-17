from __future__ import annotations


class CalibrationMissingError(Exception):
    """Exception raised when calibration data is missing."""

    def __init__(
        self,
        message="Calibration data is missing.",
        *,
        target: str | None = None,
    ):
        if target:
            message += f" ({target})"
        super().__init__(message)
