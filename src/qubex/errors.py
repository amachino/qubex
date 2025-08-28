from __future__ import annotations


class CalibrationMissingError(Exception):
    """Exception raised when calibration data is missing."""

    def __init__(
        self,
        message: str = "Calibration data is missing.",
        *,
        target: str | None = None,
    ):
        if target:
            message += f" ({target})"
        super().__init__(message)


class BackendUnavailableError(ImportError):
    """Raised when a backend-dependent API is used without backend extras.

    Guidance is provided to install backend extras when this error is raised.
    """

    def __init__(self, message: str | None = None):
        if message is None:
            message = (
                "Backend components are unavailable. Install backend extras: "
                'pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"'
            )
        super().__init__(message)


__all__ = [
    "CalibrationMissingError",
    "BackendUnavailableError",
]

