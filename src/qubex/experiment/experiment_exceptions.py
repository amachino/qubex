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


class BackendUnavailableError(Exception):
    """Exception raised when backend functionality is used without backend extras installed."""

    def __init__(
        self,
        message="Backend functionality requires backend extras to be installed.",
    ):
        install_message = 'Install backend extra: `pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"`'
        full_message = f"{message} {install_message}"
        super().__init__(full_message)
