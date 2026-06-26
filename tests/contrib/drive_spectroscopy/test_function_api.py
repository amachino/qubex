"""Public import tests for drive spectroscopy contrib helpers."""

from qubex.contrib import drive_spectroscopy
from qubex.contrib.experiment import drive_spectroscopy as experiment_drive_spectroscopy


def test_drive_spectroscopy_is_exported_from_contrib_modules() -> None:
    """The helper is available from both contrib import surfaces."""
    assert drive_spectroscopy is experiment_drive_spectroscopy
