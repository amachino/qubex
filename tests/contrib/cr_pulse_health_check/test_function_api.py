"""Tests for public CR-pulse health-check exports."""

from qubex.contrib import (
    analyze_cr_pulse_health,
    characterize_cr_pulse_health,
    plot_cr_pulse_health,
)
from qubex.contrib.experiment import (
    analyze_cr_pulse_health as experiment_analyze_cr_pulse_health,
    characterize_cr_pulse_health as experiment_characterize_cr_pulse_health,
    plot_cr_pulse_health as experiment_plot_cr_pulse_health,
)


def test_health_check_functions_are_exported_from_contrib() -> None:
    """Both contrib namespaces should expose the three public workflow functions."""
    assert experiment_analyze_cr_pulse_health is analyze_cr_pulse_health
    assert experiment_characterize_cr_pulse_health is characterize_cr_pulse_health
    assert experiment_plot_cr_pulse_health is plot_cr_pulse_health
