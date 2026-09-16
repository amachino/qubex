"""Tests for public CR dissipation exports."""

import qubex.contrib as contrib
import qubex.contrib.experiment as contrib_experiment
from qubex.contrib import (
    analyze_cr_dissipation,
    characterize_cr_dissipation,
    plot_cr_dissipation,
)
from qubex.contrib.experiment import (
    analyze_cr_dissipation as experiment_analyze_cr_dissipation,
    characterize_cr_dissipation as experiment_characterize_cr_dissipation,
    plot_cr_dissipation as experiment_plot_cr_dissipation,
)


def test_cr_dissipation_functions_are_exported_from_contrib() -> None:
    """Both contrib namespaces should expose the three public workflow functions."""
    assert experiment_analyze_cr_dissipation is analyze_cr_dissipation
    assert experiment_characterize_cr_dissipation is characterize_cr_dissipation
    assert experiment_plot_cr_dissipation is plot_cr_dissipation


def test_unreleased_legacy_names_are_not_exported() -> None:
    """The clean rename should not leave obsolete workflow aliases."""
    for namespace in (contrib, contrib_experiment):
        assert not hasattr(namespace, "characterize_cr_pulse_health")
        assert not hasattr(namespace, "analyze_cr_pulse_health")
        assert not hasattr(namespace, "plot_cr_pulse_health")
