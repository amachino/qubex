"""Tests for the public CR dissipation export."""

import qubex.contrib as contrib
import qubex.contrib.experiment as contrib_experiment
from qubex.contrib import characterize_cr_dissipation
from qubex.contrib.experiment import (
    characterize_cr_dissipation as experiment_characterize_cr_dissipation,
)


def test_only_characterize_cr_dissipation_is_exported() -> None:
    """Only the characterized workflow should be part of the public API."""
    assert experiment_characterize_cr_dissipation is characterize_cr_dissipation
    for namespace in (contrib, contrib_experiment):
        assert "characterize_cr_dissipation" in namespace.__all__
        assert "analyze_cr_dissipation" not in namespace.__all__
        assert "plot_cr_dissipation" not in namespace.__all__
        assert not hasattr(namespace, "analyze_cr_dissipation")
        assert not hasattr(namespace, "plot_cr_dissipation")


def test_unreleased_legacy_names_are_not_exported() -> None:
    """The clean rename should not leave obsolete workflow aliases."""
    for namespace in (contrib, contrib_experiment):
        assert not hasattr(namespace, "characterize_cr_pulse_health")
        assert not hasattr(namespace, "analyze_cr_pulse_health")
        assert not hasattr(namespace, "plot_cr_pulse_health")
