"""Tests for installed distribution reporting."""

from types import SimpleNamespace
from typing import cast

from qubex.experiment import experiment_context


def test_environment_reports_current_distribution_names(monkeypatch) -> None:
    """Environment output resolves the renamed distributions without legacy lookups."""
    requested = []

    def get_version(name):
        requested.append(name)
        return "1.5.0rc4"

    monkeypatch.setattr(experiment_context, "get_version", get_version)
    context = SimpleNamespace(
        config_path="config",
        params_path="params",
        chip_id="test",
        chip=SimpleNamespace(name="test"),
        qubit_labels=[],
        mux_labels=[],
        box_ids=[],
    )
    experiment_context.ExperimentContext.print_environment(
        cast(experiment_context.ExperimentContext, context), verbose=True
    )

    assert {
        "qubex-core",
        "qubex-driver-quel1",
        "qubex-pulse",
        "qubex-schema",
        "qubex-simulator",
    } <= set(requested)
    assert not any(name.startswith("qx") for name in requested)
