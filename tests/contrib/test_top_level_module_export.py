"""Tests for top-level contrib module export."""

import importlib
import sys

import qubex
from qubex import contrib


def test_contrib_module_is_exported_from_qubex() -> None:
    """`qubex` should export the `contrib` module."""
    assert "contrib" in qubex.__all__
    assert callable(contrib.measure_cr_crosstalk)
    assert callable(contrib.simultaneous_coherence_measurement)
    assert callable(contrib.purity_benchmarking)
    assert callable(contrib.build_gmm_linear_line_param)
    assert callable(contrib.build_gmm_linear_line_params)


def test_importing_gmm_linear_helper_does_not_eagerly_import_experiment_modules(
    monkeypatch,
) -> None:
    """Importing the DSP helper should not pull in contrib experiment modules."""
    for name in [
        "qubex.contrib",
        "qubex.contrib.experiment",
        "qubex.contrib.experiment.crosstalk_cross_resonance",
        "qubex.contrib.gmm_linear_classification",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("qubex.contrib.gmm_linear_classification")

    assert callable(module.build_gmm_linear_line_param)
    assert "qubex.contrib.experiment" not in sys.modules
    assert "qubex.contrib.experiment.crosstalk_cross_resonance" not in sys.modules
