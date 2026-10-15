"""Tests for compatibility imports after the qxsimulator package reorganization."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from inspect import signature
from pathlib import Path

import pytest


def test_qxsimulator_import_does_not_require_optimizer_dependencies() -> None:
    """Base package import should not require deprecated optimizer dependencies."""
    script = """
import builtins

blocked = {"IPython", "jax", "optax"}
original_import = builtins.__import__

def import_without_optimizer_dependencies(name, *args, **kwargs):
    if name.partition(".")[0] in blocked:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_optimizer_dependencies
import qxsimulator
import qubex.simulator
from qxsimulator import PulseOptimizer
from qxsimulator.optimization.optimization_result import OptimizationResult

assert qxsimulator.QuantumSimulator is not None
assert qubex.simulator.QuantumSimulator is not None
assert OptimizationResult is not None
assert PulseOptimizer is not None
"""

    subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )


def test_qxsimulator_metadata_excludes_optimizer_dependencies() -> None:
    """Package metadata should not install deprecated optimizer dependencies."""
    metadata = (
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("packages/qxsimulator/pyproject.toml")
        .read_text()
    )

    for dependency in ("ipython", "jax", "optax"):
        assert f'"{dependency} ' not in metadata.lower()


@pytest.mark.parametrize(
    ("legacy_module_name", "class_name", "public_module_name"),
    [
        ("qxsimulator.quantum_system", "QuantumSystem", "qxsimulator"),
        ("qxsimulator.quantum_simulator", "Control", "qxsimulator"),
        ("qxsimulator.quantum_simulator", "QuantumSimulator", "qxsimulator"),
        (
            "qxsimulator.quantum_simulator",
            "SimulationModel",
            "qxsimulator.simulation",
        ),
        ("qxsimulator.quantum_simulator", "SimulationResult", "qxsimulator"),
    ],
)
def test_released_module_imports_remain_compatible(
    legacy_module_name: str,
    class_name: str,
    public_module_name: str,
) -> None:
    """Released modules should preserve their public class imports."""
    legacy_module = import_module(legacy_module_name)
    public_module = import_module(public_module_name)

    assert getattr(legacy_module, class_name) is getattr(public_module, class_name)


def test_downsample_remains_private() -> None:
    """Downsample should remain internal to the simulation package."""
    sampling_module = import_module("qxsimulator.simulation._sampling")

    assert hasattr(sampling_module, "downsample")
    for module_name in (
        "qxsimulator.quantum_simulator",
        "qxsimulator.simulation",
        "qxsimulator.simulation.quantum_simulator",
        "qxsimulator.simulation.simulation_result",
    ):
        assert not hasattr(import_module(module_name), "downsample")


@pytest.mark.parametrize(
    "module_name",
    [
        "qxsimulator.quantum_simulator",
        "qxsimulator.simulation",
        "qxsimulator.simulation.quantum_simulator",
    ],
)
def test_simulate_uses_literal_default_without_time_step_export(
    module_name: str,
) -> None:
    """Simulate should default dt directly without exporting TIME_STEP."""
    module = import_module(module_name)
    parameters = signature(module.QuantumSimulator.simulate).parameters

    assert parameters["dt"].default == pytest.approx(0.1)
    assert not hasattr(module, "TIME_STEP")
