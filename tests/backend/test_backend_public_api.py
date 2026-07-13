"""Tests for backend package public API boundaries."""

from __future__ import annotations

import subprocess
import sys
from typing import get_args

import qubex.backend as backend
from qubex.backend.quel1 import (
    CAPTURE_DECIMATION_FACTOR as QUEL1_DECIMATION_FACTOR,
    SAMPLING_PERIOD_NS,
    ExecutionMode,
    Quel1BackendController,
    Quel1BackendExecutionResult,
    Quel1ExecutionPayload,
)
from qubex.backend.quel3 import Quel3BackendController


def test_backend_module_hides_quel1_specific_symbols() -> None:
    """Given backend module, when checking QuEL-1 symbols, then they are not re-exported."""
    assert not hasattr(backend, "Quel1BackendController")
    assert not hasattr(backend, "Quel1ExecutionPayload")
    assert not hasattr(backend, "Quel1BackendExecutionResult")
    assert not hasattr(backend, "CAPTURE_DECIMATION_FACTOR")
    assert not hasattr(backend, "SAMPLING_PERIOD_NS")


def test_backend_quel1_module_exposes_quel1_specific_symbols() -> None:
    """Given backend.quel1 module, when importing symbols, then QuEL-1 symbols are exposed."""
    assert Quel1BackendController.__name__ == "Quel1BackendController"
    assert Quel1ExecutionPayload.__name__ == "Quel1ExecutionPayload"
    assert Quel1BackendExecutionResult.__name__ == "Quel1BackendExecutionResult"
    assert isinstance(SAMPLING_PERIOD_NS, float)
    assert set(get_args(ExecutionMode)) == {"serial", "parallel"}


def test_backend_modules_expose_decimation_factor_constants() -> None:
    """Given backend controllers, when reading decimation constants, then values are positive integers."""
    assert QUEL1_DECIMATION_FACTOR > 0
    assert Quel3BackendController.CAPTURE_DECIMATION_FACTOR > 0


def test_backend_quel3_module_hides_module_level_decimation_constant() -> None:
    """Given backend.quel3 module, when checking exported symbols, then module-level decimation constant is not re-exported."""
    import qubex.backend.quel3 as quel3

    assert not hasattr(quel3, "CAPTURE_DECIMATION_FACTOR")


def test_backend_quel1_module_hides_migrated_system_defaults() -> None:
    """Given backend.quel1 module, when checking migrated defaults, then system-level constants are not re-exported."""
    import qubex.backend.quel1 as quel1

    assert not hasattr(quel1, "DEFAULT_PUMP_FREQUENCY_GHZ")
    assert not hasattr(quel1, "DEFAULT_LO_FREQUENCY_HZ")
    assert not hasattr(quel1, "LO_STEP_HZ")
    assert not hasattr(quel1, "EXTRA_SUM_SECTION_LENGTH")


def test_backend_quel3_module_hides_migrated_system_defaults() -> None:
    """Given backend.quel3 module, when checking migrated defaults, then system-level constants are not re-exported."""
    import qubex.backend.quel3 as quel3

    assert not hasattr(quel3, "DEFAULT_PUMP_FREQUENCY_GHZ")


def test_quel3_import_and_controller_init_do_not_require_qxdriver_dependency() -> None:
    """Given missing qxdriver dependency, QuEL-3 import and init should still succeed."""
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("qxdriver_quel1"):
        raise ModuleNotFoundError(name)
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import qubex.backend.quel3 as quel3

controller = quel3.Quel3BackendController()
assert controller.sampling_period_ns > 0
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_quel1_import_and_controller_init_do_not_require_quelware_client() -> None:
    """Given missing quelware-client dependency, QuEL-1 import and init should still succeed."""
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("quelware_client"):
        raise ModuleNotFoundError(name)
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import qubex.backend.quel1 as quel1

controller = quel1.Quel1BackendController()
assert controller.sampling_period_ns > 0
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_quantum_simulator_import_does_not_require_quel1_or_quel3_dependencies() -> (
    None
):
    """Given missing backend extras, QuantumSimulator import should still succeed."""
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("qxdriver_quel1") or name.startswith("quelware_client"):
        raise ModuleNotFoundError(name)
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from qubex.simulator import QuantumSimulator

assert QuantumSimulator.__name__ == "QuantumSimulator"
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
