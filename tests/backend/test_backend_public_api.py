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


def test_experiment_import_does_not_load_backend_driver_dependencies() -> None:
    """Given Experiment import, when loading the facade, then backend drivers stay unloaded."""
    code = """
import sys

from qubex import Experiment

assert Experiment.__name__ == "Experiment"
loaded_backend_drivers = [
    name
    for name in sys.modules
    if name.startswith(("qxdriver_quel1", "quel_ic_config", "quelware_client"))
]
assert loaded_backend_drivers == [], loaded_backend_drivers
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_quel1_package_import_defers_driver_loading_until_controller_init() -> None:
    """Given QuEL-1 package import, when creating its controller, then drivers load lazily."""
    code = """
import sys

import qubex.backend.quel1 as quel1

loaded_quel1_drivers = [
    name
    for name in sys.modules
    if name.startswith(("qxdriver_quel1", "quel_ic_config", "qubecalib"))
]
assert loaded_quel1_drivers == [], loaded_quel1_drivers

controller = quel1.Quel1BackendController()
assert controller.driver.package_name in sys.modules
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_quel3_selection_does_not_initialize_quel1_driver() -> None:
    """Given QuEL-3 selection, when initializing its controller, then QuEL-1 drivers stay unloaded."""
    code = """
import sys

from qubex.system import SystemManager

manager = SystemManager.shared()
manager.set_backend_kind("quel3")

assert manager.backend_kind == "quel3"
loaded_quel1_drivers = [
    name
    for name in sys.modules
    if name.startswith(("qxdriver_quel1", "quel_ic_config", "qubecalib"))
]
assert loaded_quel1_drivers == [], loaded_quel1_drivers
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_system_manager_has_no_backend_before_selection() -> None:
    """Given no backend selection, when reading backend state, then only the controller is unavailable."""
    code = """
import sys

from qubex.system import SystemManager

manager = SystemManager.shared()
loaded_quel1_drivers = [
    name
    for name in sys.modules
    if name.startswith(("qxdriver_quel1", "quel_ic_config", "qubecalib"))
]
assert loaded_quel1_drivers == [], loaded_quel1_drivers

assert manager.backend_kind is None
try:
    manager.backend_controller
except RuntimeError as exc:
    assert "not initialized" in str(exc)
else:
    raise AssertionError("backend_controller unexpectedly available")

loaded_quel1_drivers = [
    name
    for name in sys.modules
    if name.startswith(("qxdriver_quel1", "quel_ic_config", "qubecalib"))
]
assert loaded_quel1_drivers == [], loaded_quel1_drivers
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
