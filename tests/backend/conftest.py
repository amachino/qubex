"""Fixtures for isolated backend lifecycle tests."""

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qubex.system import SystemManager
from tests._backend_dependencies import require_distributions, require_quel1_backend


@pytest.fixture
def require_quel1_driver() -> None:
    """Skip real-driver tests when the selected QuEL-1 backend is not installed."""
    require_quel1_backend()


@pytest.fixture
def require_quel1_waveforms() -> None:
    """Require the real QuEL-1 AWG parameter models for waveform integration tests."""
    require_distributions("quel-ic-config")


def _unexpected_driver_call(*args: Any, **kwargs: Any) -> Any:
    """Reject driver calls that a controller unit test has not stubbed."""
    raise AssertionError("Provide a driver stub for this operation")


def _stub_qubecalib() -> SimpleNamespace:
    """Expose the configuration boundary without creating driver objects."""
    return SimpleNamespace(
        system_config_database=SimpleNamespace(create_box=_unexpected_driver_call),
    )


@dataclass(frozen=True)
class _Quel1DriverStub:
    """Supply constructor defaults and explicit replacement points for unit tests."""

    DEFAULT_SAMPLING_PERIOD: float = 2.0
    QubeCalib: Any = _stub_qubecalib
    BoxPool: Any = _unexpected_driver_call
    SequencerClient: Any = _unexpected_driver_call
    QuBEMasterClient: Any = _unexpected_driver_call
    Quel1ConfigOption: Any = _unexpected_driver_call


@pytest.fixture
def stub_quel1_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a fresh driver stub in controller tests without loading optional packages."""
    driver = _Quel1DriverStub()
    monkeypatch.setattr(
        "qubex.backend.quel1.quel1_runtime_context.load_quel1_driver",
        lambda: driver,
    )


@pytest.fixture
def quel3_http_transport() -> ModuleType:
    """Load the real HTTP transport only for tests requiring its optional libraries."""
    require_distributions("grpclib", "protobuf", "googleapis-common-protos")
    return import_module("qubex.backend.quel3.infra.quelware_http_transport")


@pytest.fixture
def fresh_system_manager(monkeypatch: pytest.MonkeyPatch) -> SystemManager:
    """Create fresh singleton state and restore the original instance afterward."""
    monkeypatch.setattr(SystemManager, "_instance", None)
    return SystemManager.shared()
