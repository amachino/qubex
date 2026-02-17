"""Tests for backend-kind selection in measurement entrypoints."""

from __future__ import annotations

from typing import Any, cast

from qubex.measurement.measurement_backend_manager import MeasurementBackendManager
from qubex.measurement.measurement_client import MeasurementClient


def test_measurement_backend_manager_load_forwards_backend_kind() -> None:
    """Given backend kind input, when manager loads configs, then SystemManager receives the same kind."""
    called: dict[str, Any] = {}

    class _SystemManager:
        def load(self, **kwargs: object) -> None:
            called.update(kwargs)

    manager = MeasurementBackendManager(
        system_manager=cast(Any, _SystemManager()),
        qubits=["Q00"],
    )
    manager.load_skew_file = lambda: None  # type: ignore[method-assign]
    manager.load(
        chip_id="TEST",
        backend_kind="quel3",
    )

    assert called["backend_kind"] == "quel3"


def test_measurement_client_load_forwards_backend_kind() -> None:
    """Given backend kind input, when MeasurementClient loads, then backend manager receives the same kind."""
    measurement = MeasurementClient(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called: dict[str, Any] = {}

    class _BackendManager:
        def load(self, **kwargs: object) -> None:
            called.update(kwargs)

    measurement.__dict__["_backend_manager"] = _BackendManager()

    measurement.load(
        config_dir=None,
        params_dir=None,
        backend_kind="quel3",
    )

    assert called["backend_kind"] == "quel3"


def test_measurement_client_init_forwards_backend_kind_to_load(
    monkeypatch,
) -> None:
    """Given backend kind at init, when load is enabled, then MeasurementClient passes it to load()."""
    called: dict[str, object] = {}

    def _fake_load(
        self: MeasurementClient,
        *,
        config_dir: object,
        params_dir: object,
        configuration_mode: object = None,
        backend_kind: object = None,
    ) -> None:
        called["backend_kind"] = backend_kind

    monkeypatch.setattr(MeasurementClient, "load", _fake_load)

    MeasurementClient(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=True,
        connect_devices=False,
        backend_kind="quel3",
    )

    assert called["backend_kind"] == "quel3"
