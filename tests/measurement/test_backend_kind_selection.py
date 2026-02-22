"""Tests for backend-kind selection in measurement entrypoints."""

from __future__ import annotations

from typing import Any, cast

from qubex.measurement.measurement import Measurement
from qubex.measurement.measurement_session_service import MeasurementSessionService


def test_measurement_session_service_load_forwards_backend_kind() -> None:
    """Given backend kind input, when session service loads configs, then SystemManager receives the same kind."""
    called: dict[str, Any] = {}

    class _SystemManager:
        def load(self, **kwargs: object) -> None:
            called.update(kwargs)

    session_service = MeasurementSessionService(
        system_manager=cast(Any, _SystemManager()),
        context=cast(Any, object()),
    )
    session_service.load_skew_file = lambda: None  # type: ignore[method-assign]
    session_service.load(
        chip_id="TEST",
        backend_kind="quel3",
    )

    assert called["backend_kind"] == "quel3"


def test_measurement_load_forwards_backend_kind() -> None:
    """Given backend kind input, when Measurement loads, then session service receives the same kind."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called: dict[str, Any] = {}

    class _SessionService:
        def load(self, **kwargs: object) -> None:
            called.update(kwargs)

    measurement.__dict__["_session_service"] = _SessionService()

    measurement.load(
        config_dir=None,
        params_dir=None,
        backend_kind="quel3",
    )

    assert called["backend_kind"] == "quel3"


def test_measurement_init_forwards_backend_kind_to_load(
    monkeypatch,
) -> None:
    """Given backend kind at init, when load is enabled, then Measurement passes it to load()."""
    called: dict[str, object] = {}

    def _fake_load(
        self: Measurement,
        *,
        config_dir: object,
        params_dir: object,
        configuration_mode: object = None,
        backend_kind: object = None,
    ) -> None:
        called["backend_kind"] = backend_kind

    monkeypatch.setattr(Measurement, "load", _fake_load)

    Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=True,
        connect_devices=False,
        backend_kind="quel3",
    )

    assert called["backend_kind"] == "quel3"
