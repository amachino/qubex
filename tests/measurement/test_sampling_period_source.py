"""Tests for sampling-period source resolution in measurement APIs."""

from __future__ import annotations

from qubex.backend.quel1 import SAMPLING_PERIOD
from qubex.measurement.measurement_client import MeasurementClient


def _make_measurement_client_with_backend(
    *,
    default_sampling_period: float | None,
) -> MeasurementClient:
    measurement = MeasurementClient(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )

    backend_controller = (
        type("_BC", (), {"DEFAULT_SAMPLING_PERIOD": default_sampling_period})()
        if default_sampling_period is not None
        else type("_BC", (), {})()
    )
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type("_CP", (), {"readout_amplitude": {}})(),
            "targets": [type("_Target", (), {"label": "Q00"})()],
        },
    )()
    measurement.__dict__["_backend_manager"] = type(
        "_BM",
        (),
        {
            "backend_controller": backend_controller,
            "experiment_system": experiment_system,
            "mux_dict": {},
        },
    )()
    return measurement


def test_sampling_period_uses_backend_controller_default() -> None:
    """Given backend dt, when resolving sampling period, then backend dt is returned."""
    measurement = _make_measurement_client_with_backend(default_sampling_period=4.0)

    assert measurement.sampling_period == 4.0


def test_sampling_period_falls_back_when_backend_default_is_missing() -> None:
    """Given backend without dt, when resolving sampling period, then legacy default is returned."""
    measurement = _make_measurement_client_with_backend(default_sampling_period=None)

    assert measurement.sampling_period == SAMPLING_PERIOD


def test_schedule_builder_is_initialized_with_resolved_sampling_period() -> None:
    """Given backend dt, when creating schedule builder, then builder carries the resolved period."""
    measurement = _make_measurement_client_with_backend(default_sampling_period=8.0)

    assert measurement.schedule_builder.sampling_period == 8.0
