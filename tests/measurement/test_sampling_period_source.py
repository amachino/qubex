"""Tests for sampling-period source resolution in measurement APIs."""

from __future__ import annotations

from qubex.measurement.measurement import Measurement
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)


def _make_measurement_with_backend(
    monkeypatch,
    *,
    sampling_period: float,
    backend_kind: str = "quel1",
) -> Measurement:
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )

    if backend_kind == "quel3":
        _Quel3Controller = type(
            "_Quel3Controller",
            (),
            {"sampling_period": sampling_period},
        )

        monkeypatch.setattr(
            "qubex.measurement.services.measurement_execution_service.Quel3BackendController",
            _Quel3Controller,
        )
        backend_controller = _Quel3Controller()
    else:
        _Quel1Controller = type(
            "_Quel1Controller",
            (),
            {"sampling_period": sampling_period},
        )

        monkeypatch.setattr(
            "qubex.measurement.services.measurement_execution_service.Quel1BackendController",
            _Quel1Controller,
        )
        backend_controller = _Quel1Controller()
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type("_CP", (), {"readout_amplitude": {}})(),
            "targets": [type("_Target", (), {"label": "Q00"})()],
        },
    )()
    context = type(
        "_CTX",
        (),
        {
            "backend_controller": backend_controller,
            "experiment_system": experiment_system,
            "mux_dict": {},
        },
    )()
    session_service = type(
        "_SS",
        (),
        {
            "backend_controller": backend_controller,
        },
    )()
    measurement.__dict__["_context"] = context
    measurement.__dict__["_session_service"] = session_service
    measurement.execution_service.__dict__["_context"] = context
    measurement.execution_service.__dict__["_session_service"] = session_service
    return measurement


def test_sampling_period_uses_backend_controller_default(monkeypatch) -> None:
    """Given backend dt, when resolving sampling period, backend dt is returned."""
    measurement = _make_measurement_with_backend(
        monkeypatch,
        sampling_period=4.0,
    )

    assert measurement.sampling_period == 4.0


def test_schedule_builder_is_initialized_with_resolved_sampling_period(
    monkeypatch,
) -> None:
    """Given backend dt, when creating schedule builder, then builder carries the resolved period."""
    measurement = _make_measurement_with_backend(
        monkeypatch,
        sampling_period=8.0,
    )

    assert measurement.schedule_builder.sampling_period == 8.0


def test_constraint_profile_uses_quel3_backend_type(monkeypatch) -> None:
    """Given quel3 backend type, when resolving profile, then quel3 constraints are returned."""
    measurement = _make_measurement_with_backend(
        monkeypatch,
        sampling_period=0.4,
        backend_kind="quel3",
    )

    profile = measurement.constraint_profile

    assert isinstance(profile, MeasurementConstraintProfile)
    assert profile.sampling_period_ns == 0.4
    assert profile.enforce_block_alignment is False
    assert profile.require_workaround_capture is False
