"""Tests for async measurement delegation in experiment measurement service."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import tunits.units as tunits_units
from qxpulse import PulseSchedule

import qubex.experiment.services.measurement_service as measurement_service_module
from qubex.experiment.services.measurement_service import MeasurementService
from qubex.measurement import MeasurementSchedule, SweepPoint, SweepValue
from qubex.measurement.models.capture_schedule import CaptureSchedule


def _make_service() -> tuple[MeasurementService, dict[str, Any]]:
    calls: dict[str, Any] = {
        "build_schedule": [],
        "create_config": [],
        "run_measurement": [],
        "run_sweep_measurement": [],
        "run_ndsweep_measurement": [],
    }

    def _build_measurement_schedule(**kwargs: Any) -> object:
        calls["build_schedule"].append(kwargs)
        return SimpleNamespace(tag="built", source=kwargs["pulse_schedule"])

    def _create_measurement_config(**kwargs: Any) -> object:
        calls["create_config"].append(kwargs)
        return SimpleNamespace(tag="config")

    async def _run_measurement(*, schedule: object, config: object) -> str:
        calls["run_measurement"].append({"schedule": schedule, "config": config})
        return "measurement_result"

    async def _run_sweep_measurement(
        schedule: Any,
        *,
        sweep_values: list[object],
        config: object,
        on_point: Any = None,
    ) -> str:
        built = None if not sweep_values else schedule(sweep_values[0])
        calls["run_sweep_measurement"].append(
            {
                "sweep_values": sweep_values,
                "schedule": schedule,
                "config": config,
                "built": built,
                "on_point": on_point,
            }
        )
        return "sweep_result"

    async def _run_ndsweep_measurement(
        schedule: Any,
        *,
        sweep_points: dict[str, list[object]],
        sweep_axes: tuple[str, ...] | None,
        config: object,
    ) -> str:
        first_point = {axis: values[0] for axis, values in sweep_points.items()}
        built = schedule(first_point)
        calls["run_ndsweep_measurement"].append(
            {
                "sweep_points": sweep_points,
                "sweep_axes": sweep_axes,
                "schedule": schedule,
                "config": config,
                "built": built,
            }
        )
        return "ndsweep_result"

    measurement = SimpleNamespace(
        build_measurement_schedule=_build_measurement_schedule,
        create_measurement_config=_create_measurement_config,
        run_measurement=_run_measurement,
        run_sweep_measurement=_run_sweep_measurement,
        run_ndsweep_measurement=_run_ndsweep_measurement,
    )

    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = SimpleNamespace(
        measurement=measurement,
    )
    service.__dict__["_pulse_service"] = SimpleNamespace()
    return cast(MeasurementService, service), calls


def _measurement_schedule() -> MeasurementSchedule:
    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.barrier()
    return MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )


def test_build_measurement_schedule_delegates_none_values() -> None:
    """Given omitted optional flags, when building schedule, then None values are delegated downstream."""
    service, calls = _make_service()
    pulse_schedule = object()

    result = service.build_measurement_schedule(cast(Any, pulse_schedule))

    assert cast(SimpleNamespace, result).tag == "built"
    kwargs = calls["build_schedule"][0]
    assert kwargs["pulse_schedule"] is pulse_schedule
    assert kwargs["readout_amplification"] is None
    assert kwargs["final_measurement"] is None
    assert kwargs["capture_placement"] is None
    assert kwargs["plot"] is None


def test_run_measurement_builds_schedule_and_delegates() -> None:
    """Given explicit final_measurement false, when running async measurement, then service preserves explicit value."""
    service, calls = _make_service()
    pulse_schedule = cast(Any, object())

    result = asyncio.run(
        service.run_measurement(
            pulse_schedule,
            frequencies={"Q00": 5.1},
            final_measurement=False,
            n_shots=256,
        )
    )

    assert result == "measurement_result"
    assert len(calls["create_config"]) == 1
    assert calls["create_config"][0] == {
        "n_shots": 256,
        "shot_interval": None,
        "shot_averaging": None,
        "time_integration": None,
        "state_classification": None,
    }
    built_kwargs = calls["build_schedule"][0]
    assert built_kwargs["pulse_schedule"] is pulse_schedule
    assert built_kwargs["frequencies"] == {"Q00": 5.1}
    assert built_kwargs["final_measurement"] is False
    called = calls["run_measurement"][0]
    assert cast(SimpleNamespace, called["schedule"]).tag == "built"
    assert cast(SimpleNamespace, called["config"]).tag == "config"


def test_run_measurement_normalizes_tunits_inputs_before_delegation() -> None:
    """Given tunits values, when running async measurement, then shot interval and frequencies are normalized to float units."""
    service, calls = _make_service()
    pulse_schedule = cast(Any, object())

    result = asyncio.run(
        service.run_measurement(
            pulse_schedule,
            shot_interval=2 * tunits_units.us,
            frequencies={"Q00": 5100 * tunits_units.MHz},
            readout_duration=4 * tunits_units.us,
            readout_pre_margin=80 * tunits_units.ns,
            readout_post_margin=120 * tunits_units.ns,
            readout_ramp_time=40 * tunits_units.ns,
        )
    )

    assert result == "measurement_result"
    assert calls["create_config"][0]["shot_interval"] == pytest.approx(2000.0)
    assert calls["build_schedule"][0]["frequencies"] == {"Q00": pytest.approx(5.1)}
    assert calls["build_schedule"][0]["readout_duration"] == pytest.approx(4000.0)
    assert calls["build_schedule"][0]["readout_pre_margin"] == pytest.approx(80.0)
    assert calls["build_schedule"][0]["readout_post_margin"] == pytest.approx(120.0)
    assert calls["build_schedule"][0]["readout_ramp_time"] == pytest.approx(40.0)


def test_run_measurement_uses_measurement_schedule_without_rebuild() -> None:
    """Given MeasurementSchedule input, when running async measurement, then service bypasses schedule build."""
    service, calls = _make_service()
    schedule = _measurement_schedule()

    result = asyncio.run(
        service.run_measurement(
            schedule,
            shot_interval=2 * tunits_units.us,
            frequencies={"Q00": 5100 * tunits_units.MHz},
            readout_duration=4 * tunits_units.us,
        )
    )

    assert result == "measurement_result"
    assert calls["build_schedule"] == []
    assert calls["run_measurement"][0]["schedule"] is schedule
    assert calls["create_config"][0]["shot_interval"] == pytest.approx(2000.0)


def test_run_sweep_measurement_builds_wrapped_schedule_and_delegates() -> None:
    """Given explicit final_measurement false, when running async sweep, then service preserves explicit value."""
    service, calls = _make_service()
    sweep_values: list[SweepValue] = [1, 2]

    def _schedule(value: SweepValue) -> PulseSchedule:
        return cast(Any, f"pulse-{value}")

    result = asyncio.run(
        service.run_sweep_measurement(
            _schedule,
            sweep_values=sweep_values,
            readout_amplification=True,
            final_measurement=False,
            shot_averaging=False,
            plot=False,
        )
    )

    assert result == "sweep_result"
    assert len(calls["create_config"]) == 1
    assert calls["create_config"][0] == {
        "n_shots": None,
        "shot_interval": None,
        "shot_averaging": False,
        "time_integration": None,
        "state_classification": None,
    }
    called = calls["run_sweep_measurement"][0]
    assert called["sweep_values"] == sweep_values
    assert cast(SimpleNamespace, called["config"]).tag == "config"
    assert cast(SimpleNamespace, called["built"]).source == "pulse-1"
    assert callable(called["on_point"])
    build_kwargs = calls["build_schedule"][0]
    assert build_kwargs["pulse_schedule"] == "pulse-1"
    assert build_kwargs["readout_amplification"] is True
    assert build_kwargs["final_measurement"] is False


def test_run_sweep_measurement_normalizes_tunits_inputs_before_delegation() -> None:
    """Given tunits values, when running async sweep, then interval, frequencies, and readout timings are normalized to float units."""
    service, calls = _make_service()
    sweep_values: list[SweepValue] = [1]

    def _schedule(value: SweepValue) -> PulseSchedule:
        return cast(Any, f"pulse-{value}")

    result = asyncio.run(
        service.run_sweep_measurement(
            _schedule,
            sweep_values=sweep_values,
            shot_interval=2 * tunits_units.us,
            frequencies={"Q00": 5100 * tunits_units.MHz},
            readout_duration=4 * tunits_units.us,
            readout_pre_margin=80 * tunits_units.ns,
            readout_post_margin=120 * tunits_units.ns,
            readout_ramp_time=40 * tunits_units.ns,
            plot=False,
        )
    )

    assert result == "sweep_result"
    assert calls["create_config"][0]["shot_interval"] == pytest.approx(2000.0)
    build_kwargs = calls["build_schedule"][0]
    assert build_kwargs["frequencies"] == {"Q00": pytest.approx(5.1)}
    assert build_kwargs["readout_duration"] == pytest.approx(4000.0)
    assert build_kwargs["readout_pre_margin"] == pytest.approx(80.0)
    assert build_kwargs["readout_post_margin"] == pytest.approx(120.0)
    assert build_kwargs["readout_ramp_time"] == pytest.approx(40.0)


def test_run_sweep_measurement_uses_measurement_schedule_without_rebuild() -> None:
    """Given MeasurementSchedule callback output, when running async sweep, then service bypasses schedule build."""
    service, calls = _make_service()
    sweep_values: list[SweepValue] = [1]
    schedule = _measurement_schedule()

    def _schedule(_: SweepValue) -> MeasurementSchedule:
        return schedule

    result = asyncio.run(
        service.run_sweep_measurement(
            _schedule,
            sweep_values=sweep_values,
            shot_interval=2 * tunits_units.us,
            frequencies={"Q00": 5100 * tunits_units.MHz},
            readout_duration=4 * tunits_units.us,
            plot=False,
        )
    )

    assert result == "sweep_result"
    assert calls["build_schedule"] == []
    assert calls["run_sweep_measurement"][0]["built"] is schedule
    assert callable(calls["run_sweep_measurement"][0]["on_point"])
    assert calls["create_config"][0]["shot_interval"] == pytest.approx(2000.0)


def test_run_sweep_measurement_plots_iq_and_updates_tqdm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given plot+tqdm enabled, when running async sweep, then IQ plot and progress updates are executed."""
    service, calls = _make_service()
    calls["plot"] = {}
    calls["progress"] = {}
    service.ctx.state_centers = {"Q00": {0: 0.0 + 0.0j}}  # type: ignore[attr-defined]

    class _Plotter:
        def __init__(self, state_centers: dict[str, dict[int, complex]]) -> None:
            calls["plot"]["state_centers"] = state_centers

        def update(self, data: dict[str, np.ndarray]) -> None:
            calls["plot"]["data"] = data

        def show(self) -> None:
            calls["plot"]["shown"] = True

    class _Progress:
        def __init__(self, *, total: int, desc: str, disable: bool) -> None:
            calls["progress"]["total"] = total
            calls["progress"]["desc"] = desc
            calls["progress"]["disable"] = disable
            calls["progress"]["updates"] = 0
            calls["progress"]["closed"] = False

        def update(self, step: int) -> None:
            calls["progress"]["updates"] += step

        def close(self) -> None:
            calls["progress"]["closed"] = True

    async def _run_sweep_measurement(
        schedule: Any,
        *,
        sweep_values: list[object],
        config: object,
        on_point: Any = None,
    ) -> Any:
        _ = config
        for value in sweep_values:
            _ = schedule(value)
            if on_point is not None:
                on_point(
                    value,
                    SimpleNamespace(
                        data={
                            "Q00": [
                                SimpleNamespace(
                                    data=np.array([1.0 + 2.0j]),
                                )
                            ]
                        }
                    ),
                )
        return SimpleNamespace(results=[], sweep_values=sweep_values)

    service.ctx.measurement.run_sweep_measurement = _run_sweep_measurement  # type: ignore[attr-defined]
    monkeypatch.setattr(measurement_service_module, "IQPlotter", _Plotter)
    monkeypatch.setattr(measurement_service_module, "tqdm", _Progress)

    def _schedule(value: SweepValue) -> PulseSchedule:
        return cast(Any, f"pulse-{value}")

    _ = asyncio.run(
        service.run_sweep_measurement(
            _schedule,
            sweep_values=[1, 2],
            plot=True,
            enable_tqdm=True,
        )
    )

    assert calls["progress"] == {
        "total": 2,
        "desc": "Sweeping parameters",
        "disable": False,
        "updates": 2,
        "closed": True,
    }
    assert calls["plot"]["state_centers"] == {"Q00": {0: 0.0 + 0.0j}}
    plotted = calls["plot"]["data"]["Q00"]
    assert np.asarray(plotted).shape == (2,)
    assert np.asarray(plotted)[0] == pytest.approx(1.0 + 2.0j)
    assert np.asarray(plotted)[1] == pytest.approx(1.0 + 2.0j)
    assert calls["plot"]["shown"] is True


def test_run_ndsweep_measurement_builds_wrapped_schedule_and_delegates() -> None:
    """Given explicit final_measurement false, when running async ndsweep, then service preserves explicit value."""
    service, calls = _make_service()
    sweep_points: dict[str, Sequence[SweepValue]] = {"x": [1, 2], "y": [10]}

    def _schedule(point: SweepPoint) -> PulseSchedule:
        return cast(Any, f"{point['x']}-{point['y']}")

    result = asyncio.run(
        service.run_ndsweep_measurement(
            _schedule,
            sweep_points=sweep_points,
            sweep_axes=("x", "y"),
            final_measurement=False,
            state_classification=True,
        )
    )

    assert result == "ndsweep_result"
    assert len(calls["create_config"]) == 1
    assert calls["create_config"][0] == {
        "n_shots": None,
        "shot_interval": None,
        "shot_averaging": None,
        "time_integration": None,
        "state_classification": True,
    }
    called = calls["run_ndsweep_measurement"][0]
    assert called["sweep_axes"] == ("x", "y")
    assert cast(SimpleNamespace, called["config"]).tag == "config"
    assert cast(SimpleNamespace, called["built"]).source == "1-10"
    build_kwargs = calls["build_schedule"][0]
    assert build_kwargs["pulse_schedule"] == "1-10"
    assert build_kwargs["final_measurement"] is False


def test_run_ndsweep_measurement_normalizes_tunits_inputs_before_delegation() -> None:
    """Given tunits values, when running async ndsweep, then interval, frequencies, and readout timings are normalized to float units."""
    service, calls = _make_service()
    sweep_points: dict[str, Sequence[SweepValue]] = {"x": [1], "y": [10]}

    def _schedule(point: SweepPoint) -> PulseSchedule:
        return cast(Any, f"{point['x']}-{point['y']}")

    result = asyncio.run(
        service.run_ndsweep_measurement(
            _schedule,
            sweep_points=sweep_points,
            shot_interval=2 * tunits_units.us,
            frequencies={"Q00": 5100 * tunits_units.MHz},
            readout_duration=4 * tunits_units.us,
            readout_pre_margin=80 * tunits_units.ns,
            readout_post_margin=120 * tunits_units.ns,
            readout_ramp_time=40 * tunits_units.ns,
        )
    )

    assert result == "ndsweep_result"
    assert calls["create_config"][0]["shot_interval"] == pytest.approx(2000.0)
    build_kwargs = calls["build_schedule"][0]
    assert build_kwargs["frequencies"] == {"Q00": pytest.approx(5.1)}
    assert build_kwargs["readout_duration"] == pytest.approx(4000.0)
    assert build_kwargs["readout_pre_margin"] == pytest.approx(80.0)
    assert build_kwargs["readout_post_margin"] == pytest.approx(120.0)
    assert build_kwargs["readout_ramp_time"] == pytest.approx(40.0)


def test_run_ndsweep_measurement_uses_measurement_schedule_without_rebuild() -> None:
    """Given MeasurementSchedule callback output, when running async ndsweep, then service bypasses schedule build."""
    service, calls = _make_service()
    sweep_points: dict[str, Sequence[SweepValue]] = {"x": [1], "y": [10]}
    schedule = _measurement_schedule()

    def _schedule(_: SweepPoint) -> MeasurementSchedule:
        return schedule

    result = asyncio.run(
        service.run_ndsweep_measurement(
            _schedule,
            sweep_points=sweep_points,
            shot_interval=2 * tunits_units.us,
            frequencies={"Q00": 5100 * tunits_units.MHz},
            readout_duration=4 * tunits_units.us,
        )
    )

    assert result == "ndsweep_result"
    assert calls["build_schedule"] == []
    assert calls["run_ndsweep_measurement"][0]["built"] is schedule
    assert calls["create_config"][0]["shot_interval"] == pytest.approx(2000.0)
