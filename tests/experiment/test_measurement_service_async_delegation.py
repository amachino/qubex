"""Tests for async measurement delegation in experiment measurement service."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

from qxpulse import PulseSchedule

from qubex.experiment.services.measurement_service import MeasurementService
from qubex.measurement import SweepPoint, SweepValue


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
    ) -> str:
        built = None if not sweep_values else schedule(sweep_values[0])
        calls["run_sweep_measurement"].append(
            {
                "sweep_values": sweep_values,
                "schedule": schedule,
                "config": config,
                "built": built,
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
    """Given readout overrides, when running async measurement, then service builds schedule and delegates."""
    service, calls = _make_service()
    pulse_schedule = cast(Any, object())

    result = asyncio.run(
        service.run_measurement(
            pulse_schedule,
            frequencies={"Q00": 5.1},
            final_measurement=True,
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
    assert built_kwargs["final_measurement"] is True
    called = calls["run_measurement"][0]
    assert cast(SimpleNamespace, called["schedule"]).tag == "built"
    assert cast(SimpleNamespace, called["config"]).tag == "config"


def test_run_sweep_measurement_builds_wrapped_schedule_and_delegates() -> None:
    """Given sweep options, when running async sweep, then wrapped pulse schedule is built per point."""
    service, calls = _make_service()
    sweep_values: list[SweepValue] = [1, 2]

    def _schedule(value: SweepValue) -> PulseSchedule:
        return cast(Any, f"pulse-{value}")

    result = asyncio.run(
        service.run_sweep_measurement(
            _schedule,
            sweep_values=sweep_values,
            readout_amplification=True,
            shot_averaging=False,
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
    build_kwargs = calls["build_schedule"][0]
    assert build_kwargs["pulse_schedule"] == "pulse-1"
    assert build_kwargs["readout_amplification"] is True


def test_run_ndsweep_measurement_builds_wrapped_schedule_and_delegates() -> None:
    """Given ndsweep inputs, when running async ndsweep, then wrapped pulse schedule is built per point."""
    service, calls = _make_service()
    sweep_points: dict[str, Sequence[SweepValue]] = {"x": [1, 2], "y": [10]}

    def _schedule(point: SweepPoint) -> PulseSchedule:
        return cast(Any, f"{point['x']}-{point['y']}")

    result = asyncio.run(
        service.run_ndsweep_measurement(
            _schedule,
            sweep_points=sweep_points,
            sweep_axes=("x", "y"),
            final_measurement=True,
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
    assert build_kwargs["final_measurement"] is True
