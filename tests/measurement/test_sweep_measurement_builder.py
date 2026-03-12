"""Tests for sweep measurement schedule building."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from qxpulse import Rect, Waveform

from qubex.core import Time, ValueArrayLike
from qubex.measurement.sweep_measurement_builder import (
    SweepCommandContext,
    SweepMeasurementBuilder,
)
from qubex.schema import (
    DataAcquisitionConfig,
    FrequencyConfig,
    ParameterSweepConfig,
    ParameterSweepContent,
    ParametricSequenceConfig,
    ParametricSequencePulseCommand,
    SweepMeasurementConfig,
)


def _make_config(
    *,
    channel_list: list[str] | None = None,
    command_list: list[ParametricSequencePulseCommand] | None = None,
    averaging_channels: dict[str, np.ndarray] | None = None,
    averaging_times: dict[str, Time] | None = None,
) -> SweepMeasurementConfig:
    resolved_averaging_channels = cast(
        dict[str, ValueArrayLike],
        averaging_channels or {"Q00": np.asarray([1.0 + 0.0j])},
    )
    return SweepMeasurementConfig(
        channel_list=channel_list or ["Q00"],
        sequence=ParametricSequenceConfig(
            delta_time=Time(2.0, "ns"),
            variable_list=["amp"],
            command_list=command_list
            or [
                ParametricSequencePulseCommand(
                    name="Rect",
                    channel_list=["Q00"],
                    argument_list=[10.0, "amp"],
                )
            ],
        ),
        frequency=FrequencyConfig(
            channel_to_frequency={},
            channel_to_frequency_reference={},
            channel_to_frequency_shift={},
            keep_oscillator_relative_phase=False,
        ),
        data_acquisition=DataAcquisitionConfig(
            shot_count=16,
            shot_repetition_margin=Time(100.0, "ns"),
            data_acquisition_duration=Time(4.0, "ns"),
            data_acquisition_delay=Time(2.0, "ns"),
            data_acquisition_timeout=Time(10.0, "ms"),
            flag_average_waveform=False,
            flag_average_shots=True,
            delta_time=Time(2.0, "ns"),
            channel_to_averaging_time=averaging_times
            or {channel: Time(4.0, "ns") for channel in resolved_averaging_channels},
            channel_to_averaging_window=resolved_averaging_channels,
        ),
        sweep_parameter=ParameterSweepConfig(
            sweep_content_list={
                "amp": ParameterSweepContent(
                    category="sequence_variable",
                    sweep_target=["amp"],
                    value_list=[0.1, 0.2],
                )
            },
            sweep_axis=[["amp"]],
        ),
    )


def test_build_measurement_schedule_uses_acquisition_delay_and_duration() -> None:
    """Given sweep config, when building measurement schedule, then capture windows follow acquisition delay and duration."""
    builder = SweepMeasurementBuilder(config=_make_config())

    schedule = builder.build_measurement_schedule(indices=(0,))

    assert schedule.pulse_schedule.duration == 10.0
    assert len(schedule.capture_schedule.captures) == 1
    capture = schedule.capture_schedule.captures[0]
    assert capture.channels == ["Q00"]
    assert capture.start_time == 2.0
    assert capture.duration == 4.0


def test_custom_command_registry_factory_receives_context() -> None:
    """Given custom command factory, when building schedule, then resolved context is passed to the factory."""
    called: dict[str, object] = {}

    def _factory(context: SweepCommandContext) -> Waveform:
        called["command"] = context.command
        called["args"] = context.resolved_argument_list
        called["variables"] = dict(context.sequence_variables)
        return Rect(
            duration=context.resolved_argument_list[0],
            amplitude=context.resolved_argument_list[1],
        )

    builder = SweepMeasurementBuilder(
        config=_make_config(
            command_list=[
                ParametricSequencePulseCommand(
                    name="MyRect",
                    channel_list=["Q00"],
                    argument_list=[10.0, "amp"],
                )
            ]
        ),
        command_registry={"MyRect": _factory},
    )

    schedule = builder.build_measurement_schedule(indices=(1,))

    assert schedule.pulse_schedule.duration == 10.0
    assert cast(tuple[float, ...], called["args"]) == (10.0, 0.2)
    assert cast(dict[str, float], called["variables"]) == {"amp": 0.2}


def test_reserved_command_override_is_rejected() -> None:
    """Given reserved command override, when creating builder, then validation fails."""
    with pytest.raises(ValueError, match="reserved"):
        _ = SweepMeasurementBuilder(
            config=_make_config(),
            command_registry={"Blank": lambda _context: None},
        )


def test_missing_capture_channel_pulse_raises() -> None:
    """Given capture channel without pulse range, when building measurement schedule, then validation fails."""
    builder = SweepMeasurementBuilder(
        config=_make_config(
            channel_list=["Q00", "Q01"],
            averaging_channels={"Q01": np.asarray([1.0 + 0.0j])},
            averaging_times={"Q01": Time(4.0, "ns")},
        )
    )

    with pytest.raises(ValueError, match="No pulse ranges"):
        _ = builder.build_measurement_schedule(indices=(0,))
