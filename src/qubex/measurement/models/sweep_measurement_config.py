from __future__ import annotations

from typing import Literal

import numpy.typing as npt
import tunits

from qubex.core.model import ImmutableModel


class ParametricSequencePulseCommand(ImmutableModel):
    name: str
    channel_list: list[str]
    argument_list: list[str]


class ParametricSequenceConfig(ImmutableModel):
    delta_time: tunits.Time
    variable_list: list[str]
    command_list: list[ParametricSequencePulseCommand]


class FreequencyConfig(ImmutableModel):
    channel_to_frequency: dict[str, tunits.Frequency]
    channel_to_frequency_reference: dict[str, str]
    channel_to_frequency_shift: dict[str, tunits.Frequency]
    keep_oscillator_relative_phase: bool


class DataAcquisitionConfig(ImmutableModel):
    shot_count: int
    shot_repetition_margin: tunits.Time
    data_acquisition_duration: tunits.Time
    data_acquisition_delay: tunits.Time
    data_acquisition_timeout: tunits.Time
    flag_average_waveform: bool
    flag_average_shots: bool
    delta_time: tunits.Time
    channel_to_averaging_time: dict[str, tunits.Time]
    channel_to_averaging_window_coefficients: dict[str, npt.NDArray]


class ParameterSweepContent(ImmutableModel):
    category: Literal["frequency_shift", "sequence_variable"]
    sweep_target: list[str]
    value_list: list[float] | npt.NDArray


class ParameterSweepConfig(ImmutableModel):
    sweep_content_list: list[ParameterSweepContent]
    sweep_axis: list[list[str]]


class SweepMeasurementConfig(ImmutableModel):
    channel_list: list[str]
    sequence: ParametricSequenceConfig
    frequency: FreequencyConfig
    data_acquisition: DataAcquisitionConfig
    sweep_parameter: ParameterSweepConfig
