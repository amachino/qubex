from __future__ import annotations

from typing import Literal

import tunits

from qubex.core.model import Model
from qubex.core.typing import ValueArrayLike


class ParametricSequencePulseCommand(Model):
    name: str
    channel_list: list[str]
    argument_list: list[str | float]


class ParametricSequenceConfig(Model):
    delta_time: tunits.Time
    variable_list: list[str]
    command_list: list[ParametricSequencePulseCommand]


class FrequencyConfig(Model):
    channel_to_frequency: dict[str, tunits.Frequency]
    channel_to_frequency_reference: dict[str, str]
    channel_to_frequency_shift: dict[str, tunits.Frequency]
    keep_oscillator_relative_phase: bool


class DataAcquisitionConfig(Model):
    shot_count: int
    shot_repetition_margin: tunits.Time
    data_acquisition_duration: tunits.Time
    data_acquisition_delay: tunits.Time
    data_acquisition_timeout: tunits.Time
    flag_average_waveform: bool
    flag_average_shots: bool
    delta_time: tunits.Time
    channel_to_averaging_time: dict[str, tunits.Time]
    channel_to_averaging_window: dict[str, ValueArrayLike]


class ParameterSweepContent(Model):
    category: Literal["frequency_shift", "sequence_variable"]
    sweep_target: list[str]
    value_list: ValueArrayLike


class ParameterSweepConfig(Model):
    sweep_content_list: dict[str, ParameterSweepContent]
    sweep_axis: list[list[str]]


class SweepMeasurementConfig(Model):
    channel_list: list[str]
    sequence: ParametricSequenceConfig
    frequency: FrequencyConfig
    data_acquisition: DataAcquisitionConfig
    sweep_parameter: ParameterSweepConfig
