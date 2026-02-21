"""Configuration models for sweep measurements."""

from __future__ import annotations

from typing import Literal

from qxcore import Model, Time, ValueArrayLike

from .data_acquisition_config import DataAcquisitionConfig
from .frequency_config import FrequencyConfig


class ParametricSequencePulseCommand(Model):
    """Pulse command used in parametric sequences."""

    name: str
    channel_list: list[str]
    argument_list: list[str | float]


class ParametricSequenceConfig(Model):
    """Sequence configuration with variable parameters."""

    delta_time: Time
    variable_list: list[str]
    command_list: list[ParametricSequencePulseCommand]


class ParameterSweepContent(Model):
    """Definition of a sweep dimension."""

    category: Literal["frequency_shift", "sequence_variable"]
    sweep_target: list[str]
    value_list: ValueArrayLike


class ParameterSweepConfig(Model):
    """Collection of sweep contents and axes."""

    sweep_content_list: dict[str, ParameterSweepContent]
    sweep_axis: list[list[str]]


class SweepMeasurementConfig(Model):
    """Top-level configuration for sweep measurements."""

    channel_list: list[str]
    sequence: ParametricSequenceConfig
    frequency: FrequencyConfig
    data_acquisition: DataAcquisitionConfig
    sweep_parameter: ParameterSweepConfig
