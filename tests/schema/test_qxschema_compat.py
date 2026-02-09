"""Tests for qxschema sweep measurement model exports."""

from __future__ import annotations

import numpy as np

from qubex.core import Frequency, Time


def test_qxschema_exports_sweep_measurement_config_models() -> None:
    """Given qxschema package, when importing config models, then all expected models are available."""
    # Arrange
    from qxschema import (
        DataAcquisitionConfig,
        FrequencyConfig,
        ParameterSweepConfig,
        ParameterSweepContent,
        ParametricSequenceConfig,
        ParametricSequencePulseCommand,
        SweepMeasurementConfig,
    )

    # Act
    config = SweepMeasurementConfig(
        channel_list=["Q00"],
        sequence=ParametricSequenceConfig(
            delta_time=Time(2, "ns"),
            variable_list=["amp"],
            command_list=[
                ParametricSequencePulseCommand(
                    name="Rect",
                    channel_list=["Q00"],
                    argument_list=["amp", 10.0],
                )
            ],
        ),
        frequency=FrequencyConfig(
            channel_to_frequency={"Q00": Frequency(5.0, "GHz")},
            channel_to_frequency_reference={"Q00": "f_ge"},
            channel_to_frequency_shift={"Q00": Frequency(1.0, "MHz")},
            keep_oscillator_relative_phase=False,
        ),
        data_acquisition=DataAcquisitionConfig(
            shot_count=128,
            shot_repetition_margin=Time(100, "us"),
            data_acquisition_duration=Time(500, "ns"),
            data_acquisition_delay=Time(16, "ns"),
            data_acquisition_timeout=Time(10, "ms"),
            flag_average_waveform=False,
            flag_average_shots=True,
            delta_time=Time(2, "ns"),
            channel_to_averaging_time={"Q00": Time(32, "ns")},
            channel_to_averaging_window={"Q00": [0.0, 1.0]},
        ),
        sweep_parameter=ParameterSweepConfig(
            sweep_content_list={
                "amp": ParameterSweepContent(
                    category="sequence_variable",
                    sweep_target=["amp"],
                    value_list=[0.1, 0.2, 0.3],
                )
            },
            sweep_axis=[["amp"]],
        ),
    )

    # Assert
    assert config.sequence.command_list[0].name == "Rect"


def test_qxschema_exports_sweep_measurement_result_model() -> None:
    """Given qxschema package, when importing result model, then ndarray payload is accepted."""
    # Arrange
    from qxschema import SweepMeasurementResult

    # Act
    result = SweepMeasurementResult(
        metadata={"mode": "single"},
        data=np.array([1.0, 2.0]),
        data_shape=[2],
        sweep_key_list=["amp"],
        data_key_list=["Q00"],
    )

    # Assert
    assert result.data_shape == [2]
