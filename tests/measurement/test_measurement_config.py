"""Tests for measurement configuration model."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from qubex.backend import ExperimentSystem
from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.measurement_defaults import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.measurement.models import MeasurementConfig


def test_model_requires_all_fields() -> None:
    """Given missing fields, when creating config directly, then validation fails."""
    with pytest.raises(ValidationError):
        MeasurementConfig.model_validate({"mode": "avg", "shots": 1, "interval": 100.0})


def test_factory_applies_context_defaults() -> None:
    """Given omitted fields, when factory builds config, then context-aware defaults are applied."""
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type(
                "_CP", (), {"readout_amplitude": {"RQ00": 0.25, "RQ01": 0.3}}
            )(),
            "measurement_defaults": {},
        },
    )()
    factory = MeasurementConfigFactory(
        experiment_system=cast(ExperimentSystem, experiment_system)
    )
    config = factory.create()

    assert config.mode == "avg"
    assert config.shots == DEFAULT_SHOTS
    assert config.interval == DEFAULT_INTERVAL
    assert config.frequency.frequencies == {}
    assert config.dsp.enable_dsp_demodulation is True


def test_factory_maps_dsp_and_line_params() -> None:
    """Given dsp args, when factory builds config, then DSP and line params are set in dsp config."""
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type("_CP", (), {"readout_amplitude": {}})(),
            "measurement_defaults": {},
        },
    )()
    factory = MeasurementConfigFactory(
        experiment_system=cast(ExperimentSystem, experiment_system)
    )
    config = factory.create(
        enable_dsp_demodulation=False,
        enable_dsp_sum=True,
        enable_dsp_classification=True,
        line_param0=(1.0, 2.0, 3.0),
        line_param1=(4.0, 5.0, 6.0),
    )

    assert config.dsp.enable_dsp_demodulation is False
    assert config.dsp.enable_dsp_sum is True
    assert config.dsp.enable_dsp_classification is True
    assert config.dsp.line_param0 == (1.0, 2.0, 3.0)
    assert config.dsp.line_param1 == (4.0, 5.0, 6.0)


def test_factory_maps_frequency_overrides() -> None:
    """Given frequency overrides, when factory builds config, then frequency map is set."""
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type("_CP", (), {"readout_amplitude": {}})(),
            "measurement_defaults": {},
        },
    )()
    factory = MeasurementConfigFactory(
        experiment_system=cast(ExperimentSystem, experiment_system)
    )
    config = factory.create(frequencies={"Q00": 5.0, "Q01": 5.2})

    assert config.frequency.frequencies == {"Q00": 5.0, "Q01": 5.2}
