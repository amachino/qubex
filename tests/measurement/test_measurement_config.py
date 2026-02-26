"""Tests for measurement configuration model."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.measurement_defaults import (
    DEFAULT_INTERVAL,
    DEFAULT_N_SHOTS,
    DEFAULT_SHOT_INTERVAL_NS,
    DEFAULT_SHOTS,
)
from qubex.measurement.models import MeasurementConfig
from qubex.system import ExperimentSystem


def test_model_requires_all_fields() -> None:
    """Given missing fields, when creating config directly, then validation fails."""
    with pytest.raises(ValidationError):
        MeasurementConfig.model_validate(
            {
                "n_shots": 1,
                "shot_interval_ns": 100.0,
                "shot_averaging": True,
                "time_integration": False,
            }
        )


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

    assert config.n_shots == DEFAULT_N_SHOTS
    assert config.shot_interval_ns == DEFAULT_SHOT_INTERVAL_NS
    assert config.shot_averaging is True
    assert config.time_integration is False
    assert config.state_classification is False


def test_legacy_default_aliases_match_renamed_constants() -> None:
    """Given legacy aliases, when imported, then they match renamed defaults."""
    assert DEFAULT_SHOTS == DEFAULT_N_SHOTS
    assert DEFAULT_INTERVAL == DEFAULT_SHOT_INTERVAL_NS


def test_factory_maps_boolean_overrides() -> None:
    """Given boolean overrides, when factory builds config, then values are set in config."""
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
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
    )

    assert config.shot_averaging is False
    assert config.time_integration is True
    assert config.state_classification is True


def test_factory_rejects_frequency_overrides() -> None:
    """Given frequency overrides, when factory builds config, then TypeError is raised."""
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

    with pytest.raises(TypeError):
        factory.create(frequencies={"Q00": 5.0, "Q01": 5.2})  # type: ignore[call-arg]
