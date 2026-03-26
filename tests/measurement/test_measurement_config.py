"""Tests for measurement configuration model."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.measurement_defaults import (
    DEFAULT_INTERVAL,
    DEFAULT_N_SHOTS,
    DEFAULT_SHOT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.measurement.models import MeasurementConfig, ReturnItem
from qubex.system import ExperimentSystem


def test_model_requires_all_fields() -> None:
    """Given missing fields, when creating config directly, then validation fails."""
    with pytest.raises(ValidationError):
        MeasurementConfig.model_validate(
            {
                "n_shots": 1,
                "shot_interval": 100.0,
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
    assert config.shot_interval == DEFAULT_SHOT_INTERVAL
    assert config.shot_averaging is True
    assert config.time_integration is True
    assert config.state_classification is False


def test_factory_applies_measurement_defaults_overrides() -> None:
    """Given measurement defaults overrides, when factory builds config, then execution defaults use them."""
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type("_CP", (), {"readout_amplitude": {}})(),
            "measurement_defaults": {
                "execution": {
                    "n_shots": 2048,
                    "shot_interval_ns": 200000.0,
                }
            },
        },
    )()
    factory = MeasurementConfigFactory(
        experiment_system=cast(ExperimentSystem, experiment_system)
    )

    config = factory.create()

    assert config.n_shots == 2048
    assert config.shot_interval == 200000.0
    assert config.shot_averaging is True
    assert config.time_integration is True
    assert config.state_classification is False


def test_legacy_default_aliases_match_renamed_constants() -> None:
    """Given legacy aliases, when imported, then they match renamed defaults."""
    assert DEFAULT_SHOTS == DEFAULT_N_SHOTS
    assert DEFAULT_INTERVAL == DEFAULT_SHOT_INTERVAL


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


def test_model_populates_return_items_from_flags() -> None:
    """Given legacy booleans, model should infer return items."""
    config = MeasurementConfig(
        n_shots=4,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
    )

    assert tuple(config.return_items) == (
        ReturnItem.IQ_SERIES,
        ReturnItem.STATE_SERIES,
    )


def test_model_rejects_return_items_conflicting_with_flags() -> None:
    """Given conflicting return items, model validation should fail."""
    with pytest.raises(ValidationError):
        _ = MeasurementConfig(
            n_shots=4,
            shot_interval=100.0,
            shot_averaging=True,
            time_integration=False,
            state_classification=False,
            return_items=(ReturnItem.IQ_SERIES,),
        )


def test_model_rejects_duplicate_return_items() -> None:
    """Given duplicate return items, model validation should fail."""
    with pytest.raises(ValidationError):
        _ = MeasurementConfig(
            n_shots=4,
            shot_interval=100.0,
            shot_averaging=False,
            time_integration=False,
            state_classification=False,
            return_items=(ReturnItem.WAVEFORM_SERIES, ReturnItem.WAVEFORM_SERIES),
        )


def test_model_gmm_linear_classification_forces_state_series_return_item() -> None:
    """Given gmm_linear classification, model should use state-series payloads only."""
    config = MeasurementConfig(
        n_shots=4,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
        classification_source="gmm_linear",
    )

    assert config.primary_return_item == ReturnItem.STATE_SERIES
    assert tuple(config.return_items) == (ReturnItem.STATE_SERIES,)


def test_model_rejects_invalid_gmm_linear_flag_combinations() -> None:
    """Given invalid flags, gmm_linear classification config validation should fail."""
    with pytest.raises(ValidationError, match="requires shot_averaging=False"):
        _ = MeasurementConfig(
            n_shots=4,
            shot_interval=100.0,
            shot_averaging=True,
            time_integration=True,
            state_classification=True,
            classification_source="gmm_linear",
        )

    with pytest.raises(ValidationError, match="requires time_integration=True"):
        _ = MeasurementConfig(
            n_shots=4,
            shot_interval=100.0,
            shot_averaging=False,
            time_integration=False,
            state_classification=True,
            classification_source="gmm_linear",
        )


def test_factory_forwards_classification_source() -> None:
    """Given classification_source, factory should persist it on the built config."""
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
        classification_source="gmm_linear",
    )

    assert config.classification_source == "gmm_linear"
