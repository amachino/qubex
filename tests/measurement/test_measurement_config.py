"""Tests for measurement configuration model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

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
from qubex.measurement.services.measurement_execution_service import (
    MeasurementExecutionService,
)
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


def test_factory_maps_packed_timeline_limit() -> None:
    """Given packed timeline limit, when factory builds config, then config keeps it."""
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
        schedule_packing_enabled=True,
        max_repeated_timeline_duration_ns=20_000_000_000,
    )

    assert config.schedule_packing_enabled is True
    assert config.max_repeated_timeline_duration_ns == 20_000_000_000


def test_execution_service_resolves_schedule_packing_config() -> None:
    """Given measurement config, service should resolve schedule packing options."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    cast(Any, service)._context = SimpleNamespace(  # noqa: SLF001
        config_loader=SimpleNamespace(
            measurement_config={
                "schedule_packing": {
                    "enabled": True,
                    "max_repeated_timeline_duration_ns": 20_000_000_000,
                }
            }
        )
    )

    assert service._resolve_schedule_packing_enabled() is True  # noqa: SLF001
    assert service._resolve_max_repeated_timeline_duration_ns() == (  # noqa: SLF001
        20_000_000_000
    )


def test_execution_service_rejects_boolean_schedule_packing_config() -> None:
    """Given boolean schedule packing config, service should reject it."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    cast(Any, service)._context = SimpleNamespace(  # noqa: SLF001
        config_loader=SimpleNamespace(
            measurement_config={
                "schedule_packing": True,
            }
        )
    )

    with pytest.raises(TypeError, match="measurement\\.schedule_packing"):
        service._resolve_schedule_packing_enabled()  # noqa: SLF001


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
