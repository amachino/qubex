"""Tests for the shared shot-interval default contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from qubex.contrib.experiment._measurement_defaults import (
    resolve_shot_interval as resolve_contrib_shot_interval,
)
from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.services.measurement_service import MeasurementService
from qubex.experiment.services.pulse_service import PulseService
from qubex.measurement.measurement_defaults import (
    DEFAULT_SHOT_INTERVAL,
    resolve_shot_interval_ns,
)
from qubex.measurement.models import MeasurementConfig
from qubex.system.measurement_defaults import MeasurementDefaults


def test_resolver_prefers_runtime_override_then_config_then_fallback() -> None:
    """Resolve runtime overrides ahead of YAML values and common fallback."""
    configured = {
        "execution": {
            "shot_interval_ns": 500000,
        }
    }

    assert resolve_shot_interval_ns(configured, None) == 500000.0
    assert resolve_shot_interval_ns(configured, 125000) == 125000.0
    assert (
        resolve_shot_interval_ns(MeasurementDefaults(), None) == DEFAULT_SHOT_INTERVAL
    )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_resolver_rejects_negative_or_nonfinite_values(value: float) -> None:
    """Reject invalid runtime overrides before a measurement is constructed."""
    with pytest.raises(ValueError, match="shot_interval"):
        resolve_shot_interval_ns(None, value)


def test_resolver_rejects_boolean_values() -> None:
    """Do not silently interpret a boolean as a numeric interval."""
    with pytest.raises(TypeError, match="shot_interval"):
        resolve_shot_interval_ns(None, True)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_yaml_model_uses_the_shared_shot_interval_contract(value: object) -> None:
    """Validate YAML execution defaults with the runtime value contract."""
    if value == 0:
        defaults = MeasurementDefaults.model_validate(
            {"execution": {"shot_interval_ns": value}}
        )
        assert defaults.execution.shot_interval_ns == 0.0
        return

    with pytest.raises((TypeError, ValidationError), match="shot_interval_ns"):
        MeasurementDefaults.model_validate({"execution": {"shot_interval_ns": value}})


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_measurement_config_uses_the_shared_shot_interval_contract(
    value: object,
) -> None:
    """Keep direct MeasurementConfig construction aligned with YAML validation."""
    options = {
        "n_shots": 1,
        "shot_interval": value,
        "shot_averaging": True,
        "time_integration": True,
        "state_classification": False,
    }
    if value == 0:
        assert MeasurementConfig(**options).shot_interval == 0.0
        return

    with pytest.raises((TypeError, ValidationError), match="shot_interval"):
        MeasurementConfig(**options)


def test_measurement_service_resolves_configured_interval_after_alias_handling() -> (
    None
):
    """Regular MeasurementService workflows use the configured interval."""
    service = MeasurementService(
        experiment_context=cast(
            ExperimentContext,
            SimpleNamespace(
                experiment_system=SimpleNamespace(
                    measurement_defaults={"execution": {"shot_interval_ns": 500000}},
                )
            ),
        ),
        pulse_service=cast(PulseService, SimpleNamespace()),
    )

    _, interval, _ = service.resolve_shot_options(
        n_shots=None,
        shot_interval=None,
        deprecated_options={},
    )

    assert interval == 500000.0


def test_contrib_resolver_uses_the_same_configured_default() -> None:
    """Contrib APIs resolve their omitted interval from the experiment context."""
    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            experiment_system=SimpleNamespace(
                measurement_defaults={"execution": {"shot_interval_ns": 500000}}
            )
        )
    )

    assert resolve_contrib_shot_interval(exp, None) == 500000.0
    assert resolve_contrib_shot_interval(exp, 125000) == 125000.0


def test_measurement_service_preserves_deprecated_interval_precedence() -> None:
    """The legacy interval alias still resolves before the configured default."""
    service = MeasurementService(
        experiment_context=cast(
            ExperimentContext,
            SimpleNamespace(
                experiment_system=SimpleNamespace(
                    measurement_defaults={"execution": {"shot_interval_ns": 500000}},
                )
            ),
        ),
        pulse_service=cast(PulseService, SimpleNamespace()),
    )

    with pytest.warns(DeprecationWarning, match="`interval` is deprecated"):
        _, interval, _ = service.resolve_shot_options(
            n_shots=None,
            shot_interval=None,
            deprecated_options={"interval": 125000},
        )

    assert interval == 125000.0
