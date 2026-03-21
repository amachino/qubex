"""Default values for measurement configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from qxpulse import RampType

from qubex.system.measurement_defaults import MeasurementDefaults

DEFAULT_N_SHOTS: Final[int] = 1024
DEFAULT_SHOT_INTERVAL: Final[float] = 150.0 * 1024.0  # ns

# Backward compatibility aliases.
DEFAULT_SHOTS: Final[int] = DEFAULT_N_SHOTS
DEFAULT_INTERVAL: Final[float] = DEFAULT_SHOT_INTERVAL

DEFAULT_READOUT_DURATION: Final[float] = 384.0  # ns
DEFAULT_READOUT_RAMP_TIME: Final[float] = 32.0  # ns
DEFAULT_READOUT_PRE_MARGIN: Final[float] = 32.0  # ns
DEFAULT_READOUT_POST_MARGIN: Final[float] = 128.0  # ns
DEFAULT_READOUT_DRAG_COEFF: Final[float] = 0.0
DEFAULT_READOUT_RAMP_TYPE: Final[RampType] = "RaisedCosine"

DEFAULT_SHOT_AVERAGING: Final[bool] = True
DEFAULT_TIME_INTEGRATION: Final[bool] = True
DEFAULT_STATE_CLASSIFICATION: Final[bool] = False


@dataclass(frozen=True)
class ResolvedMeasurementExecutionDefaults:
    """Execution defaults after config-file fallbacks are resolved."""

    n_shots: int
    shot_interval_ns: float


@dataclass(frozen=True)
class ResolvedReadoutDefaults:
    """Readout defaults after config-file fallbacks are resolved."""

    duration_ns: float
    ramp_time_ns: float
    pre_margin_ns: float
    post_margin_ns: float


@dataclass(frozen=True)
class ResolvedMeasurementDefaults:
    """Fully resolved measurement defaults used at runtime."""

    execution: ResolvedMeasurementExecutionDefaults
    readout: ResolvedReadoutDefaults


def _coerce_measurement_defaults(
    payload: Mapping[str, Any] | MeasurementDefaults | None,
) -> MeasurementDefaults:
    """Coerce raw or parsed measurement defaults into the partial config model."""
    if payload is None:
        return MeasurementDefaults()
    if isinstance(payload, MeasurementDefaults):
        return payload
    return MeasurementDefaults.model_validate(dict(payload))


def resolve_measurement_defaults(
    payload: Mapping[str, Any] | MeasurementDefaults | None = None,
) -> ResolvedMeasurementDefaults:
    """Resolve config-backed measurement defaults with hardcoded fallbacks."""
    partial = _coerce_measurement_defaults(payload)
    execution = partial.execution
    readout = partial.readout
    return ResolvedMeasurementDefaults(
        execution=ResolvedMeasurementExecutionDefaults(
            n_shots=(
                execution.n_shots if execution.n_shots is not None else DEFAULT_N_SHOTS
            ),
            shot_interval_ns=(
                execution.shot_interval_ns
                if execution.shot_interval_ns is not None
                else DEFAULT_SHOT_INTERVAL
            ),
        ),
        readout=ResolvedReadoutDefaults(
            duration_ns=(
                readout.duration_ns
                if readout.duration_ns is not None
                else DEFAULT_READOUT_DURATION
            ),
            ramp_time_ns=(
                readout.ramp_time_ns
                if readout.ramp_time_ns is not None
                else DEFAULT_READOUT_RAMP_TIME
            ),
            pre_margin_ns=(
                readout.pre_margin_ns
                if readout.pre_margin_ns is not None
                else DEFAULT_READOUT_PRE_MARGIN
            ),
            post_margin_ns=(
                readout.post_margin_ns
                if readout.post_margin_ns is not None
                else DEFAULT_READOUT_POST_MARGIN
            ),
        ),
    )


__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_N_SHOTS",
    "DEFAULT_READOUT_DRAG_COEFF",
    "DEFAULT_READOUT_DURATION",
    "DEFAULT_READOUT_POST_MARGIN",
    "DEFAULT_READOUT_PRE_MARGIN",
    "DEFAULT_READOUT_RAMP_TIME",
    "DEFAULT_READOUT_RAMP_TYPE",
    "DEFAULT_SHOTS",
    "DEFAULT_SHOT_AVERAGING",
    "DEFAULT_SHOT_INTERVAL",
    "DEFAULT_STATE_CLASSIFICATION",
    "DEFAULT_TIME_INTEGRATION",
    "ResolvedMeasurementDefaults",
    "ResolvedMeasurementExecutionDefaults",
    "ResolvedReadoutDefaults",
    "resolve_measurement_defaults",
]
