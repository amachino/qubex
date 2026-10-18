"""Measurement-default resolution helpers for contrib experiment APIs."""

from __future__ import annotations

from typing import Any

from qubex.measurement.measurement_defaults import resolve_shot_interval_ns


def resolve_shot_interval(exp: Any, shot_interval: object | None) -> float:
    """Resolve a contrib override through configured defaults and the common fallback."""
    if shot_interval is not None:
        return resolve_shot_interval_ns(None, shot_interval)
    context = getattr(exp, "ctx", None)
    experiment_system = getattr(context, "experiment_system", None)
    measurement_defaults = getattr(experiment_system, "measurement_defaults", None)
    return resolve_shot_interval_ns(measurement_defaults, None)
