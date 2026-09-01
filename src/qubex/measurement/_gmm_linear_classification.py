"""Internal helpers for GMM-derived DSP classification lines."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeVar

LineParam = tuple[float, float, float]
LineParamPair = tuple[LineParam, LineParam]

_StateValueT = TypeVar("_StateValueT", complex, float)


def _resolve_state_value(
    values: Mapping[int | str, _StateValueT],
    state: int | str,
) -> _StateValueT:
    if state in values:
        return values[state]
    text_state = str(state)
    if text_state in values:
        return values[text_state]
    raise ValueError(f"State value {state!r} is not available.")


def _resolve_state_center(
    centers: Mapping[int | str, complex],
    state: int | str,
) -> complex:
    return complex(_resolve_state_value(centers, state))


def _resolve_state_stddev(
    stddevs: Mapping[int | str, float],
    state: int | str,
) -> float:
    stddev = float(_resolve_state_value(stddevs, state))
    if not math.isfinite(stddev) or stddev < 0.0:
        raise ValueError(
            "GMM state standard deviations must be finite and non-negative."
        )
    return stddev


def _line_from_normal_and_point(
    normal: tuple[float, float],
    point: complex,
) -> LineParam:
    a, b = normal
    c = -(a * point.real + b * point.imag)
    return (a, b, c)


def _line_projection(line: LineParam) -> float:
    """Return the signed point projection for a normalized line normal."""
    return -line[2]


def _normalized_ge_axis(
    *,
    g_center: complex,
    e_center: complex,
) -> tuple[float, float]:
    if g_center == e_center:
        raise ValueError("GMM state centers for g/e must not be identical.")
    delta = g_center - e_center
    norm = math.hypot(delta.real, delta.imag)
    if norm == 0:
        raise ValueError("GMM state centers must define a non-degenerate line.")
    return (delta.real / norm, delta.imag / norm)


def build_gmm_linear_line_param_pair(
    centers: Mapping[int | str, complex],
    stddevs: Mapping[int | str, float],
    *,
    sigma_multiplier: float = 1.0,
    g_state: int = 0,
    e_state: int = 1,
) -> LineParamPair:
    """
    Return two parallel DSP classification lines from GMM g/e centers.

    The normal vector points from the excited-state center toward the
    ground-state center.  The returned order is stable: line 0 is the
    lower-projection/e-side boundary, and line 1 is the higher-projection/g-side
    boundary.
    """
    sigma_multiplier = float(sigma_multiplier)
    if not math.isfinite(sigma_multiplier) or sigma_multiplier < 0.0:
        raise ValueError("sigma_multiplier must be finite and non-negative.")

    g_center = _resolve_state_center(centers, g_state)
    e_center = _resolve_state_center(centers, e_state)
    g_stddev = _resolve_state_stddev(stddevs, g_state)
    e_stddev = _resolve_state_stddev(stddevs, e_state)
    normal = _normalized_ge_axis(g_center=g_center, e_center=e_center)
    normal_complex = complex(normal[0], normal[1])

    e_line_point = e_center + normal_complex * sigma_multiplier * e_stddev
    g_line_point = g_center - normal_complex * sigma_multiplier * g_stddev
    e_line = _line_from_normal_and_point(normal, e_line_point)
    g_line = _line_from_normal_and_point(normal, g_line_point)
    line0, line1 = sorted((e_line, g_line), key=_line_projection)
    return (line0, line1)
