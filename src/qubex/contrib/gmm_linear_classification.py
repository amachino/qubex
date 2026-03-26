"""Experimental helpers for DSP classification from GMM state centers."""

from __future__ import annotations

import math
from collections.abc import Mapping

LineParam = tuple[float, float, float]


def _resolve_state_center(
    centers: Mapping[int | str, complex],
    state: int,
) -> complex:
    if state in centers:
        return complex(centers[state])
    text_state = str(state)
    if text_state in centers:
        return complex(centers[text_state])
    raise ValueError(f"State center {state!r} is not available.")


def build_gmm_linear_line_param(
    centers: Mapping[int | str, complex],
    *,
    g_state: int = 0,
    e_state: int = 1,
) -> LineParam:
    """Return the normalized linear separator derived from GMM state centers."""
    g_center = _resolve_state_center(centers, g_state)
    e_center = _resolve_state_center(centers, e_state)
    if g_center == e_center:
        raise ValueError("GMM state centers for g/e must not be identical.")

    delta = e_center - g_center
    midpoint = (g_center + e_center) / 2
    a = float(delta.real)
    b = float(delta.imag)
    c = float(-(a * midpoint.real + b * midpoint.imag))
    norm = math.hypot(a, b)
    if norm == 0:
        raise ValueError("GMM state centers must define a non-degenerate line.")
    return (a / norm, b / norm, c / norm)


def build_gmm_linear_line_params(
    state_centers: Mapping[str, Mapping[int | str, complex]],
    *,
    g_state: int = 0,
    e_state: int = 1,
) -> dict[str, LineParam]:
    """Return per-target linear separators derived from GMM state centers."""
    return {
        target: build_gmm_linear_line_param(
            centers,
            g_state=g_state,
            e_state=e_state,
        )
        for target, centers in state_centers.items()
    }
