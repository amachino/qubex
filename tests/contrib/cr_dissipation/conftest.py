"""Shared construction helpers for lightweight CR dissipation v11 tests."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from qubex.contrib.experiment._cr_dissipation_pulses import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    PROTOCOL_D,
)
from qubex.contrib.experiment._cr_dissipation_types import (
    CrDissipationMeasurements,
    CrDissipationProtocolData,
    CrDissipationProtocolMeasurements,
    GefPopulationSeries,
)


def population_series(
    values: NDArray[np.float64], error: float = 2e-4
) -> GefPopulationSeries:
    """Build a deterministic GEF series with finite conditional covariance."""
    values = np.asarray(values, dtype=np.float64)
    covariance = np.tile(np.eye(3) * error**2, (values.shape[0], 1, 1))
    return GefPopulationSeries(
        values,
        covariance,
        np.full(values.shape, error),
        values.copy(),
    )


def measurements_from_protocol_data(
    counts: NDArray[np.int64],
    data: Mapping[str, CrDissipationProtocolData],
    *,
    block_duration_ns: float = 640.0,
    cr_active_duration_ns: float = 400.0,
) -> CrDissipationMeasurements:
    """Assemble the fixed-schema four-protocol measurement object."""
    protocols: dict[str, CrDissipationProtocolMeasurements] = {}
    for name in (PROTOCOL_A, PROTOCOL_B, PROTOCOL_C, PROTOCOL_D):
        protocols[name] = CrDissipationProtocolMeasurements(
            counts.astype(float) * block_duration_ns,
            counts.astype(float) * cr_active_duration_ns,
            data[name],
            None,
            None,
        )
    counts_i64 = counts.astype(np.int64)
    return CrDissipationMeasurements(
        counts_i64,
        np.int64(4) * counts_i64,
        protocols[PROTOCOL_A],
        protocols[PROTOCOL_B],
        protocols[PROTOCOL_C],
        protocols[PROTOCOL_D],
    )
