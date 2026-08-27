"""Tests for Rabi-parameter acquisition behavior."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

from qubex.experiment.models.rabi_param import RabiParam
from qubex.experiment.services.measurement_service import MeasurementService


def test_obtain_rabi_params_scopes_frequencies_to_each_sequential_target() -> None:
    """Given multiple frequencies, sequential Rabi runs should receive only their target override."""
    targets = ["Q040", "Q041", "Q042", "Q043"]
    frequencies = {target: 5.0 + index * 0.1 for index, target in enumerate(targets)}
    amplitudes = dict.fromkeys(targets, 0.1)
    received_frequencies: list[dict[str, float] | None] = []
    service = cast(Any, object.__new__(MeasurementService))

    def _rabi_experiment(
        self: MeasurementService,
        *,
        amplitudes: dict[str, float],
        frequencies: dict[str, float] | None,
        **_kwargs: object,
    ) -> Any:
        _ = self
        target = next(iter(amplitudes))
        received_frequencies.append(frequencies)
        return SimpleNamespace(
            data={
                target: SimpleNamespace(rabi_param=RabiParam.nan(target=target)),
            }
        )

    service.rabi_experiment = MethodType(_rabi_experiment, service)

    result = service.obtain_rabi_params(
        targets=targets,
        amplitudes=amplitudes,
        frequencies=frequencies,
        plot=False,
        store_params=False,
    )

    assert received_frequencies == [{target: frequencies[target]} for target in targets]
    assert list(result.data) == targets
    assert list(result.rabi_params or {}) == targets
