"""Tests for sweep-parameter Rabi parameter lookup scope."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
from qxpulse import Blank, PulseSchedule

from qubex.experiment.models.rabi_param import RabiParam
from qubex.experiment.services.measurement_service import MeasurementService
from qubex.experiment.services.pulse_service import PulseService


def test_sweep_parameter_reads_rabi_params_only_for_swept_qubits() -> None:
    """Given unrelated stored params, when sweeping one qubit, then only that qubit is queried."""
    accessed_targets: list[str] = []
    rabi_param = RabiParam.nan(target="Q33")

    @contextmanager
    def _modified_frequencies(_frequencies: dict[str, float] | None) -> Any:
        yield

    def _get_rabi_param(target: str) -> RabiParam | None:
        accessed_targets.append(target)
        return rabi_param if target == "Q33" else RabiParam.nan(target=target)

    ctx = SimpleNamespace(
        targets={
            "Q33": SimpleNamespace(is_ge=True, is_ef=False),
            "Q18": SimpleNamespace(is_ge=True, is_ef=False),
        },
        state_centers={},
        ordered_qubit_labels=lambda labels: list(labels),
        reset_awg_and_capunits=lambda *, qubits: None,
        modified_frequencies=_modified_frequencies,
        get_rabi_param=_get_rabi_param,
        resolve_ef_label=lambda label: f"{label}/ef",
    )
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = ctx
    service.__dict__["_pulse_service"] = PulseService(cast(Any, ctx))

    def _measure(self: MeasurementService, _seq: object, **_kwargs: object) -> Any:
        return SimpleNamespace(data={"Q33": SimpleNamespace(kerneled=1.0 + 0.0j)})

    service.measure = MethodType(_measure, service)

    def _sequence(_sweep_value: float) -> PulseSchedule:
        with PulseSchedule(["Q33"]) as schedule:
            schedule.add("Q33", Blank(8))
        return schedule

    result = service.sweep_parameter(
        sequence=_sequence,
        sweep_range=np.array([0.0]),
        plot=False,
    )

    assert accessed_targets == ["Q33"]
    assert result.rabi_params == {"Q33": rabi_param}
    assert result.data["Q33"].rabi_param is rabi_param
