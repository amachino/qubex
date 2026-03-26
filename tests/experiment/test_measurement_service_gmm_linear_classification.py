"""Tests for experimental gmm_linear DSP classification plumbing in MeasurementService."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import PulseSchedule

from qubex.experiment.services.measurement_service import MeasurementService


class _DummyResult:
    def plot(self) -> None:
        return None


def _evaluate_line(line: tuple[float, float, float], point: complex) -> float:
    a, b, c = line
    return a * point.real + b * point.imag + c


def _make_service() -> tuple[MeasurementService, dict[str, list[dict[str, object]]]]:
    execute_calls: list[dict[str, object]] = []
    measure_calls: list[dict[str, object]] = []

    @contextmanager
    def _modified_frequencies(_: object) -> Any:
        yield

    ctx = SimpleNamespace(
        state_centers={
            "Q01": {0: 0.0 + 0.0j, 1: 2.0 + 0.0j},
            "Q00": {0: 1.0 + 1.0j, 1: 1.0 + 3.0j},
        },
        ordered_qubit_labels=lambda labels: [
            qubit
            for qubit in ("Q01", "Q00")
            if qubit in labels or f"R{qubit}" in labels
        ],
        resolve_read_label=lambda qubit: f"R{qubit}",
        resolve_qubit_label=lambda label: label.replace("R", ""),
        modified_frequencies=_modified_frequencies,
        measurement=SimpleNamespace(
            execute=lambda **kwargs: execute_calls.append(kwargs) or _DummyResult(),
            measure=lambda **kwargs: measure_calls.append(kwargs) or _DummyResult(),
        ),
        reset_awg_and_capunits=lambda *, qubits: None,
        qubit_labels=["Q00", "Q01"],
    )
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = ctx
    service.__dict__["_pulse_service"] = SimpleNamespace()
    return cast(MeasurementService, service), {
        "execute_calls": execute_calls,
        "measure_calls": measure_calls,
    }


def test_execute_resolves_multi_qubit_gmm_linear_line_params_per_readout_target() -> None:
    """Given multi-qubit centers, execute should derive one identical DSP line per readout target."""
    service, captured = _make_service()

    with PulseSchedule(["Q01", "Q00"]) as schedule:
        pass

    service.execute(schedule, classification_source="gmm_linear", plot=False)
    kwargs = captured["execute_calls"][0]
    line_param0_by_target = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param0_by_target"],
    )
    line_param1_by_target = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param1_by_target"],
    )

    assert kwargs["classification_source"] == "gmm_linear"
    assert kwargs["state_classification"] is True
    assert line_param0_by_target == line_param1_by_target
    assert set(line_param0_by_target) == {"RQ00", "RQ01"}
    assert _evaluate_line(line_param0_by_target["RQ01"], 0.0 + 0.0j) < 0.0
    assert _evaluate_line(line_param0_by_target["RQ01"], 2.0 + 0.0j) > 0.0
    assert _evaluate_line(line_param0_by_target["RQ00"], 1.0 + 1.0j) < 0.0
    assert _evaluate_line(line_param0_by_target["RQ00"], 1.0 + 3.0j) > 0.0


def test_measure_resolves_gmm_linear_line_params_from_waveform_targets() -> None:
    """Given waveform measurement input, measure should derive per-target DSP line params too."""
    service, captured = _make_service()

    service.measure(
        {
            "Q00": np.array([0.0 + 0.0j], dtype=np.complex128),
            "Q01": np.array([0.0 + 0.0j], dtype=np.complex128),
        },
        classification_source="gmm_linear",
        plot=False,
    )

    kwargs = captured["measure_calls"][0]
    line_param0_by_target = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param0_by_target"],
    )
    assert kwargs["classification_source"] == "gmm_linear"
    assert kwargs["state_classification"] is True
    assert set(line_param0_by_target) == {"RQ00", "RQ01"}


def test_execute_rejects_manual_line_overrides_for_gmm_linear() -> None:
    """Given gmm_linear source, manual DSP line overrides should be rejected."""
    service, _ = _make_service()

    with PulseSchedule(["Q00"]) as schedule:
        pass

    with pytest.raises(ValueError, match="does not accept manual line_param overrides"):
        service.execute(
            schedule,
            classification_source="gmm_linear",
            classification_line_param0=(1.0, 0.0, 0.0),
            plot=False,
        )
