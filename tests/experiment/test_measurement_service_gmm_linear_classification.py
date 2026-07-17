"""Tests for experimental gmm_linear DSP classification plumbing."""

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

    classifiers = {
        "Q01": SimpleNamespace(
            centers={0: 0.0 + 0.0j, 1: 2.0 + 0.0j},
            stddevs={0: 0.25, 1: 0.5},
        ),
        "Q00": SimpleNamespace(
            centers={0: 1.0 + 1.0j, 1: 1.0 + 3.0j},
            stddevs={0: 0.25, 1: 0.5},
        ),
    }
    ctx = SimpleNamespace(
        state_centers={},
        state_stddevs={},
        classifiers=classifiers,
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
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=1024.0,
        readout_pre_margin=0.0,
        readout_post_margin=0.0,
    )
    return cast(MeasurementService, service), {
        "execute_calls": execute_calls,
        "measure_calls": measure_calls,
    }


def test_execute_resolves_gmm_linear_line_pairs_per_readout_target() -> None:
    """Given multi-qubit GMM classifiers, execute should derive ordered DSP lines per readout target."""
    service, captured = _make_service()

    with PulseSchedule(["Q01", "Q00"]) as schedule:
        pass

    service.execute(schedule, classification_source="gmm_linear", plot=False)
    kwargs = captured["execute_calls"][0]
    line_param0 = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param0"],
    )
    line_param1 = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param1"],
    )

    assert kwargs["classification_source"] == "gmm_linear"
    assert kwargs["state_classification"] is True
    assert set(line_param0) == {"RQ00", "RQ01"}
    assert set(line_param1) == {"RQ00", "RQ01"}
    assert line_param0["RQ01"][:2] == pytest.approx(line_param1["RQ01"][:2])
    assert _evaluate_line(line_param0["RQ01"], 1.5 + 0.0j) == pytest.approx(0.0)
    assert _evaluate_line(line_param1["RQ01"], 0.25 + 0.0j) == pytest.approx(0.0)


def test_execute_resolves_gmm_linear_line_pairs_with_sigma_multiplier() -> None:
    """Given a sigma multiplier, execute should scale generated DSP line offsets."""
    service, captured = _make_service()

    with PulseSchedule(["Q01"]) as schedule:
        pass

    service.execute(
        schedule,
        classification_source="gmm_linear",
        classification_sigma_multiplier=2.0,
        plot=False,
    )
    kwargs = captured["execute_calls"][0]
    line_param0 = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param0"],
    )
    line_param1 = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param1"],
    )

    assert _evaluate_line(line_param0["RQ01"], 1.0 + 0.0j) == pytest.approx(0.0)
    assert _evaluate_line(line_param1["RQ01"], 0.5 + 0.0j) == pytest.approx(0.0)


def test_measure_resolves_gmm_linear_line_pairs_from_waveform_targets() -> None:
    """Given waveform measurement input, measure should derive per-target DSP line pairs too."""
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
    line_param0 = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param0"],
    )
    line_param1 = cast(
        dict[str, tuple[float, float, float]],
        kwargs["classification_line_param1"],
    )
    assert kwargs["classification_source"] == "gmm_linear"
    assert kwargs["state_classification"] is True
    assert set(line_param0) == {"RQ00", "RQ01"}
    assert set(line_param1) == {"RQ00", "RQ01"}


def test_execute_rejects_removed_manual_line_overrides() -> None:
    """Given removed single-line DSP args, execute should reject them early."""
    service, _ = _make_service()

    with PulseSchedule(["Q00"]) as schedule:
        pass

    with pytest.raises(TypeError, match="Unexpected keyword"):
        service.execute(
            schedule,
            classification_source="gmm_linear",
            line_param0=(1.0, 0.0, 0.0),
            plot=False,
        )
