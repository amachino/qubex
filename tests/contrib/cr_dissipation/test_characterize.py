# ruff: noqa: SLF001
"""Public validation and fail-fast tests for CR dissipation v11."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import qubex.contrib.experiment.cr_dissipation as health
from qubex.contrib.experiment._cr_dissipation_pulses import ZX90Descriptor
from qubex.experiment import Experiment
from qubex.pulse import FlatTop


class _Context:
    def __init__(self) -> None:
        self.experiment_system = SimpleNamespace(
            measurement_defaults={"execution": {"shot_interval_ns": 500_000.0}}
        )

    @staticmethod
    def resolve_qubit_label(label: str) -> str:
        return label


class _Experiment:
    def __init__(self) -> None:
        self.ctx = _Context()


def _fake_experiment() -> Experiment:
    """Return the deliberately minimal fail-fast experiment stub."""
    return cast(Experiment, _Experiment())


@pytest.mark.parametrize(
    "counts",
    [
        (0, 1, 2, 3),
        (1, 2, 3, 4, 5),
        (0, 1, 1, 2, 3),
        (0, 2, 1, 3, 4),
        (0, 1, 2, 3, -4),
        (0, 1, 2, 3, 4.0),
    ],
)
def test_invalid_repetition_counts_are_rejected_before_hardware(counts: Any) -> None:
    """The public five-point, integer, ordered grid contract is fail-fast."""
    with pytest.raises((TypeError, ValueError)):
        health.characterize_cr_dissipation(
            _fake_experiment(),
            "Q0",
            "Q1",
            repetition_counts=counts,
            idle_t1={"Q0": 1.0, "Q1": 1.0},
            idle_t2_echo={"Q0": 1.0, "Q1": 1.0},
            plot=False,
        )


def test_ambiguous_reference_calibration_options_are_rejected() -> None:
    """A supplied amplitude and forced calibration cannot both be requested."""
    with pytest.raises(ValueError, match="cannot be specified together"):
        health.characterize_cr_dissipation(
            _fake_experiment(),
            "Q0",
            "Q1",
            reference_ix45_amplitude=0.1,
            force_reference_ix45_calibration=True,
            idle_t1={"Q0": 1.0, "Q1": 1.0},
            idle_t2_echo={"Q0": 1.0, "Q1": 1.0},
            plot=False,
        )


@pytest.mark.parametrize("name", ["n_shots", "gef_calibration_n_shots"])
def test_shot_counts_require_at_least_two(name: str) -> None:
    """Shot validation happens before pulse resolution or acquisition."""
    options: dict[str, Any] = {name: 1}
    with pytest.raises(ValueError, match=rf"{name} must be at least two"):
        health.characterize_cr_dissipation(
            _fake_experiment(),
            "Q0",
            "Q1",
            idle_t1={"Q0": 1.0, "Q1": 1.0},
            idle_t2_echo={"Q0": 1.0, "Q1": 1.0},
            plot=False,
            **options,
        )


def test_shot_counts_require_integers() -> None:
    """Boolean shot counts are rejected separately from the lower bound."""
    with pytest.raises(TypeError, match="gef_calibration_n_shots must be an integer"):
        health.characterize_cr_dissipation(
            _fake_experiment(),
            "Q0",
            "Q1",
            gef_calibration_n_shots=True,
            idle_t1={"Q0": 1.0, "Q1": 1.0},
            idle_t2_echo={"Q0": 1.0, "Q1": 1.0},
            plot=False,
        )


def test_missing_idle_coherence_fails_before_pulse_resolution() -> None:
    """Both qubits' T1 and T2_echo are mandatory fixed inputs."""
    with pytest.raises(ValueError, match="missing for qubit Q1"):
        health.characterize_cr_dissipation(
            _fake_experiment(),
            "Q0",
            "Q1",
            idle_t1={"Q0": 30_000.0},
            idle_t2_echo={"Q0": 20_000.0, "Q1": 20_000.0},
            plot=False,
        )


def test_reference_ix45_copies_complete_flat_top_geometry() -> None:
    """The reference pulse preserves duration, ramp, beta, type, and sampling."""
    source = FlatTop(
        duration=64.0,
        amplitude=0.23,
        tau=7.0,
        beta=0.37,
        type="RaisedCosine",
        sampling_period=0.5,
    )

    reference = health._make_ix45(source, 0.11)

    assert reference.duration == source.duration
    assert reference.tau == source.tau
    assert reference.beta == source.beta
    assert reference.type == source.type
    assert reference.sampling_period == source.sampling_period


def test_reference_cache_is_invalidated_when_beta_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache entry made for another DRAG beta must not be reused."""
    source = FlatTop(duration=64.0, amplitude=0.23, tau=7.0, beta=0.37)
    cached = {
        "amplitude": 0.10,
        "duration": 64.0,
        "ramptime": 7.0,
        "beta": 0.12,
        "ramp_type": str(source.type),
        "sampling_period": source.sampling_period,
        "r_squared": 0.99,
        "timestamp": "old",
    }
    note = SimpleNamespace(
        get_property=lambda *_: cached,
        put_property=lambda *_: None,
    )
    exp = SimpleNamespace(ctx=SimpleNamespace(calib_note=note))
    descriptor = SimpleNamespace(echoed=SimpleNamespace(cr_waveform=source))
    calls: list[dict[str, Any]] = []

    def calibrate(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return health._ReferenceIx45Calibration(
            0.14, 64.0, 7.0, 0.37, str(source.type), source.sampling_period, 0.98, "new"
        )

    monkeypatch.setattr(health, "_calibrate_reference_ix45", calibrate)

    pulse, calibration, was_calibrated = health._resolve_reference_ix45(
        cast(Experiment, exp),
        "Q0",
        "Q1",
        cast(ZX90Descriptor, descriptor),
        amplitude=None,
        force=False,
        n_shots=100,
        shot_interval=1000.0,
        plot=False,
        enable_tqdm=True,
    )

    assert calls
    assert calls[0]["enable_tqdm"] is True
    assert was_calibrated
    assert calibration.beta == pytest.approx(0.37)
    assert pulse.beta == pytest.approx(0.37)


def test_reference_calibration_uses_canonical_shot_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IX45 calibration must not emit deprecated shot-option warnings."""
    cr_envelope = FlatTop(duration=64.0, amplitude=0.23, tau=7.0, beta=0.37)
    hpi = FlatTop(duration=32.0, amplitude=0.2, tau=4.0, beta=0.1)
    calls: list[dict[str, Any]] = []

    def sweep_parameter(**kwargs: Any) -> Any:
        calls.append(kwargs)
        data = SimpleNamespace(normalized=np.linspace(0.0, 1.0, 21))
        return SimpleNamespace(data={"Q1": data})

    exp = SimpleNamespace(
        pulse=SimpleNamespace(get_hpi_pulse=lambda _target: hpi),
        measurement_service=SimpleNamespace(sweep_parameter=sweep_parameter),
    )
    monkeypatch.setattr(
        health.fitting,
        "fit_ampl_calib_data",
        lambda **_kwargs: {"amplitude": 0.12, "r2": 0.99},
    )

    health._calibrate_reference_ix45(
        cast(Experiment, exp),
        "Q1",
        cr_envelope,
        n_shots=100,
        shot_interval=1234.0,
        plot=False,
        enable_tqdm=True,
    )

    assert len(calls) == 1
    assert calls[0]["n_shots"] == 100
    assert calls[0]["shot_interval"] == 1234.0
    assert calls[0]["enable_tqdm"] is True
    assert "shots" not in calls[0]
    assert "interval" not in calls[0]
