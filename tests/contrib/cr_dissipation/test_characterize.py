# ruff: noqa: SLF001
"""Public validation and fail-fast tests for CR dissipation v11."""

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import qubex.contrib.experiment._cr_dissipation.experiment as health
from qubex.contrib.experiment._cr_dissipation.pulses import ZX90Descriptor
from qubex.experiment import Experiment
from qubex.pulse import Blank, FlatTop, PulseSchedule


class _Context:
    def __init__(self) -> None:
        self.experiment_system = SimpleNamespace(
            measurement_defaults={"execution": {"shot_interval_ns": 500_000.0}}
        )

    @staticmethod
    def resolve_qubit_label(label: str) -> str:
        return label


def test_optional_diagnostics_default_on_and_fidelity_uncertainty_off() -> None:
    """Public defaults acquire diagnostics but avoid costly fidelity SEs."""
    defaults = health.characterize_cr_dissipation.__kwdefaults__
    assert defaults is not None
    assert defaults["measure_reference"] is True
    assert defaults["measure_orthogonal_components"] is True
    assert defaults["compute_fidelity_uncertainty"] is False


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


def test_default_repetition_counts_span_twice_the_longer_idle_t1() -> None:
    """The ten-point default grid is geometric from one to the 2T1 endpoint."""
    counts = health._default_repetition_counts(
        longest_idle_t1_ns=50_000.0,
        ab_block_duration_ns=1_000.0,
    )

    assert counts == (0, 1, 2, 3, 6, 10, 18, 32, 56, 100)
    assert len(counts) == 10
    assert all(right > left for left, right in pairwise(counts))


@pytest.mark.parametrize(
    ("t1_ns", "block_ns", "match"),
    [
        (float("inf"), 1_000.0, "finite positive idle T1"),
        (4_000.0, 1_000.0, "fewer than nine .* blocks"),
    ],
)
def test_default_repetition_counts_require_a_representable_endpoint(
    t1_ns: float,
    block_ns: float,
    match: str,
) -> None:
    """An undefined ten-point default asks the caller for an explicit grid."""
    with pytest.raises(ValueError, match=match):
        health._default_repetition_counts(
            longest_idle_t1_ns=t1_ns,
            ab_block_duration_ns=block_ns,
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


@pytest.mark.parametrize(
    "name",
    ["n_shots", "gef_measurement_n_shots", "gef_calibration_n_shots"],
)
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


def test_explicit_idle_value_errors_are_not_masked_by_other_stored_inputs() -> None:
    """An invalid explicit T1 should retain its precise validation error."""
    loader = SimpleNamespace(
        load_param_data=lambda name: {"Q0": 20_000.0, "Q1": 20_000.0}
    )
    exp = cast(
        Experiment,
        SimpleNamespace(
            ctx=SimpleNamespace(
                system_manager=SimpleNamespace(config_loader=loader),
            )
        ),
    )

    with pytest.raises(ValueError, match="t1_ns must be positive"):
        health._load_idle_noise(
            exp,
            "Q0",
            "Q1",
            {"Q0": -1.0, "Q1": 30_000.0},
            None,
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


def test_malformed_reference_cache_falls_back_to_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed cached scalar fields should trigger recalibration instead of failing."""
    source = FlatTop(duration=64.0, amplitude=0.23, tau=7.0, beta=0.37)
    cached = {
        "amplitude": "invalid",
        "duration": 64.0,
        "ramptime": 7.0,
        "beta": 0.37,
        "ramp_type": str(source.type),
        "sampling_period": source.sampling_period,
        "r_squared": "invalid",
        "timestamp": "old",
    }
    stored: list[dict[str, Any]] = []
    note = SimpleNamespace(
        get_property=lambda *_: cached,
        put_property=lambda _note, _key, value: stored.append(value),
    )
    exp = SimpleNamespace(ctx=SimpleNamespace(calib_note=note))
    descriptor = SimpleNamespace(echoed=SimpleNamespace(cr_waveform=source))
    replacement = health._ReferenceIx45Calibration(
        0.14,
        64.0,
        7.0,
        0.37,
        str(source.type),
        source.sampling_period,
        0.98,
        "new",
    )
    monkeypatch.setattr(
        health,
        "_calibrate_reference_ix45",
        lambda *_args, **_kwargs: replacement,
    )

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
        enable_tqdm=False,
    )

    assert was_calibrated
    assert calibration is replacement
    assert pulse.amplitude == pytest.approx(replacement.amplitude)
    assert stored
    assert stored[0]["amplitude"] == pytest.approx(replacement.amplitude)


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


@pytest.mark.parametrize("r_squared", [float("nan"), float("inf"), float("-inf"), 0.49])
def test_reference_calibration_rejects_nonfinite_or_low_fit_quality(
    monkeypatch: pytest.MonkeyPatch,
    r_squared: float,
) -> None:
    """IX45 calibration accepts only finite fit quality above threshold."""
    cr_envelope = FlatTop(duration=64.0, amplitude=0.23, tau=7.0)
    hpi = FlatTop(duration=32.0, amplitude=0.2, tau=4.0)
    exp = SimpleNamespace(
        pulse=SimpleNamespace(get_hpi_pulse=lambda _target: hpi),
        measurement_service=SimpleNamespace(
            sweep_parameter=lambda **_kwargs: SimpleNamespace(
                data={"Q1": SimpleNamespace(normalized=np.linspace(0.0, 1.0, 21))}
            )
        ),
    )
    monkeypatch.setattr(
        health.fitting,
        "fit_ampl_calib_data",
        lambda **_kwargs: {"amplitude": 0.12, "r2": r_squared},
    )

    with pytest.raises(RuntimeError, match="quality validation"):
        health._calibrate_reference_ix45(
            cast(Experiment, exp),
            "Q1",
            cr_envelope,
            n_shots=100,
            shot_interval=1234.0,
            plot=False,
            enable_tqdm=False,
        )


def test_primary_gef_acquisition_is_split_by_protocol_and_shot_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A/B calibrate jointly while C/D reuse calibration without bootstrap."""
    protocols = (
        health.PROTOCOL_A,
        health.PROTOCOL_B,
        health.PROTOCOL_C,
        health.PROTOCOL_D,
    )
    schedules = SimpleNamespace(
        actual={protocol: (PulseSchedule(),) for protocol in protocols},
        timing={},
    )
    descriptor = SimpleNamespace(
        warnings=(),
        total_duration_ns=100.0,
        cr_active_duration_ns=40.0,
        cr_lobe_duration_ns=20.0,
        echo_slot_duration_ns=30.0,
        full_un_echoed=SimpleNamespace(duration=100.0),
        rotary_integrated_angle_rad=None,
    )
    calibration = {"Q0": object(), "Q1": object()}
    calls: list[dict[str, Any]] = []
    bootstrap_calls: list[dict[str, Any]] = []
    no_echo_override = PulseSchedule()
    resolved_pulses: list[tuple[Any, Any]] = []

    def measure_gef(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return health.Result(
            data={
                "calibration": calibration,
                "raw_iq": {},
                "fits": {},
                "moment_summaries": {},
                "bootstrap": {},
            }
        )

    monkeypatch.setattr(
        health,
        "_load_idle_noise",
        lambda *_args: (
            health.IdleNoiseParameters(4_000.0, 3_000.0),
            health.IdleNoiseParameters(5_000.0, 3_000.0),
        ),
    )

    def resolve_descriptor(
        _exp: Any,
        _control: str,
        _target: str,
        zx90_echo: Any,
        zx90_no_echo: Any,
    ) -> Any:
        resolved_pulses.append((zx90_echo, zx90_no_echo))
        return descriptor

    monkeypatch.setattr(health, "resolve_zx90_descriptor", resolve_descriptor)
    monkeypatch.setattr(health, "build_protocol_schedules", lambda *_args: schedules)
    monkeypatch.setattr(health, "measure_gef_populations", measure_gef)

    def run_selective_bootstrap(
        *args: Any, **kwargs: Any
    ) -> dict[str, tuple[str, ...]]:
        bootstrap_calls.append(kwargs)
        names_by_protocol = cast(dict[str, tuple[str, ...]], args[1])
        return {"Q1": names_by_protocol[health.PROTOCOL_A]}

    monkeypatch.setattr(health, "_run_selective_ab_bootstrap", run_selective_bootstrap)
    monkeypatch.setattr(
        health, "_build_measurements", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        health, "analyze_cr_dissipation", lambda *_args, **_kwargs: object()
    )
    shown: list[str] = []
    figures = {
        protocol: SimpleNamespace(show=lambda name=protocol: shown.append(name))
        for protocol in protocols
    }
    monkeypatch.setattr(health, "_plot_cr_dissipation", lambda *_args: figures)
    monkeypatch.setattr(health, "_print_summary", lambda *_args: None)

    result = health.characterize_cr_dissipation(
        _fake_experiment(),
        "Q0",
        "Q1",
        n_shots=222,
        gef_measurement_n_shots=444,
        gef_calibration_n_shots=888,
        gef_bootstrap_n_resamples=12,
        zx90_no_echo=no_echo_override,
        idle_t1={"Q0": 1.0, "Q1": 1.0},
        idle_t2_echo={"Q0": 1.0, "Q1": 1.0},
        measure_reference=False,
        measure_orthogonal_components=False,
        plot=True,
    )

    assert len(calls) == 3
    assert calls[0]["targets"] == ["Q0", "Q1"]
    assert calls[0]["n_shots"] == 444
    assert calls[0]["calibration_n_shots"] == 888
    assert calls[0]["n_bootstrap"] == 0
    assert calls[0]["progress_label"] == (
        f"{health.PROTOCOL_A} / {health.PROTOCOL_B} primary"
    )
    assert calls[1]["targets"] == ["Q0"]
    assert calls[1]["calibration"] == {"Q0": calibration["Q0"]}
    assert calls[1]["n_shots"] == 444
    assert calls[1]["n_bootstrap"] == 0
    assert calls[2]["targets"] == ["Q1"]
    assert calls[2]["calibration"] == {"Q1": calibration["Q1"]}
    assert calls[2]["n_shots"] == 444
    assert calls[2]["n_bootstrap"] == 0
    assert calls[1]["progress_label"].startswith(health.PROTOCOL_C)
    assert calls[2]["progress_label"].startswith(health.PROTOCOL_D)
    assert shown == list(protocols)
    assert resolved_pulses == [(None, no_echo_override)]
    assert result.data["pulse_timing"]["zx90_full_un_echoed"]["source"] == "override"
    assert result.data["measurement_options"]["n_shots"] == 222
    assert result.data["measurement_options"]["gef_measurement_n_shots"] == 444
    assert result.data["measurement_options"]["repetition_counts"] == (
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        11,
        17,
        25,
    )
    assert result.data["measurement_options"]["primary_bootstrap"] == {
        health.PROTOCOL_A: {"Q0": 0, "Q1": 12},
        health.PROTOCOL_B: {"Q0": 0, "Q1": 0},
        health.PROTOCOL_C: {"Q0": 0},
        health.PROTOCOL_D: {"Q1": 0},
    }
    assert result.data["measurement_options"]["orthogonal_observables"] == {}
    assert result.data["measurement_options"]["orthogonal_readout"] is None
    assert result.data["measurement_options"]["reference_readout"] is None
    assert result.data["analysis_options"]["c_d_fit_method"] == (
        "two_qutrit_physical_forward_variable_projection"
    )
    assert result.data["analysis_options"]["c_d_nonlinear_parameter_count"] == 1
    assert result.data["analysis_options"]["fidelity_uncertainty_method"] == "disabled"
    assert bootstrap_calls == [
        {
            "n_resamples": 12,
            "seed": 0,
            "confidence_level": 0.95,
            "covariance_rcond": 1e-12,
        }
    ]


def test_boundary_bootstrap_reuses_only_selected_ab_raw_iq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary diagnostics resample only the selected target's A/B sequences."""
    names = {
        health.PROTOCOL_A: ("a0", "a1"),
        health.PROTOCOL_B: ("b0", "b1"),
    }
    calibration = {"Q0": object(), "Q1": object()}
    raw_iq = {
        name: {
            configuration: {"Q0": np.ones(2), "Q1": np.ones(2)}
            for configuration in ("s1", "s4", "s5")
        }
        for name in (*names[health.PROTOCOL_A], *names[health.PROTOCOL_B])
    }
    bootstrap = {
        name: {"Q0": "disabled-control", "Q1": "disabled-target"} for name in raw_iq
    }
    result = health.Result(
        data={
            "calibration": calibration,
            "raw_iq": raw_iq,
            "bootstrap": bootstrap,
        }
    )
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    monkeypatch.setattr(
        health,
        "_select_boundary_bootstrap_sequence_names",
        lambda *_args: {"Q1": ("a0", "a1")},
    )

    def bootstrap_selected(
        selected_calibration: dict[str, Any],
        selected_raw_iq: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, dict[str, str]]:
        calls.append((selected_calibration, selected_raw_iq))
        return {name: {"Q1": f"bootstrap-{name}"} for name in selected_raw_iq}

    monkeypatch.setattr(health, "bootstrap_gef_populations", bootstrap_selected)

    selected = health._run_selective_ab_bootstrap(
        result,
        names,
        "Q0",
        "Q1",
        n_resamples=12,
        seed=0,
        confidence_level=0.95,
        covariance_rcond=1e-12,
    )

    assert selected == {"Q1": ("a0", "a1")}
    assert calls == [
        (
            {"Q1": calibration["Q1"]},
            {"a0": raw_iq["a0"], "a1": raw_iq["a1"]},
        )
    ]
    assert bootstrap["a0"]["Q1"] == "bootstrap-a0"
    assert bootstrap["a1"]["Q1"] == "bootstrap-a1"
    assert bootstrap["b0"]["Q1"] == "disabled-target"


def test_boundary_bootstrap_selects_only_protocols_with_flat_boundary_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the boundary-ambiguous A/B target series are selected for bootstrap."""
    names = {
        health.PROTOCOL_A: ("a0", "a1"),
        health.PROTOCOL_B: ("b0", "b1"),
    }
    flat_boundary = health.GefPopulationSeries(
        population=np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]),
        covariance=np.tile(np.eye(3), (2, 1, 1)),
        standard_error=np.full((2, 3), 0.01),
        population_unconstrained=np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.001]]),
    )
    resolved_boundary = health.GefPopulationSeries(
        population=np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]),
        covariance=np.tile(np.eye(3), (2, 1, 1)),
        standard_error=np.full((2, 3), 0.01),
        population_unconstrained=np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.004]]),
    )
    data = {
        health.PROTOCOL_A: health.CrDissipationProtocolData(target_gef=flat_boundary),
        health.PROTOCOL_B: health.CrDissipationProtocolData(
            target_gef=resolved_boundary
        ),
    }
    monkeypatch.setattr(
        health,
        "_protocol_data",
        lambda _result, _names, _control, _target, protocol: data[protocol],
    )

    selected = health._select_boundary_bootstrap_sequence_names(
        health.Result(data={}),
        names,
        "Q0",
        "Q1",
    )

    assert selected == {"Q1": ("a0", "a1")}


def test_all_orthogonal_diagnostics_use_pauli_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A/B/C/D orthogonal data use one Pauli acquisition."""
    base = {
        protocol: (PulseSchedule(), PulseSchedule())
        for protocol in (
            health.PROTOCOL_A,
            health.PROTOCOL_B,
            health.PROTOCOL_C,
            health.PROTOCOL_D,
        )
    }
    schedules = SimpleNamespace(base=base)
    pauli_calls: list[dict[str, Any]] = []

    def measure_pauli(*_args: Any, **kwargs: Any) -> Any:
        pauli_calls.append(kwargs)
        values = np.linspace(-1.2, -0.8, kwargs["n_shots"]).astype(complex)
        return [{"Q0": values, "Q1": values} for _ in _args[1]]

    pulse = SimpleNamespace(
        x90=lambda _qubit: Blank(4.0),
        y90m=lambda _qubit: Blank(4.0),
        rabi_params={
            "Q0": SimpleNamespace(normalize=lambda values: np.real(values)),
            "Q1": SimpleNamespace(normalize=lambda values: np.real(values)),
        },
    )
    exp = cast(Experiment, SimpleNamespace(pulse=pulse))
    monkeypatch.setattr(health, "measure_single_shot_batch", measure_pauli)

    processed, pauli_raw = health._orthogonal_acquisition(
        exp,
        cast(health.ProtocolSchedules, schedules),
        "Q0",
        "Q1",
        n_shots=2048,
        shot_interval=1000.0,
        enable_tqdm=False,
    )

    assert pauli_calls[0]["n_shots"] == 2048
    assert set(processed) == set(base)
    assert len(pauli_raw) == 24
    assert set(processed[health.PROTOCOL_A].expectations) == {
        "Q0:X",
        "Q0:Y",
        "Q1:Y",
        "Q1:Z",
    }
    assert set(processed[health.PROTOCOL_B].expectations) == {
        "Q0:X",
        "Q0:Y",
        "Q1:Y",
        "Q1:Z",
    }
    assert processed[health.PROTOCOL_A].component_roles == {
        "Q0:X": "control",
        "Q0:Y": "control",
        "Q1:Y": "target",
        "Q1:Z": "target",
    }
    assert set(processed[health.PROTOCOL_C].component_roles.values()) == {"control"}
    assert set(processed[health.PROTOCOL_D].component_roles.values()) == {"target"}
    assert set(processed[health.PROTOCOL_D].expectations) == {"Q1:X", "Q1:Y"}
    assert np.all(processed[health.PROTOCOL_A].expectations["Q0:X"] < 0.0)
    assert processed[health.PROTOCOL_C].standard_errors
    assert processed[health.PROTOCOL_D].standard_errors


@pytest.mark.parametrize(
    "normalized",
    [np.ones(1), np.array([0.0, 0.0, float("nan"), 0.0])],
)
def test_orthogonal_acquisition_rejects_invalid_normalized_shots(
    monkeypatch: pytest.MonkeyPatch,
    normalized: np.ndarray[Any, Any],
) -> None:
    """Pauli normalization must return one finite value per acquired shot."""
    base = {
        protocol: (PulseSchedule(),)
        for protocol in (
            health.PROTOCOL_A,
            health.PROTOCOL_B,
            health.PROTOCOL_C,
            health.PROTOCOL_D,
        )
    }
    schedules = SimpleNamespace(base=base)
    pulse = SimpleNamespace(
        x90=lambda _qubit: Blank(4.0),
        y90m=lambda _qubit: Blank(4.0),
        rabi_params={
            "Q0": SimpleNamespace(normalize=lambda _values: normalized),
            "Q1": SimpleNamespace(normalize=lambda _values: normalized),
        },
    )
    exp = cast(Experiment, SimpleNamespace(pulse=pulse))
    monkeypatch.setattr(
        health,
        "measure_single_shot_batch",
        lambda *_args, **_kwargs: [
            {"Q0": np.ones(4), "Q1": np.ones(4)} for _ in range(12)
        ],
    )

    with pytest.raises(ValueError, match="normalization must return 4 finite shots"):
        health._orthogonal_acquisition(
            exp,
            cast(health.ProtocolSchedules, schedules),
            "Q0",
            "Q1",
            n_shots=4,
            shot_interval=1000.0,
            enable_tqdm=False,
        )


def test_plots_share_axes_state_colors_and_smooth_model_curves() -> None:
    """Figures use common ranges and preserve state colors across trace roles."""
    counts = np.arange(5, dtype=np.int64)
    population = np.column_stack((1.0 - 0.1 * counts, 0.08 * counts, 0.02 * counts))
    covariance = np.tile(np.eye(3)[None, :, :] * 1e-4, (counts.size, 1, 1))
    series = health.GefPopulationSeries(
        population,
        covariance,
        np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)),
        population,
    )
    ab_data = health.CrDissipationProtocolData(
        control_gef=series,
        target_gef=series,
        target_x_comp=np.linspace(1.0, 0.4, counts.size),
        target_x_comp_standard_error=np.full(counts.size, 0.02),
    )
    cd_data = health.CrDissipationProtocolData(
        primary_expectation=np.linspace(1.0, 0.3, counts.size),
        primary_standard_error=np.full(counts.size, 0.02),
    )

    orthogonal = health.CrDissipationPauliData(
        expectations={
            "Q0:Y": np.linspace(0.0, 0.4, counts.size),
            "Q1:Z": np.linspace(0.2, 0.6, counts.size),
        },
        standard_errors={
            "Q0:Y": np.full(counts.size, 0.02),
            "Q1:Z": np.full(counts.size, 0.02),
        },
        component_roles={"Q0:Y": "control", "Q1:Z": "target"},
    )

    def measured(
        data: Any,
        duration: float,
        *,
        with_orthogonal: bool = False,
        with_reference: bool = False,
    ) -> Any:
        elapsed = counts.astype(float) * duration
        return health.CrDissipationProtocolMeasurements(
            elapsed,
            elapsed * 0.5,
            data,
            data if with_reference else None,
            orthogonal if with_orthogonal else None,
        )

    measurements = health.CrDissipationMeasurements(
        counts,
        4 * counts,
        measured(ab_data, 100.0, with_orthogonal=True),
        measured(ab_data, 100.0),
        measured(cd_data, 200.0, with_reference=True),
        measured(cd_data, 300.0, with_reference=True),
    )
    fitted_decay = SimpleNamespace(fitted_values=np.linspace(1.0, 0.45, counts.size))
    fitted_leakage = SimpleNamespace(fitted_values=np.linspace(0.0, 0.08, counts.size))
    fitted_cd = SimpleNamespace(fitted_values=np.linspace(1.0, 0.35, counts.size))
    fits = SimpleNamespace(
        control_population_ab=SimpleNamespace(
            fitted_populations={
                health.PROTOCOL_A: population,
                health.PROTOCOL_B: population,
            }
        ),
        target_a_t1rho=fitted_decay,
        target_b_t1rho=fitted_decay,
        target_a_leakage=fitted_leakage,
        target_b_leakage=fitted_leakage,
        control_pure_dephasing_c=fitted_cd,
        target_rotating_frame_pure_dephasing_d=fitted_cd,
    )
    idle_predictions = {
        health.PROTOCOL_A: SimpleNamespace(
            observables={
                "control_population": population,
                "target_x_comp": np.linspace(1.0, 0.5, counts.size),
                "target_pf": np.linspace(0.0, 0.05, counts.size),
            }
        ),
        health.PROTOCOL_B: SimpleNamespace(
            observables={
                "control_population": population,
                "target_x_comp": np.linspace(1.0, 0.5, counts.size),
                "target_pf": np.linspace(0.0, 0.05, counts.size),
            }
        ),
        health.PROTOCOL_C: SimpleNamespace(
            observables={
                "primary_expectation": np.linspace(1.0, 0.4, counts.size),
                "reference_primary_expectation": np.linspace(1.0, 0.55, counts.size),
            }
        ),
        health.PROTOCOL_D: SimpleNamespace(
            observables={
                "primary_expectation": np.linspace(1.0, 0.4, counts.size),
                "reference_primary_expectation": np.linspace(1.0, 0.65, counts.size),
            }
        ),
    }
    analysis = cast(
        health.CrDissipationAnalysis,
        SimpleNamespace(fits=fits, idle_predictions=idle_predictions),
    )

    figures = health._plot_cr_dissipation(measurements, analysis)

    a_traces: dict[str, Any] = {}
    for trace in figures[health.PROTOCOL_A].data:
        typed_trace = cast(Any, trace)
        a_traces[typed_trace.name] = typed_trace
    assert len(a_traces["fit Pg"].x) == 400
    assert a_traces["actual Pg"].marker.color == a_traces["fit Pg"].line.color
    assert a_traces["fit Pg"].line.color == a_traces["idle-only Pg"].line.color
    assert a_traces["Q0:Y"].yaxis == "y2"
    assert a_traces["actual Xcomp"].yaxis == "y3"
    assert a_traces["actual target Pf"].yaxis == "y4"
    assert a_traces["Q1:Z"].yaxis == "y5"
    a_layout = cast(Any, figures[health.PROTOCOL_A].layout)
    assert tuple(a_layout.yaxis.range) == (0.0, 1.0)
    assert tuple(a_layout.yaxis2.range) == (-1.0, 1.0)
    assert tuple(a_layout.yaxis3.range) == (-1.0, 1.0)
    assert tuple(a_layout.yaxis4.range) == (0.0, 0.08)
    assert tuple(a_layout.yaxis5.range) == (-1.0, 1.0)
    expected_x_range = (0.0, 1.2)
    for figure in figures.values():
        layout = cast(Any, figure.layout)
        assert tuple(layout.xaxis.range) == pytest.approx(expected_x_range)
        assert layout.legend.x == pytest.approx(1.02)

    for panel in range(1, 6):
        suffix = "" if panel == 1 else str(panel)
        bottom_axis = getattr(a_layout, f"xaxis{suffix}")
        top_axis = getattr(a_layout, f"xaxis{6 + panel}")
        assert bottom_axis.showticklabels
        assert bottom_axis.title.text == "Elapsed sequence time [us]"
        assert top_axis.side == "top"
        assert top_axis.overlaying == f"x{suffix}"
        assert top_axis.title.text == "CR gates (4n)"
        bottom_ticks = np.asarray(bottom_axis.tickvals, dtype=float)
        top_ticks = np.asarray(top_axis.tickvals, dtype=float)
        assert np.allclose(np.diff(bottom_ticks), np.diff(bottom_ticks)[0])
        assert np.allclose(np.diff(top_ticks), np.diff(top_ticks)[0])
        assert tuple(bottom_ticks) == pytest.approx((0.0, 0.25, 0.5, 0.75, 1.0))
        assert np.all(np.mod(top_ticks, 4.0) == 0.0)
        assert tuple(top_axis.ticktext) == tuple(str(int(value)) for value in top_ticks)
        protocol_duration_us = float(
            np.max(getattr(measurements, health.PROTOCOL_A).elapsed_time_ns) / 1000.0
        )
        expected_gate_limit = int(4 * counts[-1]) * (
            expected_x_range[1] / protocol_duration_us
        )
        assert tuple(top_axis.range) == pytest.approx((0, expected_gate_limit))
        y_axis = getattr(a_layout, f"yaxis{suffix}")
        assert y_axis.title.text
    assert a_layout.width == 1500
    assert a_layout.xaxis.domain[1] < a_layout.xaxis2.domain[0]
    assert a_layout.xaxis6.visible is False
    assert a_layout.yaxis6.visible is False
    assert all(annotation.yshift == 72 for annotation in a_layout.annotations[:5])
    assert len(a_layout.shapes) == 5
    c_figure = figures[health.PROTOCOL_C]
    c_layout = cast(Any, c_figure.layout)
    c_traces = tuple(cast(Any, trace) for trace in c_figure.data)
    c_trace_names = {trace.name for trace in c_traces}
    assert "reference idle-only prediction" in c_trace_names
    reference_idle = next(
        trace for trace in c_traces if trace.name == "reference idle-only prediction"
    )
    assert reference_idle.line.dash == "dot"
    assert reference_idle.line.width == pytest.approx(1.25)
    assert not any(
        str(annotation.text).startswith("Reference is diagnostic")
        for annotation in c_layout.annotations
    )
    assert c_layout.margin.b == 90
    d_traces = tuple(cast(Any, trace) for trace in figures[health.PROTOCOL_D].data)
    assert "reference idle-only prediction" in {trace.name for trace in d_traces}
    d_layout = cast(Any, figures[health.PROTOCOL_D].layout)
    d_titles = {str(annotation.text) for annotation in d_layout.annotations}
    assert "Target Zge" in d_titles


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([np.array([0.01, 0.02])], (0.0, 0.05)),
        ([np.array([-0.01, 0.08])], (-0.01, 0.08)),
    ],
)
def test_pf_axis_range_is_zero_anchored_with_minimum_span(
    values: list[np.ndarray[Any, Any]], expected: tuple[float, float]
) -> None:
    """Pf plots include zero and expand to a span of at least 0.05."""
    assert tuple(health._pf_axis_range(values)) == pytest.approx(expected)
