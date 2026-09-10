"""Tests for GEF population measurement orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib.experiment.gef_population_estimation import (
    calibrate_gef_population,
    measure_gef_populations,
)
from qubex.pulse import Arbitrary, Blank, PulseSchedule


def _state_samples() -> dict[str, np.ndarray]:
    """Return deterministic state-specific samples with full moment variation."""
    rng = np.random.default_rng(1729)
    return {
        "g": rng.normal(-2.0, 0.25, 10) + 1j * rng.normal(0.2, 0.35, 10),
        "e": rng.normal(0.1, 0.55, 10) + 1j * rng.normal(1.8, 0.20, 10),
        "f": rng.normal(2.4, 0.30, 10) + 1j * rng.normal(-0.8, 0.60, 10),
    }


def _pure(samples: dict[str, np.ndarray], state: str) -> np.ndarray:
    """Repeat one empirical state distribution to one hundred shots."""
    return np.tile(samples[state], 10)


def _mixture(
    samples: dict[str, np.ndarray],
    weights: tuple[int, int, int],
) -> np.ndarray:
    """Build an exact empirical mixture in units of ten shots."""
    return np.concatenate(
        [
            np.tile(samples[state], count)
            for state, count in zip(("g", "e", "f"), weights, strict=True)
        ]
    )


class _DummyContext:
    """Resolve dummy qubits and their GE/EF control labels."""

    def resolve_qubit_label(self, target: str) -> str:
        """Resolve supported control labels to their qubit."""
        qubit = target.removesuffix("/ef")
        if qubit not in {"Q0", "Q1"}:
            raise ValueError(target)
        return qubit

    def resolve_ge_label(self, target: str) -> str:
        """Return the GE label."""
        return self.resolve_qubit_label(target)

    def resolve_ef_label(self, target: str) -> str:
        """Return the EF label."""
        return f"{self.resolve_qubit_label(target)}/ef"


class _DummyPulseService:
    """Provide analysis pulses with distinct durations."""

    readout_duration = 0.0
    readout_pre_margin = 0.0
    readout_post_margin = 0.0

    def x180(self, target: str) -> Arbitrary:
        """Return visible GE and EF pulse stand-ins."""
        durations = {
            "Q0": 4.0,
            "Q0/ef": 6.0,
            "Q1": 8.0,
            "Q1/ef": 10.0,
        }
        pulse_length = Blank(duration=durations[target]).length
        return Arbitrary(np.ones(pulse_length, dtype=np.complex128))


class _DummyMeasurementService:
    """Return queued IQ arrays and record measurement options."""

    def __init__(
        self,
        iq_queue: Sequence[np.ndarray | dict[str, np.ndarray]],
    ) -> None:
        self.iq_queue = list(iq_queue)
        self.calls: list[dict[str, object]] = []
        self.batch_calls: list[int] = []

    async def run_sweep_measurement(self, schedule, *, sweep_values, **kwargs):
        """Record one acquisition and return ordered single-shot captures."""
        self.batch_calls.append(len(sweep_values))
        results = []
        for value in sweep_values:
            result = self.measure(schedule(value), mode="single", **kwargs)
            results.append(
                SimpleNamespace(
                    data={
                        target: [
                            SimpleNamespace(
                                data=np.resize(data.kerneled, kwargs["n_shots"])
                            )
                        ]
                        for target, data in result.data.items()
                    }
                )
            )
        return SimpleNamespace(results=results)

    def measure(self, sequence: PulseSchedule, **kwargs: object) -> SimpleNamespace:
        """Return the next queued single-shot result."""
        self.calls.append({"sequence": sequence, **kwargs})
        queued = self.iq_queue.pop(0)
        iq_by_target = {"Q0": queued} if isinstance(queued, np.ndarray) else queued
        return SimpleNamespace(
            data={
                target: SimpleNamespace(kerneled=iq)
                for target, iq in iq_by_target.items()
            },
            config={"dummy": True},
        )


class _DummyExperiment:
    """Provide the experiment interfaces used by the population workflow."""

    def __init__(
        self,
        iq_queue: Sequence[np.ndarray | dict[str, np.ndarray]],
    ) -> None:
        self.ctx = _DummyContext()
        self.pulse = _DummyPulseService()
        self.measurement_service = _DummyMeasurementService(iq_queue)


def test_measure_gef_populations_calibrates_first_and_measures_each_permutation() -> (
    None
):
    """The workflow should run six calibrations then three analyses per input sequence."""
    samples = _state_samples()
    calibration_iq = [
        _pure(samples, "g"),
        _pure(samples, "g"),
        _pure(samples, "e"),
        _pure(samples, "f"),
        _pure(samples, "e"),
        _pure(samples, "f"),
    ]
    measurement_iq = [
        _mixture(samples, (2, 3, 5)),
        _mixture(samples, (3, 5, 2)),
        _mixture(samples, (5, 2, 3)),
    ]
    exp = _DummyExperiment([*calibration_iq, *measurement_iq])
    with PulseSchedule(["Q0"]) as preparation:
        preparation.add("Q0", Blank(duration=8.0))

    result = measure_gef_populations(
        exp,  # type: ignore[arg-type]
        targets="Q0",
        sequences={"prepared": preparation},
        n_shots=100,
        calibration_n_shots=100,
        shot_interval=1234.0,
        n_bootstrap=20,
        bootstrap_seed=11,
        bootstrap_confidence_level=0.90,
    )

    assert exp.measurement_service.batch_calls == [6, 3]
    assert len(exp.measurement_service.calls) == 9
    assert not exp.measurement_service.iq_queue
    assert (
        [
            call["sequence"].duration  # type: ignore[union-attr]
            for call in exp.measurement_service.calls[:6]
        ]
        == pytest.approx([14.0] * 6)
    )
    assert (
        [
            call["sequence"].duration  # type: ignore[union-attr]
            for call in exp.measurement_service.calls[6:]
        ]
        == pytest.approx([18.0] * 3)
    )
    c2_sequence = exp.measurement_service.calls[1]["sequence"]
    assert isinstance(c2_sequence, PulseSchedule)
    c2_ef_durations = [
        waveform.duration
        for waveform in c2_sequence.get_sequence("Q0/ef").flattened_elements
        if isinstance(waveform, (Arbitrary, Blank))
    ]
    assert c2_ef_durations == pytest.approx([8.0, 6.0])
    s1_sequence = exp.measurement_service.calls[6]["sequence"]
    assert isinstance(s1_sequence, PulseSchedule)
    s1_durations = [
        waveform.duration
        for waveform in s1_sequence.get_sequence("Q0").flattened_elements
        if isinstance(waveform, (Arbitrary, Blank))
    ]
    assert s1_durations == pytest.approx([8.0, 10.0])
    for call in exp.measurement_service.calls:
        assert call["mode"] == "single"
        assert call["time_integration"] is True
        assert call["state_classification"] is False
        assert call["shot_interval"] == 1234.0

    population = result.data["populations"]["prepared"]["Q0"]
    assert_allclose(population, [0.2, 0.3, 0.5], rtol=1e-6, atol=1e-7)
    assert result.data["sequence_names"] == ("prepared",)
    assert result.data["targets"] == ("Q0",)
    assert set(result.data["raw_iq"]["prepared"]) == {"s1", "s4", "s5"}
    bootstrap = result.data["bootstrap"]["prepared"]["Q0"]
    assert bootstrap.samples.shape == (20, 3)
    assert_allclose(bootstrap.point_estimate, population, rtol=1e-12, atol=1e-12)
    assert bootstrap.confidence_level == pytest.approx(0.90)
    assert bootstrap.seed == 11
    assert result.data["measurement_options"]["n_bootstrap"] == 20
    assert result.data["measurement_options"]["bootstrap_seed"] == 11


def test_measure_gef_populations_uses_gef_specific_shot_defaults() -> None:
    """Default acquisition should use 8192 calibration and 4096 measurement shots."""
    samples = _state_samples()
    calibration_iq = [_pure(samples, state) for state in ("g", "g", "e", "f", "e", "f")]
    measurement_iq = [
        _mixture(samples, weights) for weights in ((2, 3, 5), (3, 5, 2), (5, 2, 3))
    ]
    exp = _DummyExperiment([*calibration_iq, *measurement_iq])
    with PulseSchedule(["Q0"]) as preparation:
        preparation.add("Q0", Blank(duration=2.0))

    result = measure_gef_populations(
        exp,  # type: ignore[arg-type]
        targets="Q0",
        sequences=[preparation],
        n_bootstrap=0,
    )

    assert [call["n_shots"] for call in exp.measurement_service.calls[:6]] == [8192] * 6
    assert [call["n_shots"] for call in exp.measurement_service.calls[6:]] == [4096] * 3
    assert result.data["measurement_options"]["n_shots"] == 4096
    assert result.data["measurement_options"]["calibration_n_shots"] == 8192
    fit = result.data["fits"]["sequence_0"]["Q0"]
    assert fit.population_covariance.shape == (3, 3)
    assert fit.population_standard_error.shape == (3,)


def test_measure_gef_populations_reuses_supplied_calibration() -> None:
    """Supplying calibration should avoid repeating the six calibration measurements."""
    samples = _state_samples()
    calibration_exp = _DummyExperiment(
        [
            _pure(samples, "g"),
            _pure(samples, "g"),
            _pure(samples, "e"),
            _pure(samples, "f"),
            _pure(samples, "e"),
            _pure(samples, "f"),
        ]
    )
    calibration = calibrate_gef_population(
        calibration_exp,  # type: ignore[arg-type]
        targets="Q0",
        n_shots=100,
    )
    measurement_exp = _DummyExperiment(
        [
            _mixture(samples, (6, 1, 3)),
            _mixture(samples, (1, 3, 6)),
            _mixture(samples, (3, 6, 1)),
        ]
    )
    with PulseSchedule(["Q0"]) as preparation:
        preparation.add("Q0", Blank(duration=2.0))

    result = measure_gef_populations(
        measurement_exp,  # type: ignore[arg-type]
        targets="Q0",
        sequences=[preparation],
        calibration=calibration,
        n_shots=100,
        n_bootstrap=0,
    )

    assert measurement_exp.measurement_service.batch_calls == [3]
    assert len(measurement_exp.measurement_service.calls) == 3
    assert result.data["sequence_names"] == ("sequence_0",)
    assert_allclose(
        result.data["populations"]["sequence_0"]["Q0"],
        [0.6, 0.1, 0.3],
        rtol=1e-6,
        atol=1e-7,
    )


def test_measure_gef_populations_estimates_multiple_targets_in_the_same_shots() -> None:
    """Control and target marginal populations should share nine measurement calls."""
    q0_samples = _state_samples()
    q1_samples = {
        state: 0.7 * values - 0.4j for state, values in _state_samples().items()
    }
    calibration_states = ("g", "g", "e", "f", "e", "f")
    iq_queue: list[dict[str, np.ndarray]] = [
        {
            "Q0": _pure(q0_samples, state),
            "Q1": _pure(q1_samples, state),
        }
        for state in calibration_states
    ]
    q0_weights = ((2, 3, 5), (3, 5, 2), (5, 2, 3))
    q1_weights = ((6, 3, 1), (3, 1, 6), (1, 6, 3))
    iq_queue.extend(
        {
            "Q0": _mixture(q0_samples, q0_weight),
            "Q1": _mixture(q1_samples, q1_weight),
        }
        for q0_weight, q1_weight in zip(q0_weights, q1_weights, strict=True)
    )
    exp = _DummyExperiment(iq_queue)
    with PulseSchedule(["Q0", "Q1"]) as preparation:
        preparation.add("Q0", Blank(duration=2.0))
        preparation.add("Q1", Blank(duration=2.0))

    result = measure_gef_populations(
        exp,  # type: ignore[arg-type]
        targets=["Q0", "Q1"],
        sequences={"cr": preparation},
        n_shots=100,
        calibration_n_shots=100,
        n_bootstrap=0,
    )

    assert exp.measurement_service.batch_calls == [6, 3]
    assert len(exp.measurement_service.calls) == 9
    assert (
        [
            call["sequence"].duration  # type: ignore[union-attr]
            for call in exp.measurement_service.calls[:6]
        ]
        == pytest.approx([26.0] * 6)
    )
    c2_sequence = exp.measurement_service.calls[1]["sequence"]
    assert isinstance(c2_sequence, PulseSchedule)
    q0_ef_durations = [
        waveform.duration
        for waveform in c2_sequence.get_sequence("Q0/ef").flattened_elements
        if isinstance(waveform, (Arbitrary, Blank))
    ]
    q1_ef_durations = [
        waveform.duration
        for waveform in c2_sequence.get_sequence("Q1/ef").flattened_elements
        if isinstance(waveform, (Arbitrary, Blank))
    ]
    assert q0_ef_durations == pytest.approx([16.0, 10.0])
    assert q1_ef_durations == pytest.approx([16.0, 10.0])
    q0_ef_values = c2_sequence.get_sampled_sequence("Q0/ef")
    q1_ef_values = c2_sequence.get_sampled_sequence("Q1/ef")
    assert np.flatnonzero(q0_ef_values)[-1] == len(q0_ef_values) - 1
    assert np.flatnonzero(q1_ef_values)[-1] == len(q1_ef_values) - 1
    assert_allclose(
        result.data["populations"]["cr"]["Q0"],
        [0.2, 0.3, 0.5],
        rtol=1e-6,
        atol=1e-7,
    )
    assert_allclose(
        result.data["populations"]["cr"]["Q1"],
        [0.6, 0.3, 0.1],
        rtol=1e-6,
        atol=1e-7,
    )


def test_measure_gef_populations_rejects_empty_sequences() -> None:
    """At least one state-preparation sequence should be required."""
    exp = _DummyExperiment([])

    with pytest.raises(ValueError, match="at least one"):
        measure_gef_populations(
            exp,  # type: ignore[arg-type]
            targets="Q0",
            sequences=[],
        )


def test_measure_gef_populations_rejects_invalid_bootstrap_before_measurement() -> None:
    """Invalid bootstrap options should fail before hardware measurements."""
    exp = _DummyExperiment([])
    with PulseSchedule(["Q0"]) as preparation:
        preparation.add("Q0", Blank(duration=2.0))

    with pytest.raises(TypeError, match="bootstrap_seed"):
        measure_gef_populations(
            exp,  # type: ignore[arg-type]
            targets="Q0",
            sequences=[preparation],
            bootstrap_seed=True,
        )

    assert not exp.measurement_service.calls


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_shots": True}, "n_shots"),
        ({"calibration_n_shots": 2.5}, "calibration_n_shots"),
        ({"shot_interval": True}, "shot_interval"),
    ],
)
def test_measure_gef_populations_rejects_invalid_acquisition_options_first(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Invalid acquisition options should fail before hardware measurements."""
    exp = _DummyExperiment([])
    with PulseSchedule(["Q0"]) as preparation:
        preparation.add("Q0", Blank(duration=2.0))

    with pytest.raises(TypeError, match=message):
        measure_gef_populations(
            exp,  # type: ignore[arg-type]
            targets="Q0",
            sequences=[preparation],
            **kwargs,  # type: ignore[arg-type]
        )

    assert not exp.measurement_service.calls


def test_measure_gef_populations_validates_reused_bootstrap_calibration_first() -> None:
    """Missing calibration IQ should fail before unknown-state measurements."""
    samples = _state_samples()
    calibration_exp = _DummyExperiment(
        [
            _pure(samples, "g"),
            _pure(samples, "g"),
            _pure(samples, "e"),
            _pure(samples, "f"),
            _pure(samples, "e"),
            _pure(samples, "f"),
        ]
    )
    calibration = calibrate_gef_population(
        calibration_exp,  # type: ignore[arg-type]
        targets="Q0",
        n_shots=100,
    )
    del calibration["Q0"].raw_iq["c6"]
    measurement_exp = _DummyExperiment([])
    with PulseSchedule(["Q0"]) as preparation:
        preparation.add("Q0", Blank(duration=2.0))

    with pytest.raises(ValueError, match="c6"):
        measure_gef_populations(
            measurement_exp,  # type: ignore[arg-type]
            targets="Q0",
            sequences=[preparation],
            calibration=calibration,
            n_bootstrap=2,
        )

    assert not measurement_exp.measurement_service.calls


def test_measure_gef_populations_rejects_missing_batch_target() -> None:
    """Every batch configuration should return each requested target."""
    samples = _state_samples()
    exp = _DummyExperiment([{"Q1": _pure(samples, "g")} for _ in range(6)])
    preparation = PulseSchedule(["Q0"])

    with pytest.raises(ValueError, match="did not return `Q0`"):
        measure_gef_populations(
            exp,  # type: ignore[arg-type]
            targets="Q0",
            sequences=[preparation],
            calibration_n_shots=100,
            n_bootstrap=0,
        )

    assert exp.measurement_service.batch_calls == [6]
    assert len(exp.measurement_service.calls) == 6


def test_multiple_sequences_share_one_measurement_batch() -> None:
    """Named preparations should share one sweep without mixing configurations."""
    samples = _state_samples()
    weights = [(2, 3, 5), (6, 1, 3)]
    queue = [_pure(samples, state) for state in ("g", "g", "e", "f", "e", "f")]
    for g, e, f in weights:
        queue.extend(_mixture(samples, w) for w in ((g, e, f), (e, f, g), (f, g, e)))
    exp = _DummyExperiment(queue)
    preparation = PulseSchedule(["Q0"])
    preparation.add("Q0", Blank(8))
    result = measure_gef_populations(
        cast(Any, exp),
        targets="Q0",
        sequences={"name/s1": preparation, "other": preparation},
        n_shots=100,
        calibration_n_shots=100,
        n_bootstrap=0,
    )
    assert exp.measurement_service.batch_calls == [6, 6]
    for name, expected in zip(("name/s1", "other"), weights, strict=True):
        assert_allclose(
            result.data["populations"][name]["Q0"],
            np.array(expected) / 10,
            rtol=1e-6,
            atol=1e-7,
        )
    assert preparation.duration == 8
    assert preparation.labels == ["Q0"]
    for group in (exp.measurement_service.calls[:6], exp.measurement_service.calls[6:]):
        expected_labels = cast(PulseSchedule, group[0]["sequence"]).labels
        assert all(
            cast(PulseSchedule, call["sequence"]).labels == expected_labels
            for call in group
        )


def test_appended_analyzer_preserves_preparation_frequency() -> None:
    """GEF acquisition should retain an explicitly detuned preparation channel."""
    from qubex.contrib.experiment.gef_population_estimation import _append_analyzer

    preparation = PulseSchedule(["Q0"])
    preparation.add("Q0", Blank(8))
    preparation.set_frequency("Q0", 4.91)
    analyzer = PulseSchedule(["Q0"])
    analyzer.add("Q0", Blank(4))
    combined = _append_analyzer(preparation, analyzer)
    assert combined.get_frequency("Q0") == 4.91
    assert combined.duration == 12
    assert preparation.duration == 8
