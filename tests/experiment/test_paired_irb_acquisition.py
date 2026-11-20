"""Contracts for randomized paired IRB acquisition."""

import numpy as np
import pytest

from qubex.experiment.services.benchmarking_service import (
    _paired_points as paired_points,
)


@pytest.mark.parametrize("n_trials", [1, 2, 3, 30, 31])
def test_pair_plan_preserves_adjacency_and_balance(n_trials: int) -> None:
    """Pair plans should cover every trial once with balanced adjacent AB/BA points."""
    points = paired_points(n_trials, np.random.default_rng(42))
    assert len(points) == 2 * n_trials
    assert sorted(trial for trial, _ in points[::2]) == list(range(n_trials))
    for first, second in zip(points[::2], points[1::2], strict=True):
        assert first[0] == second[0]
        assert first[1] != second[1]
    n_interleaved_first = sum(protocol for _, protocol in points[::2])
    assert abs(2 * n_interleaved_first - n_trials) <= 1
    assert points == paired_points(n_trials, np.random.default_rng(42))


@pytest.mark.parametrize("is_2q", [False, True])
@pytest.mark.parametrize("in_parallel", [False, True])
@pytest.mark.parametrize("n_trials", [1, 3, 4])
@pytest.mark.parametrize("automatic", [False, True])
def test_irb_acquires_and_restores_paired_trials(
    monkeypatch: pytest.MonkeyPatch,
    is_2q: bool,
    in_parallel: bool,
    n_trials: int,
    automatic: bool,
) -> None:
    """IRB should acquire adjacent pairs and restore trial matrices for every target."""
    from types import SimpleNamespace
    from typing import Any, cast

    from qxpulse import Blank, PulseArray, PulseSchedule

    from qubex.clifford.clifford import Clifford
    from qubex.experiment.services import benchmarking_service as module

    service = cast(Any, object.__new__(module.BenchmarkingService))
    targets = ["A", "B"]
    resets: list[set[str]] = []
    calls: list[tuple[str, int, int, bool]] = []
    sweeps: list[list[tuple[int, bool]]] = []
    service.__dict__["_experiment_context"] = SimpleNamespace(
        state_centers={},
        classifiers={"Q0": object(), "Q1": object()},
        calib_note=SimpleNamespace(cr_params=dict.fromkeys(targets, object())),
        experiment_system=SimpleNamespace(
            get_target=lambda _: SimpleNamespace(is_cr=is_2q)
        ),
        resolve_qubit_label=lambda target: target,
        cr_pair=lambda _: ("Q0", "Q1"),
        reset_awg_and_capunits=lambda *, qubits: resets.append(set(qubits)),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        rabi_params={
            target: SimpleNamespace(normalize=lambda iq: np.real(iq))
            for target in targets
        }
    )

    def sequence(
        target: str, *, n: int, seed: int, interleaved_clifford: Any, **kwargs: Any
    ) -> Any:
        calls.append((target, n, seed, interleaved_clifford is not None))
        if is_2q:
            with PulseSchedule() as schedule:
                schedule.add("Q0", Blank(4))
            return schedule
        return PulseArray([])

    service.rb_sequence_1q = sequence
    service.rb_sequence_2q = sequence

    async def measure(schedule: Any, *, sweep_values: Any, **kwargs: Any) -> Any:
        assert kwargs["shot_averaging"] is not is_2q
        assert kwargs["time_integration"] is False
        results = []
        order = []
        for index in sweep_values:
            schedule(int(index))
            _, length, seed, protocol = calls[-1]
            probability = (
                0.0
                if automatic and length >= 1
                else 0.9 - 0.001 * seed - 0.01 * protocol
            )
            order.append((seed, protocol))
            results.append(
                SimpleNamespace(
                    probability=probability,
                    data={
                        target: [
                            SimpleNamespace(kerneled=np.asarray(2 * probability - 1))
                        ]
                        for target in targets
                    },
                )
            )
        sweeps.append(order)
        return SimpleNamespace(results=results)

    service.__dict__["_measurement_service"] = SimpleNamespace(
        run_sweep_measurement=measure
    )
    monkeypatch.setattr(
        module,
        "get_mitigated_probabilities",
        lambda point, **kwargs: {"00": point.probability},
    )
    monkeypatch.setattr(
        module.fitting,
        "fit_rb",
        lambda **kwargs: dict(
            A=0.5,
            p=0.99,
            p_err=0.001,
            C=0.5,
            avg_gate_error=0.005,
            avg_gate_fidelity=0.995,
            avg_gate_fidelity_err=0.0005,
        ),
    )
    monkeypatch.setattr(module.fitting, "plot_irb", lambda **kwargs: object())
    seeds = np.arange((5 if automatic else 2) * n_trials).reshape(-1, n_trials)
    result = service.irb_experiment(
        targets,
        interleaved_clifford=Clifford.X90(),
        n_cliffords_range=None if automatic else [0, 1],
        max_n_cliffords=8,
        seeds=seeds,
        n_trials=n_trials,
        in_parallel=in_parallel,
        acquisition_seed=42,
        time_integration=False,
        plot=False,
        save_image=False,
    )
    assert len(resets) == 1
    assert len(sweeps) == (2 if in_parallel else 4)
    for sweep in sweeps:
        for first, second in zip(sweep[::2], sweep[1::2], strict=True):
            assert first[0] == second[0]
            assert first[1] != second[1]
    for target in targets:
        for key, protocol in [("rb_data", 0), ("irb_data", 1)]:
            np.testing.assert_allclose(
                result[target][key]["trials"],
                np.vstack(
                    [0.9 - 0.001 * seeds[0] - 0.01 * protocol, np.zeros(n_trials)]
                )
                if automatic
                else 0.9 - 0.001 * seeds - 0.01 * protocol,
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_array_equal(result[target][key]["seeds"], seeds[:2])
        assert len(result[target]["acquisition"]["planned_order"]) == 2 * n_trials
