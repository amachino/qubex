"""Focused acquisition tests for paired interleaved randomized benchmarking."""

from __future__ import annotations

import importlib
import random
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from qubex.clifford.clifford import Clifford
from qubex.clifford.clifford_generator import CliffordGenerator
from qubex.contrib import measure_paired_irb


class _RabiParam:
    """Map an encoded IQ value to a normalized Bloch-Z value."""

    @staticmethod
    def normalize(iq: Any) -> float:
        return 2.0 * float(np.real(np.mean(np.asarray(iq)))) - 1.0


class _MeasurementService:
    """Execute synthetic sweep points while retaining call boundaries."""

    def __init__(self, *, encode_association: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.encode_association = encode_association

    async def run_sweep_measurement(
        self,
        schedule: Any,
        *,
        sweep_values: Any,
        **kwargs: Any,
    ) -> Any:
        values = np.asarray(sweep_values)
        self.calls.append({"sweep_values": values.copy(), **kwargs})
        results = []
        for value in values:
            marker = schedule(int(value))
            decay = 0.98 if marker.interleaved_clifford is None else 0.95
            probability = (
                0.2
                + 0.4 * marker.seed / 2**32
                + 0.1 * (marker.interleaved_clifford is not None)
                if self.encode_association
                else 0.48 * decay**marker.n_cliffords + 0.5
            )
            config = SimpleNamespace(
                shot_averaging=kwargs["shot_averaging"],
                time_integration=kwargs["time_integration"],
            )
            capture = SimpleNamespace(
                data=np.asarray(probability, dtype=np.complex128),
                config=config,
            )
            results.append(SimpleNamespace(data={"Q0": [capture]}))
        return SimpleNamespace(results=results)


class _Experiment:
    """Provide the public experiment surface used by paired IRB."""

    def __init__(self) -> None:
        self.sequence_calls: list[Any] = []
        self.measurement_service = _MeasurementService()
        self.pulse = SimpleNamespace(
            rabi_params={"Q0": _RabiParam()},
            validate_rabi_params=lambda targets: None,
        )
        self.experiment_system = SimpleNamespace(
            get_target=lambda target: SimpleNamespace(is_cr=False),
            resolve_qubit_label=lambda target: target,
        )
        self.ctx = SimpleNamespace(
            reset_awg_and_capunits=lambda *, qubits: None,
            classifiers={},
        )
        self.interleaved = Clifford.X90()
        self.benchmarking_service = SimpleNamespace(clifford={"X90": self.interleaved})

    def rb_sequence(self, target: str, **kwargs: Any) -> Any:
        marker = SimpleNamespace(
            target=target,
            n_cliffords=kwargs["n"],
            seed=kwargs["seed"],
            interleaved_clifford=kwargs["interleaved_clifford"],
        )
        self.sequence_calls.append(marker)
        return marker


@pytest.mark.parametrize(
    ("clifford_type", "interleaved_clifford"),
    [("1Q", Clifford.X90()), ("2Q", Clifford.ZX90())],
)
def test_reference_and_interleaved_sequences_share_random_cliffords(
    clifford_type: Any,
    interleaved_clifford: Clifford,
) -> None:
    """A shared seed must generate the same underlying random Cliffords."""
    generator = CliffordGenerator()
    random_state = random.getstate()
    try:
        reference_cliffords, _ = generator.create_rb_sequences(
            n=12,
            type=clifford_type,
            seed=314159,
        )
        interleaved_cliffords, _ = generator.create_irb_sequences(
            n=12,
            interleave=interleaved_clifford,
            type=clifford_type,
            seed=314159,
        )
    finally:
        random.setstate(random_state)

    assert reference_cliffords == interleaved_cliffords


def test_default_sweep_submits_all_pairs_and_reuses_each_pair_seed() -> None:
    """The default should place every adjacent ref/IRB pair in one sweep call."""
    exp: Any = _Experiment()

    result = measure_paired_irb(
        exp,
        "Q0",
        interleaved_clifford="X90",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=3,
        sequence_seed=11,
        acquisition_seed=22,
    )

    assert len(exp.measurement_service.calls) == 1
    assert len(exp.measurement_service.calls[0]["sweep_values"]) == 24
    acquisition = result["Q0"]["acquisition"]
    assert acquisition["pairs_per_sweep"] is None
    assert acquisition["n_sweep_calls"] == 1
    assert acquisition["n_measurement_points"] == 24
    assert acquisition["seeds"].shape == (4, 3)
    assert len(np.unique(acquisition["seeds"])) == 12

    for pair_start in range(0, len(exp.sequence_calls), 2):
        first, second = exp.sequence_calls[pair_start : pair_start + 2]
        assert first.n_cliffords == second.n_cliffords
        assert first.seed == second.seed
        assert {first.interleaved_clifford, second.interleaved_clifford} == {
            None,
            exp.interleaved,
        }


def test_explicit_pairs_per_sweep_still_chunks_complete_pairs() -> None:
    """A positive chunk size should remain available for smaller backend calls."""
    exp: Any = _Experiment()

    result = measure_paired_irb(
        exp,
        "Q0",
        interleaved_clifford="X90",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=3,
        sequence_seed=11,
        acquisition_seed=22,
        pairs_per_sweep=2,
    )

    assert [len(call["sweep_values"]) for call in exp.measurement_service.calls] == [
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    assert result["Q0"]["acquisition"]["n_sweep_calls"] == 6


def test_results_map_back_to_their_planned_pair_cells() -> None:
    """Global randomization must not break result-to-(length, trial) association."""
    exp: Any = _Experiment()
    exp.measurement_service = _MeasurementService(encode_association=True)

    result = measure_paired_irb(
        exp,
        "Q0",
        interleaved_clifford="X90",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=3,
        sequence_seed=13,
        acquisition_seed=27,
    )

    target = result["Q0"]
    for pair in target["acquisition"]["planned_order"]:
        length_index = pair["length_index"]
        trial_index = pair["trial_index"]
        base_probability = 0.2 + 0.4 * pair["seed"] / 2**32
        assert target["reference"]["trials"][length_index, trial_index] == (
            pytest.approx(base_probability)
        )
        assert target["interleaved"]["trials"][length_index, trial_index] == (
            pytest.approx(base_probability + 0.1)
        )


def test_incomplete_sweep_result_is_rejected() -> None:
    """A backend result-count mismatch must fail instead of mispairing data."""

    class _IncompleteMeasurementService(_MeasurementService):
        async def run_sweep_measurement(
            self,
            schedule: Any,
            *,
            sweep_values: Any,
            **kwargs: Any,
        ) -> Any:
            sweep = await super().run_sweep_measurement(
                schedule,
                sweep_values=sweep_values,
                **kwargs,
            )
            return SimpleNamespace(results=sweep.results[:-1])

    exp: Any = _Experiment()
    exp.measurement_service = _IncompleteMeasurementService()

    with pytest.raises(RuntimeError, match="result count"):
        measure_paired_irb(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            sequence_seed=17,
        )


@pytest.mark.parametrize(
    ("mitigate_readout", "expected_probability"),
    [(True, 0.91), (False, 0.89)],
)
def test_two_qubit_measurement_uses_unaveraged_shots_and_p00(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mitigate_readout: bool,
    expected_probability: float,
) -> None:
    """CR acquisition should retain shots and convert each point to P(00)."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    exp: Any = _Experiment()
    exp.experiment_system = SimpleNamespace(
        get_target=lambda target: SimpleNamespace(is_cr=True),
    )
    exp.ctx = SimpleNamespace(
        reset_awg_and_capunits=lambda *, qubits: None,
        classifiers={"Q0": object(), "Q1": object()},
        calib_note=SimpleNamespace(cr_params={"CR0": object()}),
        cr_pair=lambda target: ("Q0", "Q1"),
    )
    exp.interleaved = Clifford.IX90()
    exp.benchmarking_service.clifford = {"IX90": exp.interleaved}
    converted = SimpleNamespace(
        get_mitigated_probabilities=Mock(return_value={"00": 0.91}),
        get_probabilities=Mock(return_value={"00": 0.89}),
    )
    monkeypatch.setattr(
        module.MeasurementResultConverter,
        "to_measure_result",
        Mock(return_value=converted),
    )

    result = measure_paired_irb(
        exp,
        "CR0",
        interleaved_clifford="IX90",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=2,
        sequence_seed=5,
        mitigate_readout=mitigate_readout,
    )

    call = exp.measurement_service.calls[0]
    assert call["shot_averaging"] is False
    np.testing.assert_allclose(
        result["CR0"]["reference"]["trials"], expected_probability
    )
    np.testing.assert_allclose(
        result["CR0"]["interleaved"]["trials"], expected_probability
    )
