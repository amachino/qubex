"""Tests for paired interleaved randomized-benchmarking acquisition."""

from __future__ import annotations

import importlib
import random
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from qubex.clifford.clifford import Clifford
from qubex.clifford.clifford_generator import CliffordGenerator
from qubex.contrib import (
    measure_paired_irb,
    paired_interleaved_randomized_benchmarking,
)
from qubex.measurement.models import CaptureData, MeasurementConfig, MeasurementResult
from qubex.measurement.services.measurement_execution_service import (
    MeasurementExecutionService,
)


class _RabiParam:
    """Map an encoded IQ value to a normalized Bloch-Z value."""

    @staticmethod
    def normalize(iq: Any) -> float:
        """Decode a synthetic ground-state probability."""
        return 2.0 * float(np.real(np.mean(np.asarray(iq)))) - 1.0


class _MeasurementService:
    """Execute synthetic sweep points while retaining call boundaries."""

    def __init__(
        self,
        *,
        include_leading_capture: bool = False,
        encode_association: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.include_leading_capture = include_leading_capture
        self.encode_association = encode_association

    async def run_sweep_measurement(
        self,
        schedule: Any,
        *,
        sweep_values: Any,
        **kwargs: Any,
    ) -> Any:
        """Return one canonical-shaped synthetic result per schedule."""
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
            capture_config = SimpleNamespace(
                shot_averaging=kwargs["shot_averaging"],
                time_integration=kwargs["time_integration"],
            )
            capture_data = (
                np.asarray(probability, dtype=np.complex128)
                if kwargs["time_integration"]
                else np.full(2, probability / 2.0, dtype=np.complex128)
            )
            captures = [SimpleNamespace(data=capture_data, config=capture_config)]
            if self.include_leading_capture:
                captures.insert(
                    0,
                    SimpleNamespace(
                        data=np.asarray(0.01, dtype=np.complex128),
                        config=capture_config,
                    ),
                )
            results.append(SimpleNamespace(data={"Q0": captures}))
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
        """Record schedule construction arguments and return a marker."""
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
    """A shared seed must generate the same underlying 1Q/2Q Cliffords."""
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


def test_measurement_batches_whole_pairs_and_reuses_each_cell_seed() -> None:
    """Each sweep chunk should retain complete adjacent pairs with shared seeds."""
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
        n_shots=100,
    )

    assert [len(call["sweep_values"]) for call in exp.measurement_service.calls] == [
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    for pair_start in range(0, len(exp.sequence_calls), 2):
        first, second = exp.sequence_calls[pair_start : pair_start + 2]
        assert first.n_cliffords == second.n_cliffords
        assert first.seed == second.seed
        assert {first.interleaved_clifford, second.interleaved_clifford} == {
            None,
            exp.interleaved,
        }

    acquisition = result["Q0"]["acquisition"]
    assert acquisition["seeds"].shape == (4, 3)
    assert len(np.unique(acquisition["seeds"])) == 12
    assert acquisition["sequence_seed"] == 11
    assert acquisition["n_sweep_calls"] == 6
    assert acquisition["n_measurement_points"] == 24
    first_protocols = [pair["first_protocol"] for pair in acquisition["planned_order"]]
    assert first_protocols.count("reference") == 6
    assert first_protocols.count("interleaved") == 6
    for length_index in range(4):
        length_protocols = [
            pair["first_protocol"]
            for pair in acquisition["planned_order"]
            if pair["length_index"] == length_index
        ]
        assert (
            abs(
                length_protocols.count("reference")
                - length_protocols.count("interleaved")
            )
            <= 1
        )


def test_multi_pair_sweep_results_remain_associated_with_planned_cells() -> None:
    """Returned point order should map every protocol result to its planned cell."""
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
        pairs_per_sweep=4,
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


def test_measurement_rejects_an_incomplete_sweep_result() -> None:
    """A backend result count mismatch should fail instead of mispairing trials."""

    class _IncompleteMeasurementService(_MeasurementService):
        async def run_sweep_measurement(
            self,
            schedule: Any,
            *,
            sweep_values: Any,
            **kwargs: Any,
        ) -> Any:
            """Drop one synthetic point to emulate an incomplete backend result."""
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
            pairs_per_sweep=8,
        )


def test_measurement_rejects_nonfinite_survival_probability() -> None:
    """A nonfinite normalized survival should fail before entering raw trials."""
    exp: Any = _Experiment()
    exp.pulse.rabi_params["Q0"] = SimpleNamespace(normalize=lambda iq: np.nan)

    with pytest.raises(ValueError, match="survival probability must be finite"):
        measure_paired_irb(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            sequence_seed=19,
            pairs_per_sweep=8,
        )


def test_real_sweep_service_preserves_multi_pair_result_order() -> None:
    """The real sweep orchestrator should preserve every requested pair point."""
    execution: Any = MeasurementExecutionService.__new__(MeasurementExecutionService)

    def cannot_batch(self: Any) -> bool:
        return False

    async def run_measurement(
        self: Any,
        *,
        schedule: Any,
        config: MeasurementConfig,
    ) -> Any:
        del self
        decay_offset = 0.1 * (schedule.interleaved_clifford is not None)
        probability = 0.2 + 0.4 * schedule.seed / 2**32 + decay_offset
        capture = CaptureData.from_primary_data(
            target="Q0",
            data=np.asarray([probability], dtype=np.complex128),
            config=config,
            sampling_period=2.0,
        )
        return MeasurementResult(
            data={
                "Q0": [capture],
            },
            measurement_config=config,
            device_config={},
        )

    execution._can_use_batch_execution = MethodType(  # noqa: SLF001
        cannot_batch,
        execution,
    )
    execution.run_measurement = MethodType(run_measurement, execution)

    class _ExecutionBackedService:
        """Adapt the real ordered sweep method to the experiment test surface."""

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.returned_sweep_values: list[list[Any]] = []

        async def run_sweep_measurement(
            self,
            schedule: Any,
            *,
            sweep_values: Any,
            **kwargs: Any,
        ) -> Any:
            values = np.asarray(sweep_values).tolist()
            self.calls.append({"sweep_values": values, **kwargs})
            config = MeasurementConfig(
                n_shots=kwargs["n_shots"],
                shot_interval=kwargs["shot_interval"],
                shot_averaging=kwargs["shot_averaging"],
                time_integration=kwargs["time_integration"],
                state_classification=False,
            )
            result = await execution.run_sweep_measurement(
                schedule,
                sweep_values=values,
                config=config,
            )
            self.returned_sweep_values.append(result.sweep_values)
            return result

    exp: Any = _Experiment()
    service = _ExecutionBackedService()
    exp.measurement_service = service

    result = measure_paired_irb(
        exp,
        "Q0",
        interleaved_clifford="X90",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=3,
        sequence_seed=13,
        acquisition_seed=27,
        pairs_per_sweep=4,
    )

    assert all(
        returned == call["sweep_values"]
        for returned, call in zip(
            service.returned_sweep_values,
            service.calls,
            strict=True,
        )
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


def test_ab_ba_order_remains_globally_balanced_for_odd_grid_and_trials() -> None:
    """An odd number of pairs should have the smallest possible order imbalance."""
    exp: Any = _Experiment()

    result = measure_paired_irb(
        exp,
        "Q0",
        interleaved_clifford="X90",
        n_cliffords_range=[0, 1, 2, 4, 8],
        n_trials=3,
        sequence_seed=11,
        acquisition_seed=22,
        pairs_per_sweep=15,
    )

    first_protocols = [
        pair["first_protocol"] for pair in result["Q0"]["acquisition"]["planned_order"]
    ]
    assert (
        abs(first_protocols.count("reference") - first_protocols.count("interleaved"))
        == 1
    )


def test_acquisition_rng_changes_only_the_pair_order() -> None:
    """Order randomization should not alter sequence seeds or indexed trial data."""
    first_exp: Any = _Experiment()
    second_exp: Any = _Experiment()
    common_options = {
        "interleaved_clifford": "X90",
        "n_cliffords_range": [0, 1, 2, 4],
        "n_trials": 3,
        "sequence_seed": 11,
        "pairs_per_sweep": 12,
    }

    first = measure_paired_irb(
        first_exp,
        "Q0",
        acquisition_seed=22,
        **common_options,
    )
    second = measure_paired_irb(
        second_exp,
        "Q0",
        acquisition_seed=23,
        **common_options,
    )

    first_payload = first["Q0"]
    second_payload = second["Q0"]
    np.testing.assert_array_equal(
        first_payload["acquisition"]["seeds"],
        second_payload["acquisition"]["seeds"],
    )
    assert (
        first_payload["acquisition"]["planned_order"]
        != second_payload["acquisition"]["planned_order"]
    )
    np.testing.assert_array_equal(
        first_payload["reference"]["trials"],
        second_payload["reference"]["trials"],
    )
    np.testing.assert_array_equal(
        first_payload["interleaved"]["trials"],
        second_payload["interleaved"]["trials"],
    )


def test_acquisition_order_is_random_by_default_and_fully_recorded() -> None:
    """The default acquisition RNG should be fresh while preserving its exact plan."""
    exp: Any = _Experiment()

    result = measure_paired_irb(
        exp,
        "Q0",
        interleaved_clifford="X90",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=2,
        sequence_seed=11,
        pairs_per_sweep=8,
    )

    acquisition = result["Q0"]["acquisition"]
    assert acquisition["acquisition_seed"] is None
    assert len(acquisition["planned_order"]) == 8


def test_measurement_rejects_a_seed_matrix_with_the_wrong_shape() -> None:
    """Measurement should require one independent seed for every pair cell."""
    exp: Any = _Experiment()

    with pytest.raises(ValueError, match="shape"):
        measure_paired_irb(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=3,
            seeds=np.arange(3),
        )


def test_measurement_rejects_two_length_grid_sources() -> None:
    """An explicit grid and a default-grid maximum should not conflict."""
    exp: Any = _Experiment()

    with pytest.raises(ValueError, match="Specify only one"):
        measure_paired_irb(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            max_n_cliffords=64,
            n_trials=2,
            sequence_seed=7,
        )


def test_measurement_rejects_duplicate_explicit_sequence_seeds() -> None:
    """Explicit seed matrices should preserve independence between all cells."""
    exp: Any = _Experiment()
    seeds = np.arange(12, dtype=np.int64).reshape(4, 3)
    seeds[3, 2] = seeds[0, 0]

    with pytest.raises(ValueError, match="distinct"):
        measure_paired_irb(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=3,
            seeds=seeds,
        )


def test_measurement_rejects_a_clifford_with_the_wrong_qubit_arity() -> None:
    """A two-qubit Clifford should not be accepted for a one-qubit target."""
    exp: Any = _Experiment()

    with pytest.raises(ValueError, match="one-qubit"):
        measure_paired_irb(
            exp,
            "Q0",
            interleaved_clifford=Clifford.IX90(),
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            sequence_seed=7,
        )


def test_measurement_rejects_truthy_non_boolean_options() -> None:
    """Measurement flags should not silently coerce strings to true."""
    exp: Any = _Experiment()

    with pytest.raises(TypeError, match="mitigate_readout"):
        measure_paired_irb(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            sequence_seed=7,
            mitigate_readout="false",  # type: ignore[arg-type]
        )


def test_one_qubit_measurement_uses_the_terminal_capture() -> None:
    """A preceding capture should not replace the final RB readout."""
    exp: Any = _Experiment()
    exp.measurement_service = _MeasurementService(include_leading_capture=True)

    result = measure_paired_irb(
        exp,
        "Q0",
        interleaved_clifford="X90",
        n_cliffords_range=[0, 1, 2, 4],
        n_trials=2,
        sequence_seed=7,
        pairs_per_sweep=8,
        time_integration=False,
    )

    reference = result["Q0"]["reference"]["trials"]
    interleaved = result["Q0"]["interleaved"]["trials"]
    expected_reference = 0.48 * 0.98 ** np.array([0, 1, 2, 4]) + 0.5
    expected_interleaved = 0.48 * 0.95 ** np.array([0, 1, 2, 4]) + 0.5
    np.testing.assert_allclose(
        reference,
        np.broadcast_to(expected_reference[:, None], reference.shape),
    )
    np.testing.assert_allclose(
        interleaved,
        np.broadcast_to(expected_interleaved[:, None], interleaved.shape),
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
    """CR acquisition should retain shots and convert each point to `P(00)`."""
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
        pairs_per_sweep=8,
        mitigate_readout=mitigate_readout,
    )

    call = exp.measurement_service.calls[0]
    assert call["shot_averaging"] is False
    assert call["time_integration"] is True
    np.testing.assert_allclose(
        result["CR0"]["reference"]["trials"], expected_probability
    )
    np.testing.assert_allclose(
        result["CR0"]["interleaved"]["trials"], expected_probability
    )
    selected_probability = (
        converted.get_mitigated_probabilities
        if mitigate_readout
        else converted.get_probabilities
    )
    unused_probability = (
        converted.get_probabilities
        if mitigate_readout
        else converted.get_mitigated_probabilities
    )
    assert selected_probability.call_count == 16
    unused_probability.assert_not_called()
    converter_calls = module.MeasurementResultConverter.to_measure_result.call_args_list
    assert all(call.kwargs["index"] == -1 for call in converter_calls)


def test_composed_api_returns_full_analysis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The composed API should analyze the same paired acquisition in one call."""
    exp: Any = _Experiment()
    caplog.set_level(
        "INFO",
        logger="qubex.contrib.experiment.paired_interleaved_randomized_benchmarking",
    )

    with pytest.warns(RuntimeWarning) as warning_records:
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            sequence_seed=3,
            n_bootstrap=5,
            plot=False,
            save_image=False,
        )

    assert any("SEM floor" in str(record.message) for record in warning_records)
    expected = 1.0 - 0.5 * (1.0 - 0.95 / 0.98)
    assert result["Q0"]["gate_fidelity"] == pytest.approx(expected, abs=1e-8)
    bootstrap = result["Q0"]["bootstrap"]
    assert bootstrap["execution_skipped"] is True
    assert bootstrap["n_attempted"] == 0
    assert bootstrap["n_success"] == 0
    assert "optimizer = not_run" in caplog.text


def test_composed_api_validates_analysis_options_before_measurement() -> None:
    """Invalid analysis options should not waste a hardware acquisition."""
    exp: Any = _Experiment()

    with pytest.raises(ValueError, match="n_bootstrap"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            sequence_seed=3,
            n_bootstrap=-1,
        )

    assert exp.measurement_service.calls == []
