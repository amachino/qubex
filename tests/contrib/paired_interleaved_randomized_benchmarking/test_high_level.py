"""Focused high-level tests for paired interleaved randomized benchmarking."""

from __future__ import annotations

import importlib
import inspect
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule

from qubex.clifford.clifford import Clifford
from qubex.contrib import (
    measure_paired_irb,
    paired_interleaved_randomized_benchmarking,
)


class _RabiParam:
    @staticmethod
    def normalize(iq: Any) -> float:
        return 2.0 * float(np.real(np.mean(np.asarray(iq)))) - 1.0


class _MeasurementService:
    """Execute synthetic serial or parallel sweep schedules."""

    def __init__(
        self,
        sequence_calls: list[Any],
        *,
        decays: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.sequence_calls = sequence_calls
        self.decays = decays or {}
        self.calls: list[dict[str, Any]] = []

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
            call_start = len(self.sequence_calls)
            schedule(int(value))
            markers = self.sequence_calls[call_start:]
            point_data = {}
            for marker in markers:
                reference_decay, interleaved_decay = self.decays.get(
                    marker.target,
                    (0.98, 0.95),
                )
                decay = (
                    reference_decay
                    if marker.interleaved_clifford is None
                    else interleaved_decay
                )
                probability = 0.48 * decay**marker.n_cliffords + 0.50
                config = SimpleNamespace(
                    shot_averaging=kwargs["shot_averaging"],
                    time_integration=kwargs["time_integration"],
                )
                point_data[marker.target] = [
                    SimpleNamespace(
                        data=np.asarray(probability, dtype=np.complex128),
                        config=config,
                    )
                ]
            results.append(SimpleNamespace(data=point_data))
        return SimpleNamespace(results=results)


class _Experiment:
    """Provide the experiment surface used by high-level tests."""

    def __init__(
        self,
        targets: tuple[str, ...] = ("Q0", "Q1"),
        *,
        decays: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.sequence_calls: list[Any] = []
        self.measurement_service = _MeasurementService(
            self.sequence_calls,
            decays=decays,
        )
        self.pulse = SimpleNamespace(
            rabi_params={target: _RabiParam() for target in targets},
            validate_rabi_params=lambda selected: None,
        )
        self.experiment_system = SimpleNamespace(
            get_target=lambda target: SimpleNamespace(is_cr=False),
            resolve_qubit_label=lambda target: target,
        )
        self.ctx = SimpleNamespace(
            reset_awg_and_capunits=lambda *, qubits: None,
            classifiers={},
            calib_note=SimpleNamespace(cr_params={}),
        )
        self.interleaved_1q = Clifford.X90()
        self.benchmarking_service = SimpleNamespace(
            clifford={"X90": self.interleaved_1q}
        )

    def rb_sequence(self, target: str, **kwargs: Any) -> PulseSchedule:
        marker = SimpleNamespace(
            target=target,
            n_cliffords=kwargs["n"],
            seed=kwargs["seed"],
            interleaved_clifford=kwargs["interleaved_clifford"],
            interleaved_waveform=kwargs["interleaved_waveform"],
            x90=kwargs["x90"],
            zx90=kwargs["zx90"],
        )
        self.sequence_calls.append(marker)
        with PulseSchedule([target]) as schedule:
            schedule.add(target, Blank(duration=2.0 * (marker.n_cliffords + 1)))
        return schedule


def test_public_defaults_and_fixed_grids_match_the_fast_default_workflow() -> None:
    """The public defaults should use fixed grids, one sweep, and 1800 s timeout."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    high_signature = inspect.signature(paired_interleaved_randomized_benchmarking)
    low_signature = inspect.signature(measure_paired_irb)

    assert high_signature.parameters["auto_range"].default is False
    assert high_signature.parameters["blocks_per_sweep"].default is None
    assert low_signature.parameters["blocks_per_sweep"].default is None
    assert "pairs_per_sweep" not in high_signature.parameters
    assert "pairs_per_sweep" not in low_signature.parameters
    assert high_signature.parameters["sweep_timeout"].default == 1800.0
    assert low_signature.parameters["sweep_timeout"].default == 1800.0

    one_qubit = module._TargetSpec(  # noqa: SLF001
        is_two_qubit=False,
        dimension=2,
        qubits=("Q0",),
    )
    two_qubit = module._TargetSpec(  # noqa: SLF001
        is_two_qubit=True,
        dimension=4,
        qubits=("Q0", "Q1"),
    )
    np.testing.assert_array_equal(
        module._default_fixed_n_cliffords(one_qubit),  # noqa: SLF001
        [0, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    np.testing.assert_array_equal(
        module._default_fixed_n_cliffords(two_qubit),  # noqa: SLF001
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
    )


def test_default_high_level_call_uses_full_fixed_grid_without_pilot() -> None:
    """Omitting range controls should use the 1Q fixed grid and one sweep call."""
    exp: Any = _Experiment(targets=("Q0",))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_trials=2,
            sequence_seed=21,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    np.testing.assert_array_equal(
        result["Q0"]["acquisition"]["n_cliffords"],
        [0, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    assert result["Q0"]["grid_selection"]["enabled"] is False
    assert result["Q0"]["grid_selection"]["pilot_raw_result"] is None
    assert result["Q0"]["grid_selection"]["pilot_stop_reason"] == (
        "auto_range_disabled"
    )
    assert len(exp.measurement_service.calls) == 1
    assert len(exp.measurement_service.calls[0]["sweep_values"]) == 36


def test_multi_waveform_call_uses_one_shared_reference_per_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-arm IRB should measure one reference plus every candidate per block."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    monkeypatch.setattr(module.go.Figure, "show", lambda self: None)
    exp: Any = _Experiment(targets=("Q0",))
    waveforms = {
        "rotary": PulseSchedule(),
        "no_rotary": PulseSchedule(),
        "ppr": PulseSchedule(),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            interleaved_waveforms=waveforms,
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=8,
            sequence_seed=21,
            acquisition_seed=22,
            n_bootstrap=8,
            bootstrap_seed=23,
            plot=True,
            save_image=False,
        )

    target = result["Q0"]
    assert list(target["interleaved"]) == list(waveforms)
    assert len(exp.measurement_service.calls) == 1
    assert len(exp.measurement_service.calls[0]["sweep_values"]) == 192
    assert target["acquisition"]["n_measurement_points"] == 192
    assert target["metadata"]["schema_version"] == 2
    assert len(target["comparisons"]) == 3
    assert result.figure is None
    assert set(result.figures or {}) == {
        "Q0::rotary",
        "Q0::no_rotary",
        "Q0::ppr",
    }
    for comparison in target["comparisons"]:
        assert comparison["gate_fidelity_difference"] == pytest.approx(0.0)
        assert comparison["gate_fidelity_difference_ci95"] is not None

    reference_trials = target["reference"]["trials"]
    for candidate in waveforms:
        assert target["interleaved"][candidate]["trials"].shape == (6, 8)
        assert target["interleaved"][candidate]["gate_fidelity"] == pytest.approx(
            target["interleaved"]["rotary"]["gate_fidelity"]
        )
    assert reference_trials.shape == (6, 8)


def test_multi_waveform_supports_parallel_targets() -> None:
    """Parallel targets should share block points while keeping independent seeds."""
    exp: Any = _Experiment(targets=("Q0", "Q1"))
    waveforms = {
        "rotary": {"Q0": PulseSchedule(), "Q1": PulseSchedule()},
        "ppr": {"Q0": PulseSchedule(), "Q1": PulseSchedule()},
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = paired_interleaved_randomized_benchmarking(
            exp,
            ("Q0", "Q1"),
            interleaved_clifford="X90",
            interleaved_waveforms=waveforms,
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            sequence_seed=31,
            acquisition_seed=32,
            in_parallel=True,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert list(result.data) == ["Q0", "Q1"]
    assert len(exp.measurement_service.calls) == 1
    assert len(exp.measurement_service.calls[0]["sweep_values"]) == 24
    assert result["Q0"]["acquisition"]["in_parallel"] is True
    assert result["Q0"]["acquisition"]["schedule_alignment"] == (
        "left_padded_to_group_max_duration"
    )
    assert result["Q0"]["acquisition"]["batching_semantics"] == (
        "parallel_targets_in_complete_multi_arm_blocks"
    )
    assert "target_schedule_alignment" not in result["Q0"]["acquisition"]
    assert not np.array_equal(
        result["Q0"]["acquisition"]["seeds"],
        result["Q1"]["acquisition"]["seeds"],
    )
    for target in ("Q0", "Q1"):
        acquisition = result[target]["acquisition"]
        seeds = acquisition["seeds"]
        for block in acquisition["planned_order"]:
            assert (
                block["seed"]
                == seeds[
                    block["length_index"],
                    block["trial_index"],
                ]
            )


@pytest.mark.parametrize(
    "candidate_name",
    ["", "reference", "REFERENCE", "has space", "slash/name", "-leading"],
)
def test_multi_waveform_candidate_names_are_validated(candidate_name: str) -> None:
    """Candidate names should be safe, nonreserved identifiers."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(ValueError, match="candidate name"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            interleaved_waveforms={candidate_name: PulseSchedule()},
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )
    assert exp.measurement_service.calls == []


def test_multi_waveform_rejects_conflicting_or_adaptive_configuration() -> None:
    """Plural waveforms should reject singular overrides and auto-range."""
    exp: Any = _Experiment(targets=("Q0",))
    waveform = PulseSchedule()

    with pytest.raises(ValueError, match="only one"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            interleaved_waveform=waveform,
            interleaved_waveforms={"candidate": waveform},
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )
    with pytest.raises(ValueError, match="auto_range"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            interleaved_waveforms={"candidate": waveform},
            auto_range=True,
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )
    assert exp.measurement_service.calls == []


def test_multi_waveform_warns_before_measurement_when_bootstrap_trials_are_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workflow should warn before acquiring undersized Williams strata."""
    module = importlib.import_module(
        "qubex.contrib.experiment.paired_interleaved_randomized_benchmarking"
    )
    exp: Any = _Experiment(targets=("Q0",))
    warning_emitted = False
    original_warning = module._warn_if_multi_bootstrap_design_is_undersized  # noqa: SLF001
    original_acquisition = module._run_multi_single_target_acquisition  # noqa: SLF001

    def record_warning(**kwargs: Any) -> None:
        nonlocal warning_emitted
        original_warning(**kwargs)
        warning_emitted = True

    def assert_warning_precedes_measurement(*args: Any, **kwargs: Any) -> Any:
        assert warning_emitted
        return original_acquisition(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "_warn_if_multi_bootstrap_design_is_undersized",
        record_warning,
    )
    monkeypatch.setattr(
        module,
        "_run_multi_single_target_acquisition",
        assert_warning_precedes_measurement,
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            interleaved_waveforms={
                "rotary": PulseSchedule(),
                "ppr": PulseSchedule(),
            },
            n_cliffords_range=[0, 1, 2, 4],
            n_trials=11,
            n_bootstrap=8,
            plot=False,
            save_image=False,
        )

    assert warning_emitted
    assert any("at least 12 trials" in str(record.message) for record in records)
    assert result["Q0"]["bootstrap"]["resampling_valid"] is False


def test_invalid_seed_configuration_fails_without_bootstrap_trial_warning() -> None:
    """Input errors should be reported before advisory bootstrap warnings."""
    exp: Any = _Experiment(targets=("Q0",))
    explicit_seeds = np.arange(8, dtype=np.int64).reshape(4, 2)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="only one of `seeds` and `sequence_seed`"):
            paired_interleaved_randomized_benchmarking(
                exp,
                "Q0",
                interleaved_clifford="X90",
                interleaved_waveforms={
                    "rotary": PulseSchedule(),
                    "ppr": PulseSchedule(),
                },
                n_cliffords_range=[0, 1, 2, 4],
                n_trials=2,
                seeds=explicit_seeds,
                sequence_seed=1,
                n_bootstrap=8,
                plot=False,
                save_image=False,
            )

    assert records == []
    assert exp.measurement_service.calls == []


def test_invalid_acquisition_seed_fails_without_bootstrap_trial_warning() -> None:
    """Preflight acquisition errors should precede advisory bootstrap warnings."""
    exp: Any = _Experiment(targets=("Q0",))

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="acquisition_seed"):
            paired_interleaved_randomized_benchmarking(
                exp,
                "Q0",
                interleaved_clifford="X90",
                interleaved_waveforms={
                    "rotary": PulseSchedule(),
                    "ppr": PulseSchedule(),
                },
                n_cliffords_range=[0, 1, 2, 4],
                n_trials=2,
                acquisition_seed=-1,
                n_bootstrap=8,
                plot=False,
                save_image=False,
            )

    assert records == []
    assert exp.measurement_service.calls == []


def test_multi_waveform_candidate_names_are_unique_ignoring_case() -> None:
    """Candidate names differing only by case should be rejected."""
    exp: Any = _Experiment(targets=("Q0",))

    with pytest.raises(ValueError, match="unique ignoring case"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            interleaved_waveforms={
                "Rotary": PulseSchedule(),
                "rotary": PulseSchedule(),
            },
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )


def test_explicit_auto_range_still_runs_a_fresh_pilot_and_main_acquisition() -> None:
    """The nondefault auto-range path should remain available as a smoke test."""
    exp: Any = _Experiment(
        targets=("Q0",),
        decays={"Q0": (0.90, 0.85)},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            auto_range=True,
            max_n_cliffords=128,
            pilot_n_trials=4,
            n_trials=4,
            sequence_seed=11,
            acquisition_seed=13,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    grid = result["Q0"]["grid_selection"]
    assert grid["enabled"] is True
    assert grid["pilot_raw_result"] is not None
    np.testing.assert_array_equal(
        result["Q0"]["acquisition"]["n_cliffords"],
        grid["selected_main_grid"],
    )
    assert not np.array_equal(
        grid["pilot_raw_result"]["acquisition"]["seeds"],
        result["Q0"]["acquisition"]["seeds"],
    )


def test_serial_targets_receive_independent_sequence_seed_streams() -> None:
    """Serial multi-target orchestration should not reuse one seed matrix."""
    exp: Any = _Experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = paired_interleaved_randomized_benchmarking(
            exp,
            ("Q0", "Q1"),
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=2,
            sequence_seed=101,
            acquisition_seed=103,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )

    assert list(result.data) == ["Q0", "Q1"]
    assert not np.array_equal(
        result["Q0"]["acquisition"]["seeds"],
        result["Q1"]["acquisition"]["seeds"],
    )


def test_conflicting_range_controls_fail_before_measurement() -> None:
    """An explicit grid and maximum should remain mutually exclusive."""
    exp: Any = _Experiment(targets=("Q0",))
    with pytest.raises(ValueError, match="Specify only one"):
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4],
            max_n_cliffords=8,
            n_trials=2,
            n_bootstrap=0,
            plot=False,
            save_image=False,
        )
    assert exp.measurement_service.calls == []


def test_summary_log_is_compact_and_omits_bootstrap_and_grid_debug_sections(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only the requested five-line user-facing IRB summary should be logged."""
    exp: Any = _Experiment(targets=("Q0",))
    caplog.set_level(
        "INFO",
        logger="qubex.contrib.experiment.paired_interleaved_randomized_benchmarking",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        paired_interleaved_randomized_benchmarking(
            exp,
            "Q0",
            interleaved_clifford="X90",
            n_cliffords_range=[0, 1, 2, 4, 8, 16],
            n_trials=4,
            sequence_seed=53,
            n_bootstrap=6,
            plot=False,
            save_image=False,
        )

    assert "Paired IRB: Q0" in caplog.text
    assert "p_ref =" in caplog.text
    assert "p_irb =" in caplog.text
    assert "Gate fidelity =" in caplog.text
    assert "95% bootstrap CI =" in caplog.text
    assert "Bootstrap:" not in caplog.text
    assert "Grid selection:" not in caplog.text
