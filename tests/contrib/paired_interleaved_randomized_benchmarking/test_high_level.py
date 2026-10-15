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
    assert high_signature.parameters["pairs_per_sweep"].default is None
    assert low_signature.parameters["pairs_per_sweep"].default is None
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
