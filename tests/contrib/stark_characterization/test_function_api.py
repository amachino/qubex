"""Tests for functional APIs in `qubex.contrib.experiment.stark_characterization`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.contrib import (
    ac_stark_shift_spectroscopy,
    ac_stark_shift_spectroscopy_over_time,
    calibrate_stark_default_pulse,
    calibrate_stark_drag_amplitude,
    calibrate_stark_drag_beta,
    calibrate_stark_drag_hpi_pulse,
    calibrate_stark_drag_pi_pulse,
    calibrate_stark_hpi_pulse,
    calibrate_stark_pi_pulse,
    calibrate_stark_zx90,
    insitu_target,
    make_insitu_channel,
    make_stark_channel,
    make_stark_cr_channel,
    obtain_cr_params_under_stark,
    ramsey_experiment_under_stark,
    stark_bell_state_sequence,
    stark_bell_state_tomography,
    stark_chevron_pattern,
    stark_cnot,
    stark_cr_hamiltonian_tomography,
    stark_cr_target,
    stark_interleaved_purity_benchmarking,
    stark_interleaved_randomized_benchmarking,
    stark_interleaved_randomized_benchmarking_2q,
    stark_ipurity_experiment,
    stark_irb_experiment,
    stark_measure_cr_dynamics,
    stark_obtain_cr_params,
    stark_purity_experiment_1q,
    stark_purity_sequence_1q,
    stark_rabi_experiment,
    stark_rabi_sequence,
    stark_ramsey_experiment,
    stark_ramsey_sequence_under_stark,
    stark_rb_experiment_1q,
    stark_rb_experiment_2q,
    stark_rb_sequence_1q,
    stark_rb_sequence_2q,
    stark_repeat_sequence,
    stark_repeat_sequence_sample,
    stark_t1_experiment,
    stark_t1_sequence_under_stark,
    stark_t2_sequence_under_stark,
    stark_target,
    stark_update_cr_params,
    stark_zx90,
    t1_experiment_under_stark,
    t2_experiment_under_stark,
)
from qubex.contrib.experiment import stark_characterization as sc
from qubex.experiment import Experiment
from qubex.experiment.models import Result


class _UtilStub:
    def discretize_time_range(
        self,
        values: np.ndarray,
        sampling_period: float | None = None,
    ) -> np.ndarray:
        return values


class _ExperimentStub:
    def __init__(self) -> None:
        self.ctx = SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=1.0),
            util=_UtilStub(),
        )


def test_all_stark_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then all stark helpers are available."""
    assert callable(ac_stark_shift_spectroscopy)
    assert callable(ac_stark_shift_spectroscopy_over_time)
    assert callable(stark_t1_experiment)
    assert callable(stark_ramsey_experiment)
    assert callable(stark_target)
    assert callable(insitu_target)
    assert callable(make_stark_channel)
    assert callable(make_insitu_channel)
    assert callable(stark_cr_target)
    assert callable(make_stark_cr_channel)
    assert callable(calibrate_stark_default_pulse)
    assert callable(calibrate_stark_hpi_pulse)
    assert callable(calibrate_stark_pi_pulse)
    assert callable(calibrate_stark_zx90)
    assert callable(calibrate_stark_drag_amplitude)
    assert callable(calibrate_stark_drag_beta)
    assert callable(calibrate_stark_drag_hpi_pulse)
    assert callable(calibrate_stark_drag_pi_pulse)
    assert callable(t1_experiment_under_stark)
    assert callable(t2_experiment_under_stark)
    assert callable(ramsey_experiment_under_stark)
    assert callable(stark_rabi_experiment)
    assert callable(stark_rabi_sequence)
    assert callable(stark_repeat_sequence)
    assert callable(stark_repeat_sequence_sample)
    assert callable(stark_chevron_pattern)
    assert callable(stark_zx90)
    assert callable(stark_cnot)
    assert callable(stark_bell_state_sequence)
    assert callable(stark_bell_state_tomography)
    assert callable(stark_rb_experiment_1q)
    assert callable(stark_rb_sequence_1q)
    assert callable(stark_rb_experiment_2q)
    assert callable(stark_rb_sequence_2q)
    assert callable(stark_purity_experiment_1q)
    assert callable(stark_purity_sequence_1q)
    assert callable(stark_irb_experiment)
    assert callable(stark_ipurity_experiment)
    assert callable(stark_interleaved_randomized_benchmarking)
    assert callable(stark_interleaved_randomized_benchmarking_2q)
    assert callable(stark_interleaved_purity_benchmarking)
    assert callable(stark_t1_sequence_under_stark)
    assert callable(stark_t2_sequence_under_stark)
    assert callable(stark_ramsey_sequence_under_stark)
    assert callable(stark_measure_cr_dynamics)
    assert callable(stark_cr_hamiltonian_tomography)
    assert callable(stark_update_cr_params)
    assert callable(obtain_cr_params_under_stark)
    assert callable(stark_obtain_cr_params)


def test_ac_stark_shift_spectroscopy_can_plot_applied_amplitude_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given amplitude axis, when wait times are swept, then P1 is keyed by Stark amplitude."""
    calls: list[dict[str, Any]] = []

    def fake_stark_p1_spectroscopy(
        exp: Experiment,
        target: str,
        **kwargs: Any,
    ) -> Result:
        calls.append(kwargs)
        amplitude = np.asarray(kwargs["stark_amplitude_range"], dtype=float)
        wait_time = float(kwargs["wait_time"])
        return Result(data={"p1": amplitude + wait_time / 1000})

    monkeypatch.setattr(sc, "stark_p1_spectroscopy", fake_stark_p1_spectroscopy)

    result = sc.ac_stark_shift_spectroscopy(
        cast(Experiment, _ExperimentStub()),
        "Q00",
        stark_detuning=0.15,
        stark_amplitude_range=[0.0, 0.1, 0.2],
        stark_shift_model="amplitude",
        wait_time_range=[10, 20],
        n_shots=1,
        plot=False,
    )

    np.testing.assert_allclose(result.data["x_axis_range"], [0.0, 0.1, 0.2])
    assert result.data["x_axis"] == "stark_amplitude"
    assert np.asarray(result.data["p1"]).shape == (2, 3)
    assert result.figure is not None
    heatmap_trace = cast(Any, result.figure.data[0])
    assert heatmap_trace.type == "heatmap"
    assert result.figure.layout.xaxis.title.text == "Stark amplitude (GHz)"
    assert [call["wait_time"] for call in calls] == [10, 20]


def test_ac_stark_shift_spectroscopy_over_time_uses_line_plot_for_one_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given one repeated measurement, when measured over time, then P1 is a line plot."""

    def fake_stark_p1_spectroscopy(
        exp: Experiment,
        target: str,
        **kwargs: Any,
    ) -> Result:
        return Result(data={"p1": np.asarray(kwargs["stark_amplitude_range"])})

    monkeypatch.setattr(sc, "stark_p1_spectroscopy", fake_stark_p1_spectroscopy)

    result = sc.ac_stark_shift_spectroscopy_over_time(
        cast(Experiment, _ExperimentStub()),
        "Q00",
        stark_detuning=0.15,
        stark_amplitude_range=[0.0, 0.1, 0.2],
        stark_shift_model="amplitude",
        wait_time=100,
        n_iterations=1,
        n_shots=1,
        plot=False,
    )

    assert result.data["n_iterations"] == 1
    assert np.asarray(result.data["p1"]).shape == (1, 3)
    assert result.figure is not None
    scatter_trace = cast(Any, result.figure.data[0])
    assert scatter_trace.type == "scatter"
    assert result.figure.layout.yaxis.title.text == "P1"


def test_ac_stark_shift_spectroscopy_over_time_uses_heatmap_for_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given repeated measurements, when repeated, then rows are stacked by iteration."""
    calls: list[dict[str, Any]] = []

    def fake_stark_p1_spectroscopy(
        exp: Experiment,
        target: str,
        **kwargs: Any,
    ) -> Result:
        calls.append(kwargs)
        iteration = len(calls) - 1
        amplitude = np.asarray(kwargs["stark_amplitude_range"], dtype=float)
        return Result(data={"p1": amplitude + 0.01 * iteration})

    monkeypatch.setattr(sc, "stark_p1_spectroscopy", fake_stark_p1_spectroscopy)

    result = sc.ac_stark_shift_spectroscopy_over_time(
        cast(Experiment, _ExperimentStub()),
        "Q00",
        stark_detuning=0.15,
        stark_amplitude_range=[0.0, 0.1],
        stark_shift_model="amplitude",
        wait_time=100,
        n_iterations=2,
        n_shots=1,
        plot=False,
    )

    assert np.asarray(result.data["p1"]).shape == (2, 2)
    np.testing.assert_allclose(result.data["iteration_range"], [1, 2])
    assert len(result.data["elapsed_time_s"]) == 2
    assert len(calls) == 2
    assert result.figure is not None
    heatmap_trace = cast(Any, result.figure.data[0])
    assert heatmap_trace.type == "heatmap"
    np.testing.assert_allclose(heatmap_trace.y, [1, 2])
    assert result.figure.layout.yaxis.title.text == "Iteration"


def test_ac_stark_amplitude_axis_requires_single_detuning() -> None:
    """Given multiple Stark detunings, when amplitude axis is requested, then raise."""
    with pytest.raises(ValueError, match="requires a single Stark detuning"):
        sc.ac_stark_shift_spectroscopy_over_time(
            cast(Experiment, _ExperimentStub()),
            "Q00",
            stark_detuning=[-0.15, 0.15],
            stark_shift_model="amplitude",
            wait_time=100,
            n_iterations=1,
            n_shots=1,
            plot=False,
        )
