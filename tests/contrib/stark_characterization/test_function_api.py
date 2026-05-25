"""Tests for functional APIs in `qubex.contrib.experiment.stark_characterization`."""

from __future__ import annotations

from qubex.contrib import (
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


def test_all_stark_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then all stark helpers are available."""
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
