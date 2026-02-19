"""Contributed helpers for cross-resonance crosstalk characterization."""

from __future__ import annotations

from collections import defaultdict
from typing import no_type_check

import numpy as np
import plotly.graph_objects as go
import qctrlvisualizer as qcv
from numpy.typing import ArrayLike

import qubex.visualization as viz
from qubex.analysis import fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_CR_RAMPTIME,
    DEFAULT_CR_TIME_RANGE,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.experiment.models import Result
from qubex.pulse import (
    CrossResonance,
    MultiDerivativeCrossResonance,
    PulseSchedule,
    RampType,
    Waveform,
)
from qubex.typing import TargetMap


def measure_cr_crosstalk(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    spectator_qubits: list[str],
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cr_betas: dict[int, float] | None = None,
    cr_power: int | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    cancel_betas: dict[int, float] | None = None,
    cancel_power: int | None = None,
    echo: bool | None = None,
    control_state: str | None = None,
    x90: TargetMap[Waveform] | None = None,
    x180: TargetMap[Waveform] | None = None,
    ramp_type: RampType | None = None,
    x180_margin: float | None = None,
    shots: int | None = None,
    interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot: bool | None = None,
) -> Result:
    """
    Measure CR crosstalk dynamics for target and spectator qubits.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurement.
    control_qubit
        Control qubit label of the CR drive.
    target_qubit
        Target qubit label of the CR drive.
    spectator_qubits
        Spectator qubits measured during tomography.

    Returns
    -------
    Result
        Result containing Bloch trajectories and fitted rotations.
    """
    if echo is None:
        echo = False
    if control_state is None:
        control_state = "0"
    if ramp_type is None:
        ramp_type = "RaisedCosine"
    if shots is None:
        shots = DEFAULT_SHOTS
    if interval is None:
        interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if plot is None:
        plot = True
    cr_label = f"{control_qubit}-{target_qubit}"
    if time_range is None:
        time_range = np.array(DEFAULT_CR_TIME_RANGE, dtype=float)
    else:
        time_range = np.array(time_range, dtype=float)
    if ramptime is None:
        ramptime = DEFAULT_CR_RAMPTIME
    if cr_amplitude is None:
        cr_amplitude = 1.0
    if cr_phase is None:
        cr_phase = 0.0
    if cr_betas is None:
        cr_betas = {}
    if cr_power is None:
        cr_power = 2
    if cancel_amplitude is None:
        cancel_amplitude = 0.0
    if cancel_phase is None:
        cancel_phase = 0.0
    if cancel_betas is None:
        cancel_betas = {}
    if cancel_power is None:
        cancel_power = 2
    if x180_margin is None:
        x180_margin = 0.0
    if x90 is None:
        x90 = {
            control_qubit: exp.pulse.x90(control_qubit),
            target_qubit: exp.pulse.x90(target_qubit),
        }
        x90.update(
            {spectator: exp.pulse.x90(spectator) for spectator in spectator_qubits}
        )

    if x180 is None:
        x180 = {
            control_qubit: exp.pulse.x180(control_qubit),
        }

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(
            qubits=[control_qubit, target_qubit, *spectator_qubits]
        )

    def cr_sequence(targets: list[str], drive_time: float) -> PulseSchedule:
        cr = CrossResonance(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            cr_amplitude=cr_amplitude,
            cr_duration=drive_time + ramptime * 2,
            cr_ramptime=ramptime,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            echo=echo,
            pi_pulse=x180[control_qubit],
            pi_margin=x180_margin,
            ramp_type=ramp_type,
        )
        with PulseSchedule(targets) as ps:
            ps.call(cr)
        return ps

    def multi_derivative_cr_sequence(
        targets: list[str], drive_time: float
    ) -> PulseSchedule:
        cr = MultiDerivativeCrossResonance(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            cr_amplitude=cr_amplitude,
            cr_duration=drive_time + ramptime * 2,
            cr_ramptime=ramptime,
            cr_phase=cr_phase,
            cr_betas=cr_betas,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            cancel_betas=cancel_betas,
            echo=echo,
            pi_pulse=x180[control_qubit],
            pi_margin=x180_margin,
        )
        with PulseSchedule(targets) as ps:
            ps.call(cr)
        return ps

    if ramp_type == "MultiDerivativeSintegral":
        sequence_func = multi_derivative_cr_sequence
    else:
        sequence_func = cr_sequence

    control_states = []
    target_states = []
    spectators_states = defaultdict(list)

    with exp.ctx.modified_frequencies(
        frequencies=dict.fromkeys(spectator_qubits, exp.ctx.targets[cr_label].frequency)
    ):
        for drive_time in time_range:
            result = exp.measurement_service.state_tomography(
                sequence=sequence_func(
                    targets=[control_qubit, target_qubit, *spectator_qubits],
                    drive_time=drive_time,
                ),
                x90=x90,
                initial_state={control_qubit: control_state},
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=False,
                plot=False,
            )
            control_states.append(np.array(result[control_qubit]))
            target_states.append(np.array(result[target_qubit]))

            for spectator in spectator_qubits:
                spectators_states[spectator].append(np.array(result[spectator]))

    control_states = np.array(control_states)
    target_states = np.array(target_states)
    spectators_states = {
        spectator: np.array(states) for spectator, states in spectators_states.items()
    }

    effective_drive_range = time_range + ramptime

    fit_result = fitting.fit_rotation(
        effective_drive_range,
        target_states,
        plot=False,
        title=f"Target qubit dynamics of {cr_label} : |{control_state}〉",
        xlabel="Drive time (ns)",
        ylabel=f"Target qubit : {target_qubit}",
    )

    spectators_fit_result = {}
    for spectator, states in spectators_states.items():
        fit_spectator = fitting.fit_rotation(
            effective_drive_range,
            states,
            plot=False,
            title=f"Spectator qubit dynamics of {cr_label} : |{control_state}〉",
            xlabel="Drive time (ns)",
            ylabel=f"Spectator qubit : {spectator}",
        )
        spectators_fit_result[spectator] = fit_spectator

    if plot:
        viz.plot_bloch_vectors(
            effective_drive_range,
            control_states,
            title=f"Control qubit dynamics of {cr_label} : |{control_state}〉",
            xlabel="Drive time (ns)",
            ylabel=f"Control qubit : {control_qubit}",
        )
        qcv.display_bloch_sphere_from_bloch_vectors(control_states)

        fit_result["fig"].show()
        fit_result["fig3d"].show()
        qcv.display_bloch_sphere_from_bloch_vectors(target_states)

        for spectator, fit_spectator in spectators_fit_result.items():
            fit_spectator["fig"].show()
            fit_spectator["fig3d"].show()
            qcv.display_bloch_sphere_from_bloch_vectors(spectators_states[spectator])

    return Result(
        data={
            "time_range": time_range,
            "effective_drive_range": effective_drive_range,
            "control_states": control_states,
            "target_states": target_states,
            "spectators_states": spectators_states,
            "fit_result": fit_result,
            "spectators_fit_result": spectators_fit_result,
            "cr_amplitude": cr_amplitude,
            "ramptime": ramptime,
        }
    )


@no_type_check
def cr_crosstalk_hamiltonian_tomography(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    spectator_qubits: list[str] | None = None,
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cr_betas: dict[int, float] | None = None,
    cr_power: int | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    cancel_betas: dict[int, float] | None = None,
    cancel_power: int | None = None,
    ramp_type: RampType | None = None,
    x90: TargetMap[Waveform] | None = None,
    x180_margin: float | None = None,
    shots: int | None = None,
    interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    plot: bool | None = None,
) -> Result:
    """Perform CR crosstalk Hamiltonian tomography."""
    if ramp_type is None:
        ramp_type = "RaisedCosine"
    if shots is None:
        shots = CALIBRATION_SHOTS
    if interval is None:
        interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if plot is None:
        plot = True
    cr_label = f"{control_qubit}-{target_qubit}"

    if spectator_qubits is None:
        spectator_qubits = []
        for spectator in exp.ctx.get_spectators(control_qubit):
            if (
                spectator.label in exp.ctx.qubit_labels
                and spectator.label != target_qubit
            ):
                spectator_qubits.append(spectator.label)

    if cr_amplitude is None:
        cr_amplitude = 1.0

    if ramptime is None:
        ramptime = exp.calibration_service._ramptime(control_qubit, target_qubit)  # noqa: SLF001

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(
            qubits=[control_qubit, target_qubit, *spectator_qubits]
        )

    result_0 = measure_cr_crosstalk(
        exp=exp,
        time_range=time_range,
        ramptime=ramptime,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        spectator_qubits=spectator_qubits,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cr_betas=cr_betas,
        cr_power=cr_power,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        cancel_betas=cancel_betas,
        cancel_power=cancel_power,
        echo=False,
        control_state="0",
        x90=x90,
        ramp_type=ramp_type,
        x180_margin=x180_margin,
        shots=shots,
        interval=interval,
        reset_awg_and_capunits=False,
        plot=False,
    )

    result_1 = measure_cr_crosstalk(
        exp=exp,
        time_range=time_range,
        ramptime=ramptime,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        spectator_qubits=spectator_qubits,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cr_betas=cr_betas,
        cr_power=cr_power,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        cancel_betas=cancel_betas,
        cancel_power=cancel_power,
        echo=False,
        control_state="1",
        x90=x90,
        ramp_type=ramp_type,
        x180_margin=x180_margin,
        shots=shots,
        interval=interval,
        reset_awg_and_capunits=False,
        plot=False,
    )

    Omega_0 = result_0["fit_result"]["Omega"]
    Omega_1 = result_1["fit_result"]["Omega"]
    Omega = np.concatenate([0.5 * (Omega_0 + Omega_1), 0.5 * (Omega_0 - Omega_1)])
    coeffs = dict(
        zip(
            ["IX", "IY", "IZ", "ZX", "ZY", "ZZ"],
            Omega / (2 * np.pi),  # GHz
            strict=True,
        )
    )

    f_control = exp.ctx.qubits[control_qubit].frequency
    f_target = exp.ctx.qubits[target_qubit].frequency
    f_delta = f_control - f_target

    xt_rotation = coeffs["IX"] + 1j * coeffs["IY"]
    xt_rotation_amplitude = np.abs(xt_rotation)
    xt_rotation_amplitude_hw = exp.pulse.calc_control_amplitude(
        target=target_qubit,
        rabi_rate=xt_rotation_amplitude,
    )
    xt_rotation_phase = np.angle(xt_rotation)
    xt_rotation_phase_deg = np.angle(xt_rotation, deg=True)

    cr_rotation = coeffs["ZX"] + 1j * coeffs["ZY"]
    cr_rotation_amplitude = np.abs(cr_rotation)
    cr_rotation_amplitude_hw = exp.pulse.calc_control_amplitude(
        target=target_qubit,
        rabi_rate=cr_rotation_amplitude,
    )
    cr_rotation_phase = np.angle(cr_rotation)
    cr_rotation_phase_deg = np.angle(cr_rotation, deg=True)
    zx90_duration = 1 / (4 * cr_rotation_amplitude)

    cr_rabi_rate = exp.pulse.calc_rabi_rate(control_qubit, cr_amplitude)

    fig_c = viz.make_figure()
    fig_c.set_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
    )
    fig_c_0 = viz.make_bloch_vectors_figure(
        result_0["effective_drive_range"],
        result_0["control_states"],
    )
    fig_c_1 = viz.make_bloch_vectors_figure(
        result_1["effective_drive_range"],
        result_1["control_states"],
    )
    for data in fig_c_0.data:
        data: go.Scatter
        fig_c.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode=data.mode,
                line=data.line,
                marker=data.marker,
                name=data.name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    for data in fig_c_1.data:
        data: go.Scatter
        fig_c.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode=data.mode,
                line=data.line,
                marker=data.marker,
                name=data.name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig_c.update_xaxes(title_text="Drive time (ns)", row=2, col=1)
    fig_c.update_yaxes(title_text="Control : |0〉", range=[-1.1, 1.1], row=1, col=1)
    fig_c.update_yaxes(title_text="Control : |1〉", range=[-1.1, 1.1], row=2, col=1)
    fig_c.update_layout(
        title=dict(
            text=f"Control qubit dynamics : {cr_label}",
            subtitle=dict(
                text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
            ),
        ),
        height=400,
        width=600,
        showlegend=True,
        margin=dict(t=90),
    )

    fig_t = viz.make_figure()
    fig_t.set_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
    )
    fig_t_0 = result_0["fit_result"]["fig"]
    fig_t_1 = result_1["fit_result"]["fig"]
    for data in fig_t_0.data:
        data: go.Scatter
        fig_t.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode=data.mode,
                line=data.line,
                marker=data.marker,
                name=data.name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    for data in fig_t_1.data:
        data: go.Scatter
        fig_t.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                mode=data.mode,
                line=data.line,
                marker=data.marker,
                name=data.name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig_t.update_xaxes(title_text="Drive time (ns)", row=2, col=1)
    fig_t.update_yaxes(title_text="Control : |0〉", range=[-1.1, 1.1], row=1, col=1)
    fig_t.update_yaxes(title_text="Control : |1〉", range=[-1.1, 1.1], row=2, col=1)
    fig_t.update_layout(
        title=dict(
            text=f"Target qubit dynamics : {cr_label}",
            subtitle=dict(
                text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
            ),
        ),
        height=400,
        width=600,
        showlegend=True,
        margin=dict(t=90),
    )

    fig_t_3d = viz.make_figure()
    fig_t_3d.set_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Control : |0〉", "Control : |1〉"],
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        horizontal_spacing=0.01,
    )
    fig_t_3d_0 = result_0["fit_result"]["fig3d"]
    fig_t_3d_1 = result_1["fit_result"]["fig3d"]
    for data in fig_t_3d_0.data:
        fig_t_3d.add_trace(data, row=1, col=1)
    for data in fig_t_3d_1.data:
        fig_t_3d.add_trace(data, row=1, col=2)
    fig_t_3d.update_annotations(dict(font=dict(size=13), yshift=-20))
    fig_t_3d.update_layout(
        title=dict(
            text=f"Target qubit dynamics : {cr_label}",
            subtitle=dict(
                text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
            ),
        ),
        height=400,
        width=600,
        showlegend=False,
        margin=dict(t=90, b=10, l=10, r=10),
    )

    spectators_fit_results_0 = result_0["spectators_fit_result"]
    spectators_fit_results_1 = result_1["spectators_fit_result"]
    figs_s = {}
    figs_s_3d = {}
    for label in spectators_fit_results_0:
        f_delta = (
            exp.ctx.qubits[control_qubit].frequency
            - exp.ctx.qubits[target_qubit].frequency
        )
        f_delta_st = (
            exp.ctx.qubits[label].frequency - exp.ctx.qubits[target_qubit].frequency
        )

        fig_s_0: go.Figure = spectators_fit_results_0[label]["fig"]
        fig_s_1: go.Figure = spectators_fit_results_1[label]["fig"]

        fig_s = viz.make_figure()
        fig_s.set_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
        )
        for data in fig_s_0.data:
            data: go.Scatter
            fig_s.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        for data in fig_s_1.data:
            data: go.Scatter
            fig_s.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        fig_s.update_xaxes(title_text="Drive time (ns)", row=2, col=1)
        fig_s.update_yaxes(title_text="control : |0〉", range=[-1.1, 1.1], row=1, col=1)
        fig_s.update_yaxes(title_text="control : |1〉", range=[-1.1, 1.1], row=2, col=1)
        fig_s.update_layout(
            title=dict(
                text=f"Spectator qubit dynamics : {label} in {cr_label}",
                subtitle=dict(
                    text=f"Δ = {f_delta * 1e3:.0f} MHz ,Δ_st = {f_delta_st * 1e3:.0f} MHz  Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                ),
            ),
            height=400,
            width=800,
            showlegend=True,
            margin=dict(t=90),
        )

        fig_s_3d_0 = spectators_fit_results_0[label]["fig3d"]
        fig_s_3d_1 = spectators_fit_results_1[label]["fig3d"]

        fig_s_3d = viz.make_figure()
        fig_s_3d.set_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Control |0〉", "Control |1〉"],
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            horizontal_spacing=0.01,
        )

        @no_type_check
        def add_trace_with_color(
            fig: go.Figure,
            incoming_fig: go.Figure,
            row: int,
            col: int,
            name: str | None = None,
        ) -> None:
            for ddx, data in enumerate(incoming_fig.data):
                if ddx % 2 == 0:
                    name_suffix = "data"
                else:
                    name_suffix = "fit"
                fig.add_trace(data, row=row, col=col)
                if not isinstance(data, go.Surface):
                    if data.mode == "markers":
                        fig.data[-1].marker.color = viz.COLORS[ddx]
                    if data.mode == "lines":
                        fig.data[-1].line.color = viz.COLORS[ddx]
                    if name is not None:
                        fig.data[-1].name = f"{name_suffix} ({name})"
                        fig.data[-1].showlegend = True

        add_trace_with_color(fig=fig_s_3d, incoming_fig=fig_s_3d_0, row=1, col=1)
        add_trace_with_color(
            fig=fig_s_3d,
            incoming_fig=fig_s_3d_1,
            row=1,
            col=2,
            name="raw",
        )

        fig_s_3d.update_layout(
            title=dict(
                text=f"Spectator qubit dynamics : {label} of {cr_label}",
                subtitle=dict(
                    text=f"Δ = {f_delta_st * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                ),
            ),
            height=400,
            width=600,
            showlegend=False,
            margin=dict(t=90, b=10, l=10, r=10),
        )

        figs_s[label] = fig_s
        figs_s_3d[label] = fig_s_3d

    fig_c.show()
    fig_t.show()
    fig_t_3d.show()
    for fig_s_ in figs_s.values():
        fig_s_.show()
    for fig_s_3d in figs_s_3d.values():
        fig_s_3d.show()

    print("Qubit frequencies:")
    print(f"  ω_c ({control_qubit}) : {f_control * 1e3:.3f} MHz")
    print(f"  ω_t ({target_qubit}) : {f_target * 1e3:.3f} MHz")
    print(f"  Δ ({cr_label}) : {f_delta * 1e3:.3f} MHz")
    print("CR drive:")
    print(f"  Ω : {cr_rabi_rate * 1e3:.3f} MHz ({cr_amplitude:.4f})")
    print("Rotation rates:")
    for key, value in coeffs.items():
        print(f"  {key} : {value * 1e3:+.4f} MHz")
    print("XT (crosstalk) rotation:")
    print(
        f"  rate  : {xt_rotation_amplitude * 1e3:.4f} MHz ({xt_rotation_amplitude_hw:.6f})"
    )
    print(f"  phase : {xt_rotation_phase:.4f} rad ({xt_rotation_phase_deg:.1f} deg)")
    print("CR (cross-resonance) rotation:")
    print(
        f"  rate  : {cr_rotation_amplitude * 1e3:.4f} MHz ({cr_rotation_amplitude_hw:.6f})"
    )
    print(f"  phase : {cr_rotation_phase:.4f} rad ({cr_rotation_phase_deg:.1f} deg)")
    print(f"Estimated ZX90 gate length : {zx90_duration:.1f} ns")

    coeffs_ = {}
    for spectator in spectator_qubits:
        print(f" Spectator qubit: {spectator}")
        f_s = exp.ctx.qubits[spectator].frequency
        print("")
        print(f"  ω_s ({spectator}) : {f_s * 1e3:.3f} MHz")
        print(f"  ω_t ({target_qubit}) : {f_target * 1e3:.3f} MHz")
        print(f"  Δ_st ({spectator}-{target_qubit}) : {(f_s - f_target) * 1e3:.3f} MHz")

        spectator_Omega_0 = result_0["spectators_fit_result"][spectator]["Omega"]
        spectator_Omega_1 = result_1["spectators_fit_result"][spectator]["Omega"]
        spectator_Omega = np.concatenate(
            [
                0.5 * (spectator_Omega_0 + spectator_Omega_1),
                0.5 * (spectator_Omega_0 - spectator_Omega_1),
            ]
        )
        spectator_coeffs = dict(
            zip(
                ["IX", "IY", "IZ", "ZX", "ZY", "ZZ"],
                spectator_Omega / (2 * np.pi),  # GHz
                strict=True,
            )
        )

        coeffs_[spectator] = spectator_coeffs
        print("")
        for key, value in spectator_coeffs.items():
            print(f"  {key} : {value * 1e3:+.4f} MHz")
        print("")
        print(
            f"  |IX + 1j * IY| : {np.abs(spectator_coeffs['IX'] + 1j * spectator_coeffs['IY']) * 1e3:.4f} MHz"
        )
        print(
            f"  |ZX + 1j * ZY| : {np.abs(spectator_coeffs['ZX'] + 1j * spectator_coeffs['ZY']) * 1e3:.4f} MHz"
        )
        print(
            f"  √ (|IX + 1j * IY|² + IZ²) : {np.sqrt(spectator_coeffs['IX'] ** 2 + spectator_coeffs['IY'] ** 2 + spectator_coeffs['IZ'] ** 2) * 1e3:.4f} MHz"
        )
        print(
            f"  √ (|ZX + 1j * ZY|² + ZZ²) : {np.sqrt(spectator_coeffs['ZX'] ** 2 + spectator_coeffs['ZY'] ** 2 + spectator_coeffs['ZZ'] ** 2) * 1e3:.4f} MHz"
        )
        print(f" r2 (control |0〉): {spectators_fit_results_0[spectator]['r2']:.4f}")
        print(f" r2 (control |1〉): {spectators_fit_results_1[spectator]['r2']:.4f}")
        print("")

    coeffs_[target_qubit] = coeffs
    return Result(
        data={
            "Omega": Omega,
            "coeffs": coeffs_,
            "cr_rotation_amplitude": cr_rotation_amplitude,
            "cr_rotation_amplitude_hw": cr_rotation_amplitude_hw,
            "cr_rotation_phase": cr_rotation_phase,
            "xt_rotation_amplitude": xt_rotation_amplitude,
            "xt_rotation_amplitude_hw": xt_rotation_amplitude_hw,
            "xt_rotation_phase": xt_rotation_phase,
            "cr_drive_amplitude": cr_rabi_rate,
            "cr_drive_amplitude_hw": cr_amplitude,
            "zx90_duration": zx90_duration,
            "result_0": result_0,
            "result_1": result_1,
            "fig_c": fig_c,
            "fig_t": fig_t,
            "figs_s": figs_s,
        }
    )
