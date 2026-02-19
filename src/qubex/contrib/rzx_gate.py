"""Contributed helpers for RZX gate construction and characterization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from tqdm import tqdm

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import CrossResonance, PulseSchedule, Waveform
from qubex.style import COLORS
from qubex.typing import TargetMap


def rzx(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    angle: float,
    *,
    cr_duration: float | None = None,
    cr_ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cr_beta: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    cancel_beta: float | None = None,
    rotary_amplitude: float | None = None,
    echo: bool | None = None,
    x180: TargetMap[Waveform] | Waveform | None = None,
    x180_margin: float | None = None,
) -> PulseSchedule:
    """
    Return an RZX pulse schedule for a given angle.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse construction.
    control_qubit
        Control qubit label.
    target_qubit
        Target qubit label.
    angle
        Target RZX rotation angle in radians.

    Returns
    -------
    PulseSchedule
        Constructed RZX pulse schedule.
    """
    if echo is None:
        echo = True
    if x180_margin is None:
        x180_margin = 0.0

    reference_angle = np.pi / 2
    coeff_value = angle / reference_angle
    cr_label = f"{control_qubit}-{target_qubit}"
    cr_param = exp.ctx.calib_note.get_cr_param(
        cr_label,
        valid_days=exp.ctx.calibration_valid_days,
    )
    if cr_param is None:
        raise ValueError(f"CR parameters for {cr_label} are not stored.")

    if x180 is None:
        pi_pulse = exp.pulse.x180(control_qubit)
    elif isinstance(x180, Waveform):
        pi_pulse = x180
    else:
        pi_pulse = x180[control_qubit]

    if cr_amplitude is None:
        cr_amplitude = cr_param["cr_amplitude"] * coeff_value
    if cr_duration is None:
        cr_duration = cr_param["duration"]
    if cr_ramptime is None:
        cr_ramptime = cr_param["ramptime"]
    if cr_phase is None:
        cr_phase = cr_param["cr_phase"]
    if cr_beta is None:
        cr_beta = cr_param["cr_beta"]
    if cancel_amplitude is None:
        cancel_amplitude = cr_param["cancel_amplitude"] * coeff_value
    if cancel_phase is None:
        cancel_phase = cr_param["cancel_phase"]
    if cancel_beta is None:
        cancel_beta = cr_param["cancel_beta"]
    if rotary_amplitude is None:
        rotary_amplitude = cr_param["rotary_amplitude"]

    cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude

    return CrossResonance(
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        cr_amplitude=cr_amplitude,
        cr_duration=cr_duration,
        cr_ramptime=cr_ramptime,
        cr_phase=cr_phase,
        cr_beta=cr_beta,
        cancel_amplitude=np.abs(cancel_pulse),
        cancel_phase=np.angle(cancel_pulse),
        cancel_beta=cancel_beta,
        echo=echo,
        pi_pulse=pi_pulse,
        pi_margin=x180_margin,
    )


def rzx_gate_property(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    angle_arr: NDArray | None = None,
    measurement_times: int | None = None,
) -> Result:
    """
    Estimate RZX gate properties for a qubit pair.

    Parameters
    ----------
    exp
        Experiment instance to use for tomography.
    control_qubit
        Control qubit label.
    target_qubit
        Target qubit label.
    angle_arr
        Sweep angles in radians.
    measurement_times
        Number of repeated tomography measurements per angle.

    Returns
    -------
    Result
        Result containing angle sweep statistics and figure.
    """
    if measurement_times is None:
        measurement_times = 10
    if angle_arr is None:
        angle_arr = np.linspace(np.pi / 18, 4 * np.pi / 9, 8)

    rad_to_deg = 180 / np.pi

    def cartesian_to_spherical(
        x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        radius = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x) * rad_to_deg
        phi = np.arccos(z / radius) * rad_to_deg if radius != 0 else 0
        return radius, theta, phi

    result_rzx_angle = []
    for angle in tqdm(angle_arr, leave=False):
        results = []
        for _ in tqdm(range(measurement_times), leave=False):
            result = exp.measurement_service.state_tomography(
                rzx(
                    exp=exp,
                    control_qubit=control_qubit,
                    target_qubit=target_qubit,
                    angle=angle,
                )
            )
            x, y, z = result[target_qubit]
            _radius, _theta, phi = cartesian_to_spherical(x, y, z)
            results.append(phi)
        result_array = np.array(results)
        mean = np.mean(result_array)
        std = np.std(result_array)
        result_rzx_angle.append([angle, mean, std])

    fig = viz.make_figure()
    fig.add_trace(
        go.Scatter(
            x=np.array(result_rzx_angle).T[0] * rad_to_deg,
            y=np.array(result_rzx_angle).T[1],
            marker=dict(color=COLORS[1]),
            error_y=dict(
                type="data",
                array=np.array(result_rzx_angle).T[2],
                color=COLORS[1],
            ),
            name="Measured",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 90],
            y=[0, 90],
            mode="lines",
            line=dict(color=COLORS[0], dash="dash"),
            name="Ideal",
        ),
    )
    fig.update_layout(
        title=f"Sweep result : {control_qubit}-{target_qubit}",
        xaxis=dict(
            title="Angle (deg)",
            range=(0, 90),
            tickvals=angle_arr * rad_to_deg,
            dtick=5,
            gridcolor="gray",
            gridwidth=3,
            griddash="dot",
        ),
        yaxis=dict(
            title="Z Angle (deg)",
            range=(0, 90),
            tickvals=angle_arr * rad_to_deg,
            dtick=5,
            gridcolor="gray",
            gridwidth=3,
            griddash="dot",
        ),
    )
    fig.show()

    return Result(data={"result_rzx_angle": result_rzx_angle, "fig": fig})
