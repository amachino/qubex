from ..experiment import Experiment
from ... import PulseSchedule
from ...pulse import FlatTop
from ...style import COLORS
from typing import Collection
from tqdm import tqdm

import numpy as np
from scipy.stats import ttest_ind
import plotly.graph_objects as go

def estimate_crosstalk(
        exp: Experiment,
        control_qubit: str,
        target_qubit: str, 
        spectators:  Collection[str] | str,
        measurement_times: int = 20,
        cr_duration: float | None = None,
        plot: bool = True,
    ):
    """Estimtes the crosstalk between diffrent qubit channels."""

    if isinstance(spectators, str):
        spectators = [spectators]
    else:
        spectators = list(spectators)

    for spectator in spectators:
        cr_cross_check_seq, duration = _cr_cross_check_seq(exp,control_qubit=control_qubit,target_qubit=target_qubit,spectator_qubit=spectator,duration=cr_duration,reference=False,)
        cr_cross_check_ref_seq, _ = _cr_cross_check_seq(exp,control_qubit=control_qubit,target_qubit=target_qubit,spectator_qubit=spectator,duration=cr_duration,reference=True,)

        if plot:
            cr_cross_check_seq.plot()
            cr_cross_check_ref_seq.plot()

        result_mea=[]
        result_ref_mea=[]

        for i in tqdm(range(measurement_times),leave=False):
            result_cross_check = exp.state_tomography(cr_cross_check_seq)
            r,theta,phi = cartesian_to_spherical(result_cross_check[spectator][0],result_cross_check[spectator][1],result_cross_check[spectator][2])
            result_mea.append([r,theta,phi])
            
            result_cross_check_ref = exp.state_tomography(cr_cross_check_ref_seq)
            r,theta,phi = cartesian_to_spherical(result_cross_check_ref[spectator][0],result_cross_check_ref[spectator][1],result_cross_check_ref[spectator][2])
            result_ref_mea.append([r,theta,phi])            

        data1 = np.array(result_mea)[:,1] 
        data2 = np.array(result_ref_mea)[:,1]  

        if np.max(data1)-np.min(data1) > 180:
            for id in range(len(data1)):
                if data1[id] < 0:
                    data1[id] += 360
            print("Data with CR is shifted range of angles")

        if np.max(data2)-np.min(data2) > 180:
            for id in range(len(data2)):
                if data2[id] < 0:
                    data2[id] += 360
            print("Data without CR is shifted range of angles")
        
        if np.abs(np.mean(data1)-np.mean(data2)) > 180:
            if np.mean(data1) < np.mean(data2):
                data1 += 360
            else:
                data2 += 360
            print("Shifted range of angles")

        t_stat, p_value = ttest_ind(data1, data2)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data1-np.mean(data2), name="CR-pulse on",  marker=dict(color=COLORS[0]), opacity=0.6, xbins=dict(size=2)))
        fig.add_trace(go.Histogram(x=data2-np.mean(data2), name="CR-pulse off", marker=dict(color=COLORS[1]), opacity=0.6, xbins=dict(size=2)))
        fig.add_vline(x=np.mean(data1-np.mean(data2)), line_width=2, line_dash="dash", line_color=COLORS[0], name="CR-pulse on")
        fig.add_vline(x=np.mean(data2-np.mean(data2)), line_width=2, line_dash="dash", line_color=COLORS[1], name="CR-pulse off")
        fig.update_layout(
            title=f"CR control:{control_qubit}, target:{target_qubit}, spectator:{spectator} duration:{duration}ns <br>the result of t-test: p value = {p_value:.4f}",
            xaxis_title="angle difference",
            yaxis_title="times",
            barmode="overlay",
        )
        fig.show()


def _cr_cross_check_seq(
        exp: Experiment,
        control_qubit: str,
        target_qubit: str,
        spectator_qubit: str,
        reference: bool,
        duration: float,
    ) -> PulseSchedule:
    cr_label = f"{control_qubit}-{target_qubit}"
    cr_param = exp.calib_note.get_cr_param(
            cr_label,
            valid_days=exp._calibration_valid_days,
        )
    if cr_param is None:
        raise ValueError("CR parameters are not stored.")
    
    if reference:
        cr_amp = 0
        cr_cancel_amp = 0
    else:
        cr_amp = cr_param["cr_amplitude"]
        cr_cancel_amp = cr_param["cancel_amplitude"]

    if duration is None:
        duration = cr_param["duration"]

    cr_waveform = FlatTop(
        duration=duration,
        amplitude=cr_amp,
        tau=cr_param["ramptime"],
        phase=cr_param["cr_phase"],
        beta=cr_param["cr_beta"],
        type="RaisedCosine",
        )

    cancel_waveform = FlatTop(
        duration=duration,
        amplitude=cr_cancel_amp,
        tau=cr_param["ramptime"],
        phase=cr_param["cancel_phase"],
        beta=cr_param["cancel_beta"],
        type="RaisedCosine",
        )

    with PulseSchedule([cr_label, target_qubit]) as cr:
        cr.add(cr_label, cr_waveform)
        cr.add(target_qubit, cancel_waveform)
    with PulseSchedule([control_qubit, cr_label, target_qubit]) as cross_check:
        cross_check.add(spectator_qubit,exp.x90(spectator_qubit))
        cross_check.barrier()
        cross_check.call(cr)
        cross_check.barrier()
    return cross_check, duration

def cartesian_to_spherical(x, y, z):
    RADTODEG = np.pi/180
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)/RADTODEG
    phi = np.arccos(z / r)/RADTODEG if r != 0 else 0
    return r, theta, phi