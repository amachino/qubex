from __future__ import annotations

from typing import Collection

import cma
import numpy as np
import scipy.optimize

from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import (
    CrossResonance,
    Drag,
    Pulse,
    PulseSchedule,
    Waveform,
)
from ...typing import TargetMap
from ..protocol import (
    BaseProtocol,
    BenchmarkingProtocol,
    CalibrationProtocol,
    MeasurementProtocol,
    OptimizationProtocol,
)


class OptimizationMixin(
    BaseProtocol,
    MeasurementProtocol,
    BenchmarkingProtocol,
    CalibrationProtocol,
    OptimizationProtocol,
):
    def optimize_x90(
        self,
        qubit: str,
        *,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform:
        pulse = self.get_drag_hpi_pulse(qubit)
        N = pulse.length
        initial_params = list(pulse.real) + list(pulse.imag)
        es = cma.CMAEvolutionStrategy(
            initial_params,
            sigma0,
            {
                "seed": seed,
                "ftarget": ftarget,
                "timeout": timeout,
                "bounds": [[-1] * 2 * N, [1] * 2 * N],
            },
        )

        def objective_func(params):
            pulse = Pulse(params[:N] + 1j * params[N:])
            result = self.state_tomography(
                {qubit: pulse.repeated(2)},
                x90={qubit: pulse},
            )
            loss = np.linalg.norm(result[qubit] - np.array((0, 0, -1)))
            return loss

        es.optimize(objective_func)
        x = es.result.xbest
        opt_pulse = Pulse(x[:N] + 1j * x[N:])
        return opt_pulse

    def optimize_drag_x90(
        self,
        qubit: str,
        *,
        duration: float = 16,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform:
        param = self.calib_note.get_drag_hpi_param(qubit)
        if param is None:
            raise ValueError("DRAG HPI parameters are not stored.")
        initial_params = [param["amplitude"], param["beta"]]
        es = cma.CMAEvolutionStrategy(
            initial_params,
            sigma0,
            {
                "seed": seed,
                "ftarget": ftarget,
                "timeout": timeout,
                "bounds": [[-1, -1], [1, 1]],
            },
        )

        def objective_func(params):
            pulse = Drag(
                duration=duration,
                amplitude=params[0],
                beta=params[1],
            )
            result = self.state_tomography(
                {qubit: pulse.repeated(2)},
                x90={qubit: pulse},
            )
            loss = np.linalg.norm(result[qubit] - np.array((0, 0, -1)))
            return loss

        es.optimize(objective_func)
        x = es.result.xbest
        opt_pulse = Drag(duration=duration, amplitude=x[0], beta=x[1])
        return opt_pulse

    def optimize_pulse(
        self,
        qubit: str,
        *,
        pulse: Waveform,
        x90: Waveform,
        target_state: tuple[float, float, float],
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform:
        N = pulse.length
        initial_params = list(pulse.real) + list(pulse.imag)
        es = cma.CMAEvolutionStrategy(
            initial_params,
            sigma0,
            {
                "seed": seed,
                "ftarget": ftarget,
                "timeout": timeout,
                "bounds": [[-1] * 2 * N, [1] * 2 * N],
            },
        )

        def objective_func(params):
            pulse = Pulse(params[:N] + 1j * params[N:])
            result = self.state_tomography({qubit: pulse}, x90={qubit: x90})
            loss = np.linalg.norm(result[qubit] - np.array(target_state))
            return loss

        es.optimize(objective_func)
        x = es.result.xbest
        opt_pulse = Pulse(x[:N] + 1j * x[N:])
        return opt_pulse

    def optimize_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        obejective_type: str = "orbit",
        optimize_method: str = "cma",
        update_cr_param: bool = True,
        opt_params: Collection[str] | None = None,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
        maxiter: int = 30,
        n_cliffords: int = 10,
        n_trials: int = 3,
        duration: float | None = None,
        ramptime: float | None = None,
        x180: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ):
        if opt_params is None:
            opt_params = [
                "cr_amplitude",
                "cr_phase",
                "cr_beta",
                "cancel_amplitude",
                "cancel_phase",
                "cancel_beta",
                "rotary_amplitude",
            ]

        if x180 is None:
            x180 = {
                control_qubit: self.x180(control_qubit),
                target_qubit: self.x180(target_qubit),
            }

        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.calib_note.get_cr_param(cr_label)
        if cr_param is None:
            raise ValueError("CR parameters are not stored.")

        if duration is None:
            duration = cr_param["duration"]
        if ramptime is None:
            ramptime = cr_param["ramptime"]
        if x180_margin is None:
            x180_margin = 0.0

        defaults = {
            "cr_amplitude": {
                "initial": cr_param["cr_amplitude"],
                "bounds": [0.0, 1.0],
                "std": 0.001,
            },
            "cr_phase": {
                "initial": cr_param["cr_phase"],
                "bounds": [-np.pi, np.pi],
                "std": 0.001,
            },
            "cr_beta": {
                "initial": cr_param["cr_beta"],
                "bounds": [-10.0, 10.0],
                "std": 0.001,
            },
            "cancel_amplitude": {
                "initial": cr_param["cancel_amplitude"],
                "bounds": [0.0, 1.0],
                "std": 0.001,
            },
            "cancel_phase": {
                "initial": cr_param["cancel_phase"],
                "bounds": [-np.pi, np.pi],
                "std": 0.001,
            },
            "cancel_beta": {
                "initial": cr_param["cancel_beta"],
                "bounds": [-10.0, 10.0],
                "std": 0.001,
            },
            "rotary_amplitude": {
                "initial": cr_param["rotary_amplitude"],
                "bounds": [0.0, 20.0],
                "std": 0.01,
            },
        }

        for opt_param in opt_params:
            if opt_param not in defaults:
                raise ValueError(f"Invalid optimization parameter: {opt_param}")

        if isinstance(opt_params, list):
            opt_params_dict = {p: defaults[p] for p in opt_params}
        elif isinstance(opt_params, dict):
            opt_params_dict = opt_params
        else:
            raise ValueError("opt_params must be a list or dictionary.")
        opt_params = list(opt_params_dict.keys())

        best_loss = float("inf")
        best_params = None

        def objective_func(params_vec):
            nonlocal best_loss, best_params

            params = {k: v["initial"] for k, v in defaults.items()}
            for k, v in zip(opt_params, params_vec):
                params[k] = v

            cr_amplitude = params["cr_amplitude"]
            cr_phase = params["cr_phase"]
            cr_beta = params["cr_beta"]
            cancel_amplitude = params["cancel_amplitude"]
            cancel_phase = params["cancel_phase"]
            cancel_beta = params["cancel_beta"]
            rotary_amplitude = params["rotary_amplitude"]

            cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + (
                rotary_amplitude + 0j
            )

            pi_pulse = x180.get(control_qubit, self.x180(control_qubit))

            ecr = CrossResonance(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                cr_amplitude=cr_amplitude,
                cr_duration=duration,
                cr_ramptime=ramptime,
                cr_phase=cr_phase,
                cr_beta=cr_beta,
                cancel_amplitude=np.abs(cancel_pulse),
                cancel_phase=np.angle(cancel_pulse),
                cancel_beta=cancel_beta,
                echo=True,
                pi_pulse=pi_pulse,
                pi_margin=x180_margin,
            )

            if obejective_type == "orbit":
                loss_list = []
                for _ in range(n_trials):
                    rb_seq = self.rb_sequence_2q(
                        cr_label,
                        n=n_cliffords,
                        zx90=ecr,
                    )
                    result = self.measure(
                        rb_seq,
                        mode="single",
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    prob = result.get_mitigated_probabilities(
                        [control_qubit, target_qubit]
                    )
                    loss = 1 - prob["00"]
                    loss_list.append(loss)
                loss = np.mean(loss_list)
            else:
                ecr_0 = ecr
                with PulseSchedule([control_qubit, cr_label, target_qubit]) as ecr_1:
                    ecr_1.add(control_qubit, pi_pulse)
                    ecr_1.barrier()
                    ecr_1.call(ecr_0)

                result_0 = self.state_tomography(
                    ecr_0,
                    shots=shots,
                    interval=interval,
                )
                result_1 = self.state_tomography(
                    ecr_1,
                    shots=shots,
                    interval=interval,
                )

                loss_c0 = np.linalg.norm(result_0[control_qubit] - np.array((0, 0, 1)))
                loss_t0 = np.linalg.norm(result_0[target_qubit] - np.array((0, -1, 0)))
                loss_c1 = np.linalg.norm(result_1[control_qubit] - np.array((0, 0, -1)))
                loss_t1 = np.linalg.norm(result_1[target_qubit] - np.array((0, 1, 0)))

                loss = loss_c0 + loss_t0 + loss_c1 + loss_t1

            if loss < best_loss:
                best_loss = loss
                best_params = params_vec.copy()

            return loss

        initial_params = [opt_params_dict[k]["initial"] for k in opt_params]

        try:
            if optimize_method == "Nelder-Mead":

                def nm_callback(xk):
                    current_loss = objective_func(xk)
                    print(f"[Nelder-Mead] step: loss = {current_loss:.6f}, x = {xk}")

                result = scipy.optimize.minimize(
                    objective_func,
                    x0=initial_params,
                    method="Nelder-Mead",
                    callback=nm_callback,
                    options={
                        "maxiter": maxiter,
                        "fatol": ftarget,
                        "disp": True,
                    },
                )

                print("Optimized parameters:")
                opt_result = {}
                for key, value in zip(opt_params, result.x):
                    opt_result[key] = value
                    print(f"  {key} : {value:.6f}")

                cr_param.update(opt_result)
                if update_cr_param:
                    self.calib_note.update_cr_param(cr_label, cr_param)

                return {
                    "cr_param": cr_param,
                    "result": result,
                }
            elif optimize_method == "cma":
                bounds0 = [opt_params_dict[k]["bounds"][0] for k in opt_params]
                bounds1 = [opt_params_dict[k]["bounds"][1] for k in opt_params]
                bounds = [bounds0, bounds1]
                stds = [opt_params_dict[k]["std"] for k in opt_params]

                es = cma.CMAEvolutionStrategy(
                    initial_params,
                    1.0,
                    {
                        "seed": seed,
                        "maxiter": maxiter,
                        "ftarget": ftarget,
                        "timeout": timeout,
                        "bounds": bounds,
                        "CMA_stds": stds,
                    },
                )
                es.optimize(objective_func)

                print("Optimized parameters:")
                opt_result = {}
                for key, value in zip(opt_params, es.result.xbest):
                    opt_result[key] = value
                    print(f"  {key} : {value:.6f}")

                cr_param.update(opt_result)
                if update_cr_param:
                    self.calib_note.update_cr_param(cr_label, cr_param)

                return {
                    "cr_param": cr_param,
                    "result": es.result,
                }
            else:
                raise ValueError(f"Unsupported optimization method: {optimize_method}")

        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            if best_params is not None:
                opt_result = {k: v for k, v in zip(opt_params, best_params)}
                cr_param.update(opt_result)  # type: ignore
                if update_cr_param:
                    self.calib_note.update_cr_param(cr_label, cr_param)
                return {
                    "cr_param": cr_param,
                    "result": {
                        "xbest": best_params,
                        "fbest": best_loss,
                    },
                }
