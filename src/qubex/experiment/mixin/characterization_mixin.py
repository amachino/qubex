from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from typing import Collection, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from tqdm import tqdm

from ...analysis import fitting
from ...analysis import visualization as viz
from ...analysis.fitting import RabiParam
from ...backend import BoxType, MixingUtil, Target
from ...backend.experiment_system import (
    CNCO_CENTER_CTRL,
    CNCO_CETNER_READ,
    CNCO_CETNER_READ_R8,
)
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS, SAMPLING_PERIOD
from ...pulse import CPMG, Blank, FlatTop, Gaussian, PulseSchedule, Rect, Waveform
from ...typing import TargetMap
from ..experiment_constants import CALIBRATION_SHOTS, RABI_FREQUENCY, RABI_TIME_RANGE
from ..experiment_result import (
    AmplRabiData,
    ExperimentResult,
    FreqRabiData,
    RabiData,
    RamseyData,
    T1Data,
    T2Data,
)
from ..experiment_util import ExperimentUtil
from ..protocol import BaseProtocol, CharacterizationProtocol, MeasurementProtocol

logger = logging.getLogger(__name__)


class CharacterizationMixin(
    BaseProtocol,
    MeasurementProtocol,
    CharacterizationProtocol,
):
    def measure_readout_snr(
        self,
        targets: Collection[str] | str | None = None,
        *,
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        sequence = {
            target: self.get_pulse_for_state(
                target=target,
                state=initial_state,
            )
            for target in targets
        }

        result = self.measure(
            sequence=sequence,
            mode="single",
            shots=shots,
            interval=interval,
            capture_window=capture_window,
            capture_margin=capture_margin,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
        )

        if plot:
            result.plot(save_image=save_image)

        signal = {}
        noise = {}
        snr = {}
        for target, data in result.data.items():
            iq = data.kerneled
            signal[target] = np.abs(np.average(iq))
            noise[target] = np.std(iq)
            snr[target] = signal[target] / noise[target]
        return {
            "signal": signal,
            "noise": noise,
            "snr": snr,
        }

    def sweep_readout_amplitude(
        self,
        targets: Collection[str] | str | None = None,
        *,
        amplitude_range: ArrayLike = np.linspace(0.0, 0.1, 21),
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        amplitude_range = np.asarray(amplitude_range)

        signal_buf = defaultdict(list)
        noise_buf = defaultdict(list)
        snr_buf = defaultdict(list)

        for amplitude in tqdm(amplitude_range):
            result = self.measure_readout_snr(
                targets=targets,
                initial_state=initial_state,
                capture_window=capture_window,
                capture_margin=capture_margin,
                readout_duration=readout_duration,
                readout_amplitudes={target: amplitude for target in targets},
                shots=shots,
                interval=interval,
                plot=False,
            )
            for target in targets:
                signal_buf[target].append(result["signal"][target])
                noise_buf[target].append(result["noise"][target])
                snr_buf[target].append(result["snr"][target])

        signal = {target: np.array(signal_buf[target]) for target in targets}
        noise = {target: np.array(noise_buf[target]) for target in targets}
        snr = {target: np.array(snr_buf[target]) for target in targets}

        if plot:
            for target in targets:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
                fig.add_trace(
                    go.Scatter(
                        x=amplitude_range,
                        y=signal[target],
                        mode="lines+markers",
                        name="Signal",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=amplitude_range,
                        y=noise[target],
                        mode="lines+markers",
                        name="Noise",
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=amplitude_range,
                        y=snr[target],
                        mode="lines+markers",
                        name="SNR",
                    ),
                    row=3,
                    col=1,
                )
                fig.update_layout(
                    title=f"Readout SNR : {target}",
                    xaxis3_title="Readout amplitude (arb. units)",
                    yaxis_title="Signal",
                    yaxis2_title="Noise",
                    yaxis3_title="SNR",
                    showlegend=False,
                    width=600,
                    height=400,
                )
                fig.show()
                viz.save_figure_image(
                    fig,
                    f"readout_snr_{target}",
                    width=600,
                    height=400,
                )

        return {
            "signal": signal,
            "noise": noise,
            "snr": snr,
        }

    def sweep_readout_duration(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(128, 2048, 128),
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_margin: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = np.asarray(time_range)

        signal_buf = defaultdict(list)
        noise_buf = defaultdict(list)
        snr_buf = defaultdict(list)

        for T in time_range:
            result = self.measure_readout_snr(
                targets=targets,
                initial_state=initial_state,
                capture_window=T + 512,
                capture_margin=capture_margin,
                readout_duration=T,
                readout_amplitudes=readout_amplitudes,
                shots=shots,
                interval=interval,
                plot=False,
            )
            for target in targets:
                signal_buf[target].append(result["signal"][target])
                noise_buf[target].append(result["noise"][target])
                snr_buf[target].append(result["snr"][target])

        signal = {target: np.array(signal_buf[target]) for target in targets}
        noise = {target: np.array(noise_buf[target]) for target in targets}
        snr = {target: np.array(snr_buf[target]) for target in targets}

        if plot:
            for target in targets:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
                fig.add_trace(
                    go.Scatter(
                        x=time_range,
                        y=signal[target],
                        mode="lines+markers",
                        name="Signal",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_range,
                        y=noise[target],
                        mode="lines+markers",
                        name="Noise",
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_range,
                        y=snr[target],
                        mode="lines+markers",
                        name="SNR",
                    ),
                    row=3,
                    col=1,
                )
                fig.update_layout(
                    title=f"Readout SNR : {target}",
                    xaxis3_title="Readout duration (ns)",
                    yaxis_title="Signal",
                    yaxis2_title="Noise",
                    yaxis3_title="SNR",
                    showlegend=False,
                    width=600,
                    height=400,
                )
                fig.show()
                viz.save_figure_image(
                    fig,
                    f"readout_snr_{target}",
                    width=600,
                    height=400,
                )

        return {
            "signal": signal,
            "noise": noise,
            "snr": snr,
        }

    def chevron_pattern(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.05, 0.05, 51),
        time_range: ArrayLike = RABI_TIME_RANGE,
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        rabi_params: dict[str, RabiParam] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if frequencies is None:
            frequencies = {target: self.targets[target].frequency for target in targets}

        detuning_range = np.array(detuning_range, dtype=np.float64)
        time_range = np.array(time_range, dtype=np.float64)

        if amplitudes is None:
            amplitudes = {
                target: self.params.control_amplitude[target] for target in targets
            }

        shared_rabi_params: dict[str, RabiParam]
        if rabi_params is None:
            print("Obtaining Rabi parameters...")
            shared_rabi_params = self.obtain_rabi_params(
                targets=targets,
                amplitudes=amplitudes,
                time_range=time_range,
                frequencies=frequencies,
                shots=shots,
                interval=interval,
                plot=False,
                store_params=False,
            ).rabi_params  # type: ignore
        else:
            shared_rabi_params = rabi_params

        rabi_rates: dict[str, NDArray] = {}
        chevron_data: dict[str, NDArray] = {}
        resonant_frequencies: dict[str, float] = {}

        print(f"Targets : {targets}")
        subgroups = self.util.create_qubit_subgroups(targets)
        for idx, subgroup in enumerate(subgroups):
            if len(subgroup) == 0:
                continue

            print(f"Subgroup ({idx + 1}/{len(subgroups)}) : {subgroup}")

            rabi_rates_buffer: dict[str, list[float]] = defaultdict(list)
            chevron_data_buffer: dict[str, list[NDArray]] = defaultdict(list)

            for detuning in tqdm(detuning_range):
                with self.util.no_output():
                    sweep_result = self.sweep_parameter(
                        sequence=lambda t: {
                            label: Rect(duration=t, amplitude=amplitudes[label])
                            for label in subgroup
                        },
                        sweep_range=time_range,
                        frequencies={
                            label: frequencies[label] + detuning for label in subgroup
                        },
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    sweep_data = sweep_result.data

                    for target, data in sweep_data.items():
                        fit_result = fitting.fit_rabi(
                            target=data.target,
                            times=data.sweep_range,
                            data=data.data,
                            plot=False,
                        )
                        rabi_param = fit_result["rabi_param"]
                        rabi_rates_buffer[target].append(rabi_param.frequency)
                        data.rabi_param = shared_rabi_params[target]
                        chevron_data_buffer[target].append(data.normalized)

            for target in subgroup:
                rabi_rates[target] = np.array(rabi_rates_buffer[target])
                chevron_data[target] = np.array(chevron_data_buffer[target]).T

                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(
                        x=detuning_range + frequencies[target],
                        y=time_range,
                        z=chevron_data[target],
                        colorscale="Viridis",
                    )
                )
                fig.update_layout(
                    title=dict(
                        text=f"Chevron pattern : {target}",
                        subtitle=dict(
                            text=f"control_amplitude={amplitudes[target]}",
                            font=dict(
                                size=13,
                                family="monospace",
                            ),
                        ),
                    ),
                    xaxis_title="Drive frequency (GHz)",
                    yaxis_title="Time (ns)",
                    width=600,
                    height=400,
                    margin=dict(t=80),
                )
                if plot:
                    fig.show()

                fit_result = fitting.fit_detuned_rabi(
                    target=target,
                    control_frequencies=detuning_range + frequencies[target],
                    rabi_frequencies=rabi_rates[target],
                    plot=plot,
                )
                resonant_frequencies[target] = fit_result["f_resonance"]

                if save_image:
                    viz.save_figure_image(
                        fig,
                        name=f"chevron_pattern_{target}",
                        width=600,
                        height=400,
                    )
                    fig_fit = fit_result["fig"]
                    if fig_fit is not None:
                        viz.save_figure_image(
                            fig_fit,
                            name=f"chevron_pattern_fit_{target}",
                            width=600,
                            height=300,
                        )

        rabi_rates = dict(sorted(rabi_rates.items()))
        chevron_data = dict(sorted(chevron_data.items()))
        resonant_frequencies = dict(sorted(resonant_frequencies.items()))

        return {
            "time_range": time_range,
            "detuning_range": detuning_range,
            "frequencies": frequencies,
            "chevron_data": chevron_data,
            "rabi_rates": rabi_rates,
            "resonant_frequencies": resonant_frequencies,
        }

    def obtain_freq_rabi_relation(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = np.arange(0, 101, 4),
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[FreqRabiData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        detuning_range = np.array(detuning_range, dtype=np.float64)
        time_range = np.array(time_range, dtype=np.float64)
        ampl = self.params.control_amplitude
        rabi_rates: dict[str, list[float]] = defaultdict(list)
        rabi_data: dict[str, list[RabiData]] = defaultdict(list)

        for detuning in tqdm(detuning_range):
            if rabi_level == "ge":
                rabi_result = self.rabi_experiment(
                    time_range=time_range,
                    amplitudes={target: ampl[target] for target in targets},
                    detuning=detuning,
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
            elif rabi_level == "ef":
                rabi_result = self.ef_rabi_experiment(
                    time_range=time_range,
                    amplitudes={
                        target: ampl[target] / np.sqrt(2) for target in targets
                    },
                    detuning=detuning,
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
            else:
                raise ValueError("Invalid rabi_level.")
            if plot:
                rabi_result.fit()
            rabi_params = rabi_result.rabi_params
            if rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
            for target, param in rabi_params.items():
                rabi_rate = param.frequency
                rabi_rates[target].append(rabi_rate)
            for target, data in rabi_result.data.items():
                rabi_data[target].append(data)

        frequencies = {
            target: detuning_range + self.targets[target].frequency
            for target in rabi_rates
        }

        data = {
            target: FreqRabiData(
                target=target,
                data=np.array(values, dtype=np.float64),
                sweep_range=detuning_range,
                frequency_range=frequencies[target],
                rabi_data=rabi_data[target],
            )
            for target, values in rabi_rates.items()
        }
        result = ExperimentResult(data=data)
        if plot:
            result.fit()
        return result

    def obtain_ampl_rabi_relation(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        amplitude_range: ArrayLike = np.linspace(0.01, 0.1, 10),
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[AmplRabiData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = np.array(time_range, dtype=np.float64)
        amplitude_range = np.array(amplitude_range, dtype=np.float64)
        rabi_rates: dict[str, list[float]] = defaultdict(list)
        rabi_data: dict[str, list[RabiData]] = defaultdict(list)

        for amplitude in tqdm(amplitude_range):
            rabi_result = self.rabi_experiment(
                amplitudes={target: amplitude for target in targets},
                time_range=time_range,
                shots=shots,
                interval=interval,
                plot=False,
            )
            rabi_params = rabi_result.rabi_params
            if rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
            for target, param in rabi_params.items():
                rabi_rate = param.frequency
                rabi_rates[target].append(rabi_rate)
            for target, data in rabi_result.data.items():
                rabi_data[target].append(data)
            if plot:
                rabi_result.fit()

        data = {
            target: AmplRabiData(
                target=target,
                data=np.array(rabi_rate),
                sweep_range=amplitude_range,
                rabi_data=rabi_data[target],
            )
            for target, rabi_rate in rabi_rates.items()
        }
        result = ExperimentResult(data=data)
        if plot:
            result.fit()
        return result

    def calibrate_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = range(0, 101, 4),
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]:
        result = self.chevron_pattern(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            frequencies=frequencies,
            amplitudes=amplitudes,
            shots=shots,
            interval=interval,
            plot=plot,
        )
        resonant_frequencies = result["resonant_frequencies"]

        print("\nResults\n-------")
        print("ge frequency (GHz):")
        for target, frequency in resonant_frequencies.items():
            print(f"    {target}: {frequency:.6f}")
        return resonant_frequencies

    def calibrate_ef_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = np.arange(0, 101, 4),
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        detuning_range = np.array(detuning_range, dtype=np.float64)
        time_range = np.array(time_range, dtype=np.float64)

        result = self.obtain_freq_rabi_relation(
            targets=targets,
            detuning_range=detuning_range,
            rabi_level="ef",
            time_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
        )
        fit_data = {
            target: data.fit()["f_resonance"] for target, data in result.data.items()
        }

        print("\nResults\n-------")
        print("ef frequency (GHz):")
        for target, fit in fit_data.items():
            label = Target.ge_label(target)
            print(f"    {label}: {fit:.6f}")
        print("anharmonicity (GHz):")
        for target, fit in fit_data.items():
            label = Target.ge_label(target)
            ge_freq = self.targets[label].frequency
            print(f"    {label}: {fit - ge_freq:.6f}")
        return fit_data

    def calibrate_readout_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = range(0, 101, 4),
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict[str, float]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        detuning_range = np.array(detuning_range, dtype=np.float64)

        # store the original readout amplitudes
        original_readout_amplitudes = deepcopy(self.params.readout_amplitude)

        result = defaultdict(list)
        for detuning in tqdm(detuning_range):
            with self.util.no_output():
                if readout_amplitudes is not None:
                    # modify the readout amplitudes if necessary
                    for target, amplitude in readout_amplitudes.items():
                        label = Target.qubit_label(target)
                        self.params.readout_amplitude[label] = amplitude

                rabi_result = self.rabi_experiment(
                    time_range=time_range,
                    amplitudes={
                        target: self.params.control_amplitude[target]
                        for target in targets
                    },
                    frequencies={
                        resonator.label: resonator.frequency + detuning
                        for resonator in self.resonators.values()
                    },
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
                for qubit, data in rabi_result.data.items():
                    rabi_amplitude = data.rabi_param.amplitude
                    result[qubit].append(rabi_amplitude)

        # restore the original readout amplitudes
        self.params.readout_amplitude = original_readout_amplitudes

        fit_data = {}
        for target, values in result.items():
            freq = self.resonators[target].frequency
            fit_result = fitting.fit_lorentzian(
                target=target,
                x=detuning_range + freq,
                y=np.array(values),
                plot=plot,
                title="Readout frequency calibration",
                xlabel="Readout frequency (GHz)",
            )
            if "f0" in fit_result:
                fit_data[target] = fit_result["f0"]

            if save_image:
                fig = fit_result["fig"]
                if fig is not None:
                    viz.save_figure_image(
                        fig,
                        name=f"readout_frequency_{target}",
                        width=600,
                        height=300,
                    )

        print("\nResults\n-------")
        for target, freq in fit_data.items():
            print(f"{target}: {freq:.6f}")

        return fit_data

    def t1_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
        xaxis_type: Literal["linear", "log"] = "log",
    ) -> ExperimentResult[T1Data]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        self.validate_rabi_params(targets)

        if time_range is None:
            time_range = np.logspace(
                np.log10(100),
                np.log10(100 * 1000),
                51,
            )
        time_range = self.util.discretize_time_range(np.asarray(time_range))

        data: dict[str, T1Data] = {}

        subgroups = self.util.create_qubit_subgroups(targets)
        print(f"Target qubits: {targets}")
        print(f"Subgroups: {subgroups}")
        for idx, subgroup in enumerate(subgroups):
            if len(subgroup) == 0:
                continue

            def t1_sequence(T: int) -> PulseSchedule:
                with PulseSchedule(subgroup) as ps:
                    for target in subgroup:
                        ps.add(target, self.hpi_pulse[target].repeated(2))
                        ps.add(target, Blank(T))
                return ps

            print(
                f"({idx + 1}/{len(subgroups)}) Conducting T1 experiment for {subgroup}...\n"
            )

            sweep_result = self.sweep_parameter(
                sequence=t1_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
                title="T1 decay",
                xlabel="Time (μs)",
                ylabel="Measured value",
                xaxis_type=xaxis_type,
            )

            for target, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_exp_decay(
                    target=target,
                    x=sweep_data.sweep_range,
                    y=0.5 * (1 - sweep_data.normalized),
                    plot=plot,
                    title="T1",
                    xlabel="Time (μs)",
                    ylabel="Normalized signal",
                    xaxis_type=xaxis_type,
                    yaxis_type="linear",
                )
                if fit_result["status"] == "success":
                    t1 = fit_result["tau"]
                    t1_err = fit_result["tau_err"]
                    r2 = fit_result["r2"]

                    t1_data = T1Data.new(
                        sweep_data,
                        t1=t1,
                        t1_err=t1_err,
                        r2=r2,
                    )
                    data[target] = t1_data

                    fig = fit_result["fig"]

                    if save_image:
                        viz.save_figure_image(
                            fig,
                            name=f"t1_{target}",
                        )

        return ExperimentResult(data=data)

    def t2_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        n_cpmg: int = 1,
        pi_cpmg: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> ExperimentResult[T2Data]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        self.validate_rabi_params(targets)

        if time_range is None:
            time_range = np.logspace(
                np.log10(300),
                np.log10(100 * 1000),
                51,
            )
        time_range = self.util.discretize_time_range(
            time_range=np.asarray(time_range),
            sampling_period=2 * SAMPLING_PERIOD,
        )

        data: dict[str, T2Data] = {}

        subgroups = self.util.create_qubit_subgroups(targets)

        print(f"Target qubits: {targets}")
        print(f"Subgroups: {subgroups}")
        for idx, subgroup in enumerate(subgroups):
            if len(subgroup) == 0:
                continue

            def t2_sequence(T: int) -> PulseSchedule:
                with PulseSchedule(subgroup) as ps:
                    for target in subgroup:
                        hpi = self.hpi_pulse[target]
                        pi = pi_cpmg or hpi.repeated(2)
                        ps.add(target, hpi)
                        if T > 0:
                            ps.add(
                                target,
                                CPMG(
                                    tau=(T - pi.duration * n_cpmg) // (2 * n_cpmg),
                                    pi=pi,
                                    n=n_cpmg,
                                ),
                            )
                        ps.add(target, hpi.shifted(np.pi))
                return ps

            print(
                f"({idx + 1}/{len(subgroups)}) Conducting T2 experiment for {subgroup}...\n"
            )

            sweep_result = self.sweep_parameter(
                sequence=t2_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
            )

            for target, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_exp_decay(
                    target=target,
                    x=sweep_data.sweep_range,
                    y=0.5 * (1 - sweep_data.normalized),
                    plot=plot,
                    title="T2 echo",
                    xlabel="Time (μs)",
                    ylabel="Normalized signal",
                )
                if fit_result["status"] == "success":
                    t2 = fit_result["tau"]
                    t2_err = fit_result["tau_err"]
                    r2 = fit_result["r2"]

                    t2_data = T2Data.new(
                        sweep_data,
                        t2=t2,
                        t2_err=t2_err,
                        r2=r2,
                    )
                    data[target] = t2_data

                    fig = fit_result["fig"]

                    if save_image:
                        viz.save_figure_image(
                            fig,
                            name=f"t2_echo_{target}",
                        )

        return ExperimentResult(data=data)

    def ramsey_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(0, 10_001, 100),
        detuning: float = 0.001,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> ExperimentResult[RamseyData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = self.util.discretize_time_range(time_range)
        self.validate_rabi_params(targets)

        target_groups = self.util.create_qubit_subgroups(targets)
        spectator_groups = reversed(target_groups)  # TODO: make it more general

        data: dict[str, RamseyData] = {}

        for target_qubits, spectator_qubits in zip(target_groups, spectator_groups):
            if spectator_state != "0":
                target_list = target_qubits + spectator_qubits
            else:
                target_list = target_qubits

            if len(target_list) == 0:
                continue

            print(f"Target qubits: {target_qubits}")
            print(f"Spectator qubits: {spectator_qubits}")

            def ramsey_sequence(T: int) -> PulseSchedule:
                with PulseSchedule(target_list) as ps:
                    # Excite spectator qubits if needed
                    if spectator_state != "0":
                        for spectator in spectator_qubits:
                            if spectator in self.qubit_labels:
                                pulse = self.get_pulse_for_state(
                                    target=spectator,
                                    state=spectator_state,
                                )
                                ps.add(spectator, pulse)
                        ps.barrier()

                    # Ramsey sequence for the target qubit
                    for target in target_qubits:
                        hpi = self.hpi_pulse[target]
                        ps.add(target, hpi)
                        ps.add(target, Blank(T))
                        ps.add(target, hpi.shifted(np.pi))
                return ps

            detuned_frequencies = {
                target: self.qubits[target].frequency + detuning
                for target in target_qubits
            }

            sweep_result = self.sweep_parameter(
                sequence=ramsey_sequence,
                sweep_range=time_range,
                frequencies=detuned_frequencies,
                shots=shots,
                interval=interval,
                plot=plot,
            )

            for target, sweep_data in sweep_result.data.items():
                if target in target_qubits:
                    fit_result = fitting.fit_ramsey(
                        target=target,
                        times=sweep_data.sweep_range,
                        data=sweep_data.normalized,
                        plot=plot,
                    )
                    if fit_result["status"] == "success":
                        f = self.qubits[target].frequency
                        t2 = fit_result["tau"]
                        ramsey_freq = fit_result["f"]
                        bare_freq = f + detuning - ramsey_freq
                        r2 = fit_result["r2"]
                        ramsey_data = RamseyData.new(
                            sweep_data=sweep_data,
                            t2=t2,
                            ramsey_freq=ramsey_freq,
                            bare_freq=bare_freq,
                            r2=r2,
                        )
                        data[target] = ramsey_data

                        print(f"Bare frequency with |{spectator_state}〉:")
                        print(f"  {target}: {ramsey_data.bare_freq:.6f}")
                        print("")

                        fig = fit_result["fig"]

                        if save_image:
                            viz.save_figure_image(
                                fig,
                                name=f"ramsey_{target}",
                            )

        return ExperimentResult(data=data)

    def obtain_effective_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(0, 10001, 100),
        detuning: float = 0.001,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = np.asarray(time_range)
        self.validate_rabi_params(targets)

        result_0 = self.ramsey_experiment(
            targets=targets,
            time_range=time_range,
            detuning=detuning,
            spectator_state="0",
            shots=shots,
            interval=interval,
            plot=plot,
        )

        result_1 = self.ramsey_experiment(
            targets=targets,
            time_range=time_range,
            detuning=detuning,
            spectator_state="1",
            shots=shots,
            interval=interval,
            plot=plot,
        )

        effective_freq = {
            target: (result_0.data[target].bare_freq + result_1.data[target].bare_freq)
            * 0.5
            for target in targets
        }

        for target in targets:
            print(f"Target: {target}")
            print(f"  Original frequency: {self.targets[target].frequency:.6f}")
            print(f"  Bare frequency with |0>: {result_0.data[target].bare_freq:.6f}")
            print(f"  Bare frequency with |1>: {result_1.data[target].bare_freq:.6f}")
            print(f"  Effective control frequency: {effective_freq[target]:.6f}")
            print("")

        return {
            "effective_freq": effective_freq,
            "result_0": result_0,
            "result_1": result_1,
        }

    def jazz_experiment(
        self,
        target_qubit: str,
        spectator_qubit: str,
        *,
        time_range: ArrayLike = np.arange(0, 2001, 100),
        x90: Waveform | TargetMap[Waveform] | None = None,
        x180: Waveform | TargetMap[Waveform] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ):
        if x90 is None:
            x90 = {
                target_qubit: self.hpi_pulse[target_qubit],
            }
        elif isinstance(x90, Waveform):
            x90 = {
                target_qubit: x90,
            }

        if x180 is None:
            x180 = {
                target_qubit: self.hpi_pulse[target_qubit].repeated(2),
                spectator_qubit: self.hpi_pulse[spectator_qubit].repeated(2),
            }
        elif isinstance(x180, Waveform):
            x180 = {
                target_qubit: x180,
                spectator_qubit: x180,
            }

        def jazz_sequence(tau: float) -> PulseSchedule:
            with PulseSchedule([target_qubit, spectator_qubit]) as ps:
                ps.add(target_qubit, x90[target_qubit])
                ps.add(target_qubit, Blank(tau))
                ps.barrier()
                ps.add(target_qubit, x180[target_qubit])
                ps.add(spectator_qubit, x180[spectator_qubit])
                ps.add(target_qubit, Blank(tau))
                ps.add(target_qubit, x90[target_qubit].scaled(-1))
            return ps

        time_range = np.asarray(time_range)
        self.validate_rabi_params([target_qubit, spectator_qubit])

        result = self.sweep_parameter(
            sequence=jazz_sequence,
            sweep_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
            title=f"JAZZ : {target_qubit}-{spectator_qubit}",
            xlabel="Time (ns)",
            ylabel="Measured value",
        )

        fit_result = fitting.fit_cosine(
            time_range * 2,
            (1 - result.data[target_qubit].normalized) * 0.5,
            is_damped=True,
            plot=plot,
            title=f"JAZZ : {target_qubit}-{spectator_qubit}",
            xlabel="Wait time (ns)",
            ylabel=f"Normalized value : {target_qubit}",
        )

        xi = fit_result["f"]
        zeta = 2 * xi

        print(f"ξ: {xi * 1e6:.2f} kHz")
        print(f"ζ: {zeta * 1e6:.2f} kHz")

        return {
            "xi": xi,
            "zeta": zeta,
            **fit_result,
        }

    def obtain_coupling_strength(
        self,
        target_qubit: str,
        spectator_qubit: str,
        *,
        time_range: ArrayLike = np.arange(0, 20001, 500),
        x90: Waveform | TargetMap[Waveform] | None = None,
        x180: Waveform | TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        qubit_1 = target_qubit
        qubit_2 = spectator_qubit

        result = self.jazz_experiment(
            target_qubit=qubit_1,
            spectator_qubit=qubit_2,
            time_range=time_range,
            x90=x90,
            x180=x180,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        xi = result["xi"]
        zeta = result["zeta"]

        f_1 = self.qubits[qubit_1].frequency
        f_2 = self.qubits[qubit_2].frequency

        a_1 = self.qubits[qubit_1].anharmonicity
        a_2 = self.qubits[qubit_2].anharmonicity

        Delta_12 = f_1 - f_2

        g = np.sqrt(np.abs((xi * (Delta_12 + a_1) * (Delta_12 - a_2)) / (a_1 + a_2)))

        print(f"frequency_1: {f_1:.2f} GHz")
        print(f"frequency_2: {f_2:.2f} GHz")
        print(f"Delta_12: {Delta_12 * 1e3:.2f} MHz")
        print(f"anharmonicity_1: {a_1 * 1e3:.2f} MHz")
        print(f"anharmonicity_2: {a_2 * 1e3:.2f} MHz")
        print(f"g: {g * 1e3:.2f} MHz")

        return {
            "xi": xi,
            "zeta": zeta,
            "g": g,
        }

    def measure_phase_shift(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        amplitude: float | None = None,
        subrange_width: float = 0.3,
        shots: int = 128,
        interval: float = 0,
        plot: bool = True,
    ) -> float:
        read_label = Target.read_label(target)
        qubit_label = Target.qubit_label(target)
        mux = self.experiment_system.get_mux_by_qubit(qubit_label)
        read_box = self.experiment_system.get_readout_box_for_qubit(qubit_label)

        if amplitude is None:
            amplitude = self.params.readout_amplitude[qubit_label]

        if frequency_range is None:
            if read_box.type == BoxType.QUEL1SE_R8:
                frequency_range = np.arange(5.90, 5.95, 0.002)
            else:
                frequency_range = np.arange(9.90, 9.95, 0.002)
        else:
            frequency_range = np.array(frequency_range)
        # split frequency range to avoid the frequency sweep range limit
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=subrange_width,
        )

        # result buffer
        phases: list[float] = []
        # measure phase shift
        idx = 0

        # phase offset to avoid the phase jump after changing the LO/NCO frequency
        phase_offset = 0.0

        if read_box.type == BoxType.QUEL1SE_R8:
            ssb = "L"
            cnco_center = CNCO_CETNER_READ_R8
        else:
            ssb = "U"
            cnco_center = CNCO_CETNER_READ

        for subrange in subranges:
            # change LO/NCO frequency to the center of the subrange
            f_center = (subrange[0] + subrange[-1]) / 2
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_center * 1e9,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            with self.state_manager.modified_device_settings(
                label=read_label,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                logger.debug(f"LO: {lo}, CNCO: {cnco}")
                for sub_idx, freq in enumerate(subrange):
                    if idx > 0 and sub_idx == 0:
                        # measure the phase at the previous frequency with the new LO/NCO settings
                        with self.modified_frequencies(
                            {read_label: frequency_range[idx - 1]}
                        ):
                            new_result = self.measure(
                                {qubit_label: np.zeros(0)},
                                mode="avg",
                                readout_amplitudes={qubit_label: amplitude},
                                shots=shots,
                                interval=interval,
                                plot=False,
                            )
                            new_signal = new_result.data[target].kerneled
                            new_phase = np.angle(new_signal)
                            phase_offset = new_phase - phases[-1]  # type: ignore

                    with self.modified_frequencies({read_label: freq}):
                        result = self.measure(
                            {qubit_label: np.zeros(0)},
                            mode="avg",
                            readout_amplitudes={qubit_label: amplitude},
                            shots=shots,
                            interval=interval,
                            plot=False,
                        )
                        signal = result.data[target].kerneled
                        phase = np.angle(signal)
                        phase = phase - phase_offset
                        phases.append(phase)  # type: ignore

                        idx += 1

        # fit the phase shift
        unwrapped = np.unwrap(phases)
        diff = np.diff(unwrapped)
        shift_steps = np.where(diff < 0, 2 * np.pi, 0)
        cum_shift = np.cumsum(np.insert(shift_steps, 0, 0))
        unwrapped += cum_shift

        x, y = frequency_range, unwrapped
        coefficients = np.polyfit(x, y, 1)
        y_fit = np.polyval(coefficients, x)

        if plot:
            fig = go.Figure()
            fig.add_scatter(name="data", mode="markers", x=x, y=y)
            fig.add_scatter(name="fit", mode="lines", x=x, y=y_fit)
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.95,
                y=0.95,
                text=f"Phase shift: {coefficients[0] * 1e-3:.3f} rad/MHz",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
            )
            fig.update_layout(
                title=f"Phase shift : {mux.label}",
                xaxis_title="Frequency (GHz)",
                yaxis_title="Unwrapped phase (rad)",
                showlegend=True,
            )
            fig.show()

        # return the phase shift
        phase_shift = coefficients[0]
        return phase_shift

    def scan_resonator_frequencies(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        amplitude: float | None = None,
        phase_shift: float | None = None,
        subrange_width: float = 0.3,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict:
        read_label = Target.read_label(target)
        qubit_label = Target.qubit_label(target)
        mux = self.experiment_system.get_mux_by_qubit(qubit_label)
        read_box = self.experiment_system.get_readout_box_for_qubit(qubit_label)

        if frequency_range is None:
            if read_box.type == BoxType.QUEL1SE_R8:
                frequency_range = np.arange(5.75, 6.75, 0.002)
            else:
                frequency_range = np.arange(9.75, 10.75, 0.002)
        else:
            frequency_range = np.array(frequency_range)

        if amplitude is None:
            amplitude = self.params.readout_amplitude[qubit_label]

        # measure phase shift if not provided
        if phase_shift is None:
            phase_shift = self.measure_phase_shift(
                target,
                frequency_range=frequency_range[0:30],
                amplitude=amplitude,
                shots=shots,
                interval=interval,
                plot=plot,
            )

        # split frequency range to avoid the frequency sweep range limit
        frequency_range = np.array(frequency_range)
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=subrange_width,
        )

        bounds = [
            subranges[0][0],
        ] + [subrange[-1] for subrange in subranges]

        phases: list[float] = []
        signals: list[complex] = []

        idx = 0
        phase_offset = 0.0

        if read_box.type == BoxType.QUEL1SE_R8:
            ssb = "L"
            cnco_center = CNCO_CETNER_READ_R8
        else:
            ssb = "U"
            cnco_center = CNCO_CETNER_READ

        for subrange in subranges:
            f_center = (subrange[0] + subrange[-1]) / 2
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_center * 1e9,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            with self.state_manager.modified_device_settings(
                label=read_label,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                for sub_idx, freq in enumerate(subrange):
                    if idx > 0 and sub_idx == 0:
                        prev_freq = frequency_range[idx - 1]
                        with self.modified_frequencies({read_label: prev_freq}):
                            new_result = self.measure(
                                {qubit_label: np.zeros(0)},
                                mode="avg",
                                readout_amplitudes={qubit_label: amplitude},
                                shots=shots,
                                interval=interval,
                            )
                            new_signal = new_result.data[target].kerneled
                            new_phase = np.angle(new_signal) - prev_freq * phase_shift
                            phase_offset = new_phase - phases[-1]  # type: ignore

                    with self.modified_frequencies({read_label: freq}):
                        result = self.measure(
                            {qubit_label: np.zeros(0)},
                            mode="avg",
                            readout_amplitudes={qubit_label: amplitude},
                            shots=shots,
                            interval=interval,
                        )

                        raw = result.data[target].kerneled
                        ampl = np.abs(raw)
                        phase = np.angle(raw)
                        phase = phase - freq * phase_shift - phase_offset
                        signal = ampl * np.exp(1j * phase)
                        signals.append(signal)
                        phases.append(phase)  # type: ignore

                        idx += 1

        phases_unwrap = np.unwrap(phases)
        phases_unwrap -= phases_unwrap[0]
        phases_diff = np.gradient(phases_unwrap)
        phases_diff -= np.median(phases_diff)
        phases_diff_std = np.std(phases_diff)
        peaks, _ = find_peaks(
            np.abs(phases_diff),
            height=phases_diff_std * 2,
            distance=10,
        )
        peak_freqs = frequency_range[peaks]

        fig1 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
        fig1.add_scatter(
            name=target,
            mode="markers+lines",
            row=1,
            col=1,
            x=frequency_range,
            y=phases_unwrap,
        )
        fig1.add_scatter(
            name=target,
            mode="markers+lines",
            row=2,
            col=1,
            x=frequency_range,
            y=np.abs(signals),
        )
        for bound in bounds:
            fig1.add_vline(
                x=bound,
                line_width=1,
                line_color="black",
                line_dash="dot",
                opacity=0.1,
            )
        fig1.update_xaxes(title_text="Readout frequency (GHz)", row=2, col=1)
        fig1.update_yaxes(title_text="Unwrapped phase (rad)", row=1, col=1)
        fig1.update_yaxes(title_text="Amplitude (arb. units)", row=2, col=1)
        fig1.update_layout(
            title=dict(
                text=f"Resonator frequency scan : {mux.label}",
                subtitle=dict(
                    text=f"readout_amplitude={amplitude}",
                    font=dict(
                        size=13,
                        family="monospace",
                    ),
                ),
            ),
            height=450,
            margin=dict(t=80),
            showlegend=False,
        )

        fig2 = go.Figure()
        fig2.add_scatter(
            name=target,
            mode="markers+lines",
            x=frequency_range,
            y=phases_diff,
        )
        for bound in bounds:
            fig2.add_vline(
                x=bound,
                line_width=1,
                line_color="black",
                line_dash="dot",
                opacity=0.1,
            )
        for freq in peak_freqs:
            fig2.add_vline(
                x=freq,
                line_width=2,
                line_color="red",
                opacity=0.6,
            )
            fig2.add_annotation(
                x=freq,
                y=0.03,
                xref="x",
                yref="paper",
                xanchor="left",
                xshift=3,
                yanchor="bottom",
                text=f"<b>{freq:.3f}</b>",
                showarrow=False,
                font=dict(
                    size=9,
                    color="red",
                    family="sans-serif",
                ),
                opacity=0.6,
                bgcolor="rgba(255, 255, 255, 0.8)",
            )

        fig2.update_layout(
            title=dict(
                text=f"Resonator frequency scan : {mux.label}",
                subtitle=dict(
                    text=f"readout_amplitude={amplitude}",
                    font=dict(
                        size=13,
                        family="monospace",
                    ),
                ),
            ),
            xaxis_title="Readout frequency (GHz)",
            yaxis_title="Phase diff (rad)",
            margin=dict(t=80),
        )

        if plot:
            fig1.show()
            fig2.show()
            print("Found peaks:")
            for peak in peak_freqs:
                print(f"  {peak:.4f}")

        if save_image:
            viz.save_figure_image(
                fig1,
                name=f"resonator_frequency_scan_{mux.label}_phase",
                width=600,
                height=450,
            )
            viz.save_figure_image(
                fig2,
                name=f"resonator_frequency_scan_{mux.label}_phase_diff",
            )

        return {
            "peaks": peak_freqs,
            "frequency_range": frequency_range,
            "subranges": subranges,
            "signals": np.array(signals),
            "phases_diff": phases_diff,
            "fig_phase": fig1,
            "fig_phase_diff": fig2,
        }

    def resonator_spectroscopy(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        power_range: ArrayLike = np.arange(-60, 5, 5),
        phase_shift: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        power_range = np.array(power_range)
        qubit_label = Target.qubit_label(target)
        mux = self.experiment_system.get_mux_by_qubit(qubit_label)
        read_box = self.experiment_system.get_readout_box_for_qubit(qubit_label)

        if frequency_range is None:
            if read_box.type == BoxType.QUEL1SE_R8:
                frequency_range = np.arange(5.75, 6.75, 0.002)
            else:
                frequency_range = np.arange(9.75, 10.75, 0.002)
        else:
            frequency_range = np.array(frequency_range)

        # measure phase shift if not provided
        if phase_shift is None:
            phase_shift = self.measure_phase_shift(
                target,
                frequency_range=frequency_range[0:30],
                shots=shots,
                interval=interval,
                plot=False,
            )

        result = []
        for power in tqdm(power_range):
            power_linear = 10 ** (power / 10)
            amplitude = np.sqrt(power_linear)
            phase_diff = self.scan_resonator_frequencies(
                target,
                frequency_range=frequency_range,
                phase_shift=phase_shift,
                amplitude=amplitude,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )["phases_diff"]
            phase_diff = np.abs(phase_diff)
            phase_diff = np.append(phase_diff, phase_diff[-1])
            result.append(phase_diff)

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=frequency_range,
                y=power_range,
                z=result,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(
                        text="Phase shift (rad)",
                        side="right",
                    ),
                ),
            )
        )
        fig.update_layout(
            title=f"Resonator spectroscopy : {mux.label}",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Power (dB)",
            width=600,
            height=300,
        )

        if plot:
            fig.show()

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"resonator_spectroscopy_{mux.label}",
            )

        return {
            "frequency_range": frequency_range,
            "power_range": power_range,
            "data": np.array(result),
            "fig": fig,
        }

    def measure_reflection_coefficient(
        self,
        target: str,
        *,
        frequency_range: ArrayLike,
        amplitude: float | None = None,
        phase_shift: float,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        freq_range = np.array(frequency_range)
        center_frequency = np.mean(freq_range).astype(float)
        frequency_width = (freq_range[-1] - freq_range[0]).astype(float)

        if frequency_width > 0.4:
            raise ValueError("Frequency scan range must be less than 400 MHz.")

        read_label = Target.read_label(target)
        qubit_label = Target.qubit_label(target)
        read_box = self.experiment_system.get_readout_box_for_qubit(qubit_label)

        if amplitude is None:
            amplitude = self.params.readout_amplitude[qubit_label]

        if read_box.type == BoxType.QUEL1SE_R8:
            ssb = "L"
            cnco_center = CNCO_CETNER_READ_R8
        else:
            ssb = "U"
            cnco_center = CNCO_CETNER_READ

        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            center_frequency * 1e9,
            ssb=ssb,
            cnco_center=cnco_center,
        )

        signals = []

        with self.state_manager.modified_device_settings(
            label=read_label,
            lo_freq=lo,
            cnco_freq=cnco,
            fnco_freq=0,
        ):
            for freq in freq_range:
                with self.modified_frequencies({read_label: freq}):
                    result = self.measure(
                        {qubit_label: np.zeros(0)},
                        mode="avg",
                        readout_amplitudes={qubit_label: amplitude},
                        shots=shots,
                        interval=interval,
                    )
                    signal = result.data[target].kerneled
                    signal = signal * np.exp(-1j * freq * phase_shift)
                    signals.append(signal)

        phi = (np.angle(signals[0]) + np.angle(signals[-1])) / 2
        coeffs = np.array(signals) * np.exp(-1j * phi)

        fit_result = fitting.fit_reflection_coefficient(
            target=target,
            freq_range=freq_range,
            data=coeffs,
            plot=plot,
        )

        fig = fit_result["fig"]

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"reflection_coefficient_{target}",
                width=800,
                height=450,
            )

        return {
            "frequency_range": freq_range,
            "reflection_coefficients": coeffs,
            **fit_result,
        }

    def scan_qubit_frequencies(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        control_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        readout_frequency: float | None = None,
        subrange_width: float = 0.3,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict:
        # control and readout pulses
        qubit = Target.qubit_label(target)
        resonator = Target.read_label(target)
        ctrl_box = self.experiment_system.get_control_box_for_qubit(qubit)

        if control_amplitude is None:
            control_amplitude = self.params.control_amplitude[qubit]

        if readout_amplitude is None:
            readout_amplitude = self.params.readout_amplitude[qubit]

        control_pulse = Gaussian(
            duration=1024,
            amplitude=control_amplitude,
            sigma=128,
        )
        readout_pulse = FlatTop(
            duration=1024,
            amplitude=readout_amplitude,
            tau=128,
        )

        # split frequency range to avoid the frequency sweep range limit
        if frequency_range is None:
            if ctrl_box.type == BoxType.QUEL1SE_R8:
                frequency_range = np.arange(3.0, 5.0, 0.002)
            else:
                frequency_range = np.arange(6.5, 9.5, 0.002)
        else:
            frequency_range = np.array(frequency_range)
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=subrange_width,
        )

        bounds = [
            subranges[0][0],
        ] + [subrange[-1] for subrange in subranges]

        # readout frequency
        readout_frequency = readout_frequency or self.targets[resonator].frequency

        # result buffer
        signals = []
        phases = []
        amplitudes = []

        if ctrl_box.type == BoxType.QUEL1SE_R8:
            ssb = None
            cnco_center = CNCO_CENTER_CTRL
        else:
            ssb = "L"
            cnco_center = CNCO_CENTER_CTRL

        # measure the phase and amplitude
        idx = 0
        for subrange in subranges:
            f_center = (subrange[0] + subrange[-1]) / 2
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_center * 1e9,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            with self.state_manager.modified_device_settings(
                label=qubit,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                for control_frequency in subrange:
                    with self.modified_frequencies(
                        {
                            qubit: control_frequency,
                            resonator: readout_frequency,
                        }
                    ):
                        with PulseSchedule([qubit, resonator]) as ps:
                            ps.add(qubit, control_pulse)
                            ps.add(resonator, readout_pulse)
                        result = self.execute(
                            schedule=ps,
                            mode="avg",
                            shots=shots,
                            interval=interval,
                        )
                        signal = result.data[qubit][0].kerneled
                        signals.append(signal)
                        phases.append(np.angle(signal))
                        amplitudes.append(np.abs(signal))

                        idx += 1

        phases_unwrap = np.unwrap(phases)
        phases_unwrap -= np.median(phases_unwrap)
        phases_unwrap_std = np.std(phases_unwrap)
        peaks, _ = find_peaks(
            np.abs(phases_unwrap),
            height=phases_unwrap_std * 3,
            distance=10,
        )
        peak_freqs = frequency_range[peaks]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
        fig.add_scatter(
            name=target,
            mode="markers+lines",
            row=1,
            col=1,
            x=frequency_range,
            y=phases_unwrap,
        )
        fig.add_scatter(
            name=target,
            mode="markers+lines",
            row=2,
            col=1,
            x=frequency_range,
            y=amplitudes,
        )
        for bound in bounds:
            fig.add_vline(
                x=bound,
                line_width=1,
                line_color="black",
                line_dash="dot",
                opacity=0.1,
            )
        for freq in peak_freqs:
            fig.add_vline(
                x=freq,
                line_width=2,
                line_color="red",
                opacity=0.6,
            )
            fig.add_annotation(
                x=freq,
                y=0.01,
                xref="x",
                yref="paper",
                xanchor="left",
                xshift=3,
                yanchor="bottom",
                text=f"<b>{freq:.3f}</b>",
                showarrow=False,
                font=dict(
                    size=9,
                    color="red",
                    family="sans-serif",
                ),
                opacity=0.6,
                bgcolor="rgba(255, 255, 255, 0.8)",
            )

        fig.update_xaxes(title_text="Control frequency (GHz)", row=2, col=1)
        fig.update_yaxes(title_text="Unwrapped phase (rad)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (arb. units)", row=2, col=1)
        fig.update_layout(
            title=dict(
                text=f"Control frequency scan : {qubit}",
                subtitle=dict(
                    text=f"control_amplitude={control_amplitude}, readout_amplitude={readout_amplitude}",
                    font=dict(size=11, family="monospace"),
                ),
            ),
            height=450,
            margin=dict(t=80),
            showlegend=False,
        )

        f_ge = None
        f_gf = None
        f_ef = None
        anharmonicity = None

        if plot:
            fig.show()
            n_peaks = len(peak_freqs)
            if n_peaks > 0:
                print("Found peaks:")
                for peak in peak_freqs:
                    print(f"  {peak:.4f}")
                print("Frequency guess:")
                f_ge = peak_freqs[-1]
                print(f"  f_ge: {f_ge:.4f} GHz")
            if n_peaks > 1:
                f_gf = peak_freqs[-2]
                print(f"  f_gf: {f_gf:.4f} GHz")
                anharmonicity = (f_gf - f_ge) * 2
            if n_peaks > 2:
                f_ef = peak_freqs[-3]
                print(f"  f_ef: {f_ef:.4f} GHz")
                anharmonicity = f_ef - f_ge
            if anharmonicity is not None:
                print(f"  anharmonicity: {anharmonicity:.4f} GHz")

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"qubit_frequency_scan_{qubit}",
                width=600,
                height=450,
            )

        return {
            "peaks": peak_freqs,
            "frequency_guess": {
                "f_ge": f_ge,
                "f_gf": f_gf,
                "f_ef": f_ef,
                "anharmonicity": anharmonicity,
            },
            "frequency_range": frequency_range,
            "subranges": subranges,
            "readout_amplitude": readout_amplitude,
            "signals": np.array(signals),
            "phases": phases_unwrap,
            "amplitudes": np.asarray(amplitudes),
            "fig": fig,
        }

    def estimate_control_amplitude(
        self,
        target: str,
        *,
        frequency_range: ArrayLike,
        control_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        target_rabi_rate: float = RABI_FREQUENCY,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ):
        frequency_range = np.asarray(frequency_range)
        qubit_label = Target.qubit_label(target)
        if control_amplitude is None:
            control_amplitude = self.params.control_amplitude[qubit_label]
        if readout_amplitude is None:
            readout_amplitude = self.params.readout_amplitude[qubit_label]

        data = self.scan_qubit_frequencies(
            target,
            frequency_range=frequency_range,
            control_amplitude=control_amplitude,
            readout_amplitude=readout_amplitude,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )["phases"]
        result = fitting.fit_sqrt_lorentzian(
            target=target,
            x=frequency_range,
            y=data,
            plot=plot,
            title="Qubit resonance fit",
        )
        rabi_rate = result["Omega"]
        estimated_amplitude = target_rabi_rate / rabi_rate * control_amplitude

        if plot:
            fig = result["fig"]
            fig.update_layout(
                title=dict(
                    text=f"Control amplitude estimation : {target}",
                    subtitle=dict(
                        text=f"readout_amplitude={readout_amplitude}",
                        font=dict(size=13, family="monospace"),
                    ),
                ),
                xaxis_title="Control frequency (GHz)",
                yaxis_title="Unwrapped phase (rad)",
                width=600,
                height=300,
                margin=dict(t=80),
            )
            fig.show()

            print("")
            print(f"Control amplitude estimation : {target}")
            print(f"  {control_amplitude:.6f} -> {rabi_rate * 1e3:.3f} MHz")
            print(f"  {estimated_amplitude:.6f} -> {target_rabi_rate * 1e3:.3f} MHz")

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"control_amplitude_estimation_{target}",
                width=600,
                height=300,
            )
        return estimated_amplitude

    def qubit_spectroscopy(
        self,
        target: str,
        frequency_range: ArrayLike | None = None,
        power_range: ArrayLike = np.arange(-60, 5, 5),
        readout_amplitude: float | None = None,
        readout_frequency: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        power_range = np.array(power_range)
        result2d = []
        for power in tqdm(power_range):
            power_linear = 10 ** (power / 10)
            amplitude = np.sqrt(power_linear)
            result1d = self.scan_qubit_frequencies(
                target,
                frequency_range=frequency_range,
                control_amplitude=amplitude,
                readout_amplitude=readout_amplitude,
                readout_frequency=readout_frequency,
                shots=shots,
                interval=interval,
                plot=False,
            )
            phase = result1d["phases"]
            result2d.append(phase)

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=result1d["frequency_range"],
                y=power_range,
                z=result2d,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(
                        text="Phase (rad)",
                        side="right",
                    )
                ),
            )
        )
        fig.update_layout(
            title=dict(
                text=f"Qubit spectroscopy : {target}",
                subtitle=dict(
                    text=f"readout_amplitud={result1d['readout_amplitude']}",
                    font=dict(
                        size=13,
                        family="monospace",
                    ),
                ),
            ),
            xaxis_title="Frequency (GHz)",
            yaxis_title="Power (dB)",
            width=600,
            height=300,
            margin=dict(t=80),
        )
        if plot:
            fig.show()

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"qubit_spectroscopy_{target}",
                width=600,
                height=300,
            )

        return {
            "frequency_range": result1d["frequency_range"],
            "power_range": power_range,
            "data": np.array(result2d),
            "fig": fig,
        }
