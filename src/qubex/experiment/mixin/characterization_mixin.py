from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from typing import Collection, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from rich.prompt import Confirm
from scipy.signal import find_peaks
from tqdm import tqdm
from typing_extensions import deprecated

from ...analysis import fitting
from ...analysis import visualization as viz
from ...backend import BoxType, MixingUtil, Target
from ...backend.experiment_system import (
    CNCO_CENTER_CTRL,
    CNCO_CETNER_READ,
    CNCO_CETNER_READ_R8,
)
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS, SAMPLING_PERIOD
from ...pulse import (
    CPMG,
    Blank,
    FlatTop,
    Gaussian,
    PulseSchedule,
    RampType,
    Rect,
    VirtualZ,
    Waveform,
)
from ...style import COLORS
from ...typing import TargetMap
from ..experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_RABI_FREQUENCY,
    DEFAULT_RABI_TIME_RANGE,
)
from ..experiment_result import (
    SweepData,
    AmplRabiData,
    ExperimentResult,
    FreqRabiData,
    RabiData,
    RamseyData,
    T1Data,
    T2Data,
)
from ..experiment_util import ExperimentUtil
from ..protocol import (
    BaseProtocol,
    CalibrationProtocol,
    CharacterizationProtocol,
    MeasurementProtocol,
)
from ..rabi_param import RabiParam
from ..result import Result

logger = logging.getLogger(__name__)


class CharacterizationMixin(
    BaseProtocol,
    MeasurementProtocol,
    CalibrationProtocol,
    CharacterizationProtocol,
):
    def measure_readout_snr(
        self,
        targets: Collection[str] | str | None = None,
        *,
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> Result:
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

        return Result(
            data={
                "signal": signal,
                "noise": noise,
                "snr": snr,
            }
        )

    def sweep_readout_amplitude(
        self,
        targets: Collection[str] | str | None = None,
        *,
        amplitude_range: ArrayLike | None = None,
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        readout_duration: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if amplitude_range is None:
            amplitude_range = np.linspace(0.0, 0.25, 51)
        else:
            amplitude_range = np.asarray(amplitude_range)

        signal_buf = defaultdict(list)
        noise_buf = defaultdict(list)
        snr_buf = defaultdict(list)

        for amplitude in tqdm(amplitude_range):
            result = self.measure_readout_snr(
                targets=targets,
                initial_state=initial_state,
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
        figs = {}

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
                figs[target] = fig

        return Result(
            data={
                "signal": signal,
                "noise": noise,
                "snr": snr,
                "fig": figs,
            }
        )

    def sweep_readout_duration(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(128, 2048, 128),
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
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

        return Result(
            data={
                "signal": signal,
                "noise": noise,
                "snr": snr,
            }
        )

    def chevron_pattern(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.05, 0.05, 51),
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        rabi_params: dict[str, RabiParam] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
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
                fit_threshold=0.0,
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
        figs = {}
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
                        rabi_rates_buffer[target].append(
                            fit_result.get("frequency", np.nan)
                        )
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
                            text=f"control_amplitude={amplitudes[target]:.6g}",
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
                figs[target] = fig
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

        return Result(
            data={
                "time_range": time_range,
                "detuning_range": detuning_range,
                "frequencies": frequencies,
                "chevron_data": chevron_data,
                "rabi_rates": rabi_rates,
                "resonant_frequencies": resonant_frequencies,
                "fig": figs,
            }
        )

    def obtain_freq_rabi_relation(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike | None = None,
        time_range: ArrayLike | None = None,
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        verbose: bool = False,
    ) -> ExperimentResult[FreqRabiData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if detuning_range is None:
            detuning_range = np.linspace(-0.01, 0.01, 21)
        else:
            detuning_range = np.asarray(detuning_range, dtype=np.float64)

        if time_range is None:
            time_range = np.arange(0, 101, 4)
        else:
            time_range = np.asarray(time_range, dtype=np.float64)

        amplitudes = {
            target: self.params.get_control_amplitude(target) for target in targets
        }
        rabi_rates: dict[str, list[float]] = defaultdict(list)
        rabi_data: dict[str, list[RabiData]] = defaultdict(list)

        for detuning in tqdm(detuning_range):
            if rabi_level == "ge":
                rabi_result = self.rabi_experiment(
                    time_range=time_range,
                    amplitudes=amplitudes,
                    detuning=detuning,
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
            elif rabi_level == "ef":
                rabi_result = self.ef_rabi_experiment(
                    time_range=time_range,
                    amplitudes=amplitudes,
                    detuning=detuning,
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
            else:
                raise ValueError("Invalid rabi_level.")
            if verbose:
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
        time_range: ArrayLike = np.arange(0, 201, 4),
        amplitude_range: ArrayLike = np.linspace(0.01, 0.1, 10),
        ramptime: float | None = None,
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

        if ramptime is None:
            ramptime = 0

        time_range = np.array(time_range, dtype=np.float64)
        amplitude_range = np.array(amplitude_range, dtype=np.float64)
        rabi_rates: dict[str, list[float]] = defaultdict(list)
        rabi_data: dict[str, list[RabiData]] = defaultdict(list)

        for amplitude in tqdm(amplitude_range):
            rabi_result = self.rabi_experiment(
                amplitudes={target: amplitude for target in targets},
                time_range=time_range,
                ramptime=ramptime,
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
    ) -> Result:
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
        return Result(data=resonant_frequencies)

    def calibrate_ef_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike | None = None,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        verbose: bool = True,
    ) -> Result:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        targets = [
            target for target in targets if Target.ef_label(target) in self.targets
        ]
        if len(targets) == 0:
            print("No ef targets found for the given targets.")
            return Result(data={})

        if detuning_range is None:
            detuning_range = np.linspace(-0.05, 0.05, 21)
        else:
            detuning_range = np.asarray(detuning_range, dtype=np.float64)

        if time_range is None:
            time_range = np.arange(0, 101, 4)
        else:
            time_range = np.asarray(time_range, dtype=np.float64)

        result = self.obtain_freq_rabi_relation(
            targets=targets,
            detuning_range=detuning_range,
            rabi_level="ef",
            time_range=time_range,
            shots=shots,
            interval=interval,
            plot=False,
            verbose=verbose,
        )
        fit_data = {
            target: data.fit()["f_resonance"] for target, data in result.data.items()
        }

        if plot:
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

        return Result(data=fit_data)

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
    ) -> Result:
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
        figs = {}
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

            if "fig" in fit_result:
                figs[target] = fit_result["fig"]

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

        return Result(data={"data": fit_data, "fig": figs})

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
                np.log10(200 * 1000),
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
                        ps.add(target, self.get_hpi_pulse(target).repeated(2))
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
        n_cpmg: int | None = 1,
        pi_cpmg: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
        xaxis_type: Literal["linear", "log"] = "log",
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
                np.log10(200 * 1000),
                51,
            )

        if n_cpmg is not None:
            time_range = self.util.discretize_time_range(
                time_range=np.asarray(time_range),
                sampling_period=2 * SAMPLING_PERIOD * n_cpmg,
            )
        else:
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
                        hpi = self.get_hpi_pulse(target)
                        pi = pi_cpmg or hpi.repeated(2).shifted(np.pi / 2)
                        ps.add(target, hpi)
                        if n_cpmg is not None:
                            total_blank = T - pi.duration * n_cpmg
                            if total_blank > 0:
                                tau = total_blank // (2 * n_cpmg)
                                ps.add(
                                    target,
                                    CPMG(
                                        tau=tau,
                                        pi=pi,
                                        n=n_cpmg,
                                    ),
                                )
                            else:
                                ps.add(target, Blank(T))
                        else:
                            tau = pi.duration * 5
                            cpmg = CPMG(
                                tau=tau,
                                pi=pi,
                                n=2,
                            )
                            n_repeats = int(T // cpmg.duration)
                            remainder = T % cpmg.duration
                            if n_repeats > 0:
                                ps.add(target, cpmg.repeated(n_repeats))
                            if remainder > 0:
                                ps.add(target, Blank(remainder))
                        ps.add(target, hpi.scaled(-1))
                return ps

            print(
                f"({idx + 1}/{len(subgroups)}) Conducting T2 experiment for {subgroup}...\n"
            )

            # if plot:
            #     t2_sequence(time_range[-1]).plot()

            sweep_result = self.sweep_parameter(
                sequence=t2_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
                xaxis_type=xaxis_type,
            )

            for target, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_exp_decay(
                    target=target,
                    x=sweep_data.sweep_range,
                    y=0.5 * (1 + sweep_data.normalized),
                    plot=plot,
                    title="T2 echo",
                    xlabel="Time (μs)",
                    ylabel="Normalized signal",
                    xaxis_type=xaxis_type,
                    yaxis_type="linear",
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
        time_range: ArrayLike | None = None,
        detuning: float | None = None,
        second_rotation_axis: Literal["X", "Y"] = "Y",
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = CALIBRATION_SHOTS,
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

        if time_range is None:
            time_range = np.arange(0, 10001, 100)
        else:
            time_range = self.util.discretize_time_range(time_range)

        if detuning is None:
            detuning = 0.001

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
                        x90 = self.get_hpi_pulse(target)
                        ps.add(target, x90)
                        ps.add(target, Blank(T))
                        if second_rotation_axis == "X":
                            ps.add(target, x90.shifted(np.pi))
                        else:
                            ps.add(target, x90.shifted(-np.pi / 2))
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
                        amplitude_est=1.0,
                        offset_est=0.0,
                        plot=plot,
                    )
                    if fit_result["status"] == "success":
                        f = self.qubits[target].frequency
                        t2 = fit_result["tau"]
                        ramsey_freq = fit_result["f"]
                        phi = fit_result["phi"]
                        if second_rotation_axis == "Y":
                            if phi > 0:
                                bare_freq = f + detuning + ramsey_freq
                            else:
                                bare_freq = f + detuning - ramsey_freq
                        else:
                            # NOTE: For X rotation, we cannot guarantee the sign of frequency
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

    def _simultaneous_measurement_coherence(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        detuning: float | None = None,
        second_rotation_axis: Literal["X", "Y"] = "Y",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict[str, ExperimentResult]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if time_range is None:
            time_range = np.arange(0, 50_001, 1000)

        time_range = self.util.discretize_time_range(
            time_range=np.asarray(time_range),
            sampling_period=2 * SAMPLING_PERIOD,
        )

        if detuning is None:
            detuning = 0.001

        self.validate_rabi_params(targets)

        modes = ("T1", "T2", "Ramsey")

        signals: dict[str, defaultdict[str, list[float]]] = {
            mode: defaultdict(list) for mode in modes
        }

        data_t1: dict[str, T1Data] = {}
        data_t2: dict[str, T2Data] = {}
        data_ramsey: dict[str, RamseyData] = {}

        x90_pulses = {target: self.get_hpi_pulse(target) for target in targets}

        def t1_sequence(target, T: int) -> PulseSchedule:
            with PulseSchedule([target]) as ps:
                ps.add(target, x90_pulses[target].repeated(2))
                ps.add(target, Blank(T))
            return ps

        def t2_sequence(target, T: int) -> PulseSchedule:
            half_T = T // 2
            with PulseSchedule([target]) as ps:
                ps.add(target, x90_pulses[target])
                ps.add(target, Blank(half_T))
                ps.add(target, x90_pulses[target].repeated(2).shifted(np.pi / 2))
                ps.add(target, Blank(half_T))
                ps.add(target, x90_pulses[target].scaled(-1))
            return ps

        def ramsey_sequence(target, T: int) -> PulseSchedule:
            with PulseSchedule([target]) as ps:
                x90 = x90_pulses[target]
                ps.add(target, x90)
                ps.add(target, Blank(T))
                if second_rotation_axis == "X":
                    ps.add(target, x90.shifted(np.pi))
                else:
                    ps.add(target, x90.shifted(-np.pi / 2))
            return ps

        for target in targets:
            for T in time_range:
                t1_schedules = t1_sequence(target, T)
                t2_schedules = t2_sequence(target, T)
                ramsey_schedules = ramsey_sequence(target, T)

                detuned_frequencies = {
                    target: self.qubits[target].frequency + detuning
                    for target in targets
                }
                measurements = {
                    "T1": self.measure(
                        sequence=t1_schedules,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    ),
                    "T2": self.measure(
                        sequence=t2_schedules,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    ),
                    "Ramsey": self.measure(
                        sequence=ramsey_schedules,
                        frequencies=detuned_frequencies,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    ),
                }
                for mode in modes:
                    for measured_target, data in measurements[mode].data.items():
                        signals[mode][measured_target].append(data.kerneled)

        sweep_data: dict[str, dict[str, SweepData]] = {
            mode: {
                target: SweepData(
                    target=target,
                    data=np.asarray(values),
                    sweep_range=time_range,
                    rabi_param=self.rabi_params.get(target),
                    state_centers=self.state_centers.get(target),
                    title="Sweep result",
                    xlabel="Sweep value",
                    ylabel="Measured value",
                    xaxis_type="linear",
                    yaxis_type="linear",
                )
                for target, values in signals[mode].items()
            }
            for mode in modes
        }

        for target, sweep_result in sweep_data["T1"].items():
            fit_result_t1 = fitting.fit_exp_decay(
                target=target,
                x=sweep_result.sweep_range,
                y=0.5 * (1 - sweep_result.normalized),
                plot=plot,
                title="T1",
                xlabel="Time (μs)",
                ylabel="Normalized signal",
                xaxis_type="linear",
            )
            if fit_result_t1["status"] == "success":
                t1 = fit_result_t1["tau"]
                t1_err = fit_result_t1["tau_err"]
                r2 = fit_result_t1["r2"]

                t1_data = T1Data.new(
                    sweep_result,
                    t1=t1,
                    t1_err=t1_err,
                    r2=r2,
                )
                data_t1[target] = t1_data

                fig = fit_result_t1["fig"]

                if save_image:
                    viz.save_figure_image(
                        fig,
                        name=f"t1_{target}",
                    )

        for target, sweep_result in sweep_data["T2"].items():
            fit_result_t2 = fitting.fit_exp_decay(
                target=target,
                x=sweep_result.sweep_range,
                y=0.5 * (1 + sweep_result.normalized),
                plot=plot,
                title="T2 echo",
                xlabel="Time (μs)",
                ylabel="Normalized signal",
                xaxis_type="linear",
            )
            if fit_result_t2["status"] == "success":
                t2 = fit_result_t2["tau"]
                t2_err = fit_result_t2["tau_err"]
                r2 = fit_result_t2["r2"]

                t2_data = T2Data.new(
                    sweep_result,
                    t2=t2,
                    t2_err=t2_err,
                    r2=r2,
                )
                data_t2[target] = t2_data

                fig = fit_result_t2["fig"]

                if save_image:
                    viz.save_figure_image(
                        fig,
                        name=f"t2_echo_{target}",
                    )

        for target, sweep_result in sweep_data["Ramsey"].items():
            fit_result_ramsey = fitting.fit_ramsey(
                target=target,
                times=sweep_result.sweep_range,
                data=sweep_result.normalized,
                amplitude_est=1.0,
                offset_est=0.0,
                plot=plot,
            )
            if fit_result_ramsey["status"] == "success":
                f = self.qubits[target].frequency
                t2 = fit_result_ramsey["tau"]
                ramsey_freq = fit_result_ramsey["f"]
                phi = fit_result_ramsey["phi"]
                if second_rotation_axis == "Y":
                    if phi > 0:
                        bare_freq = f + detuning + ramsey_freq
                    else:
                        bare_freq = f + detuning - ramsey_freq
                else:
                    bare_freq = f + detuning - ramsey_freq
                r2 = fit_result_ramsey["r2"]
                ramsey_data = RamseyData.new(
                    sweep_result,
                    t2=t2,
                    ramsey_freq=ramsey_freq,
                    bare_freq=bare_freq,
                    r2=r2,
                )
                data_ramsey[target] = ramsey_data

                fig = fit_result_ramsey["fig"]

                if save_image:
                    viz.save_figure_image(
                        fig,
                        name=f"ramsey_{target}",
                    )

        exp_t1 = ExperimentResult(data=data_t1)
        exp_t2 = ExperimentResult(data=data_t2)
        exp_ramsey = ExperimentResult(data=data_ramsey)

        return {
            "T1": exp_t1,
            "T2": exp_t2,
            "Ramsey": exp_ramsey,
        }

    def _stark_t1_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        stark_detuning: float | dict[str, float] | None = None,
        stark_amplitude: float | dict[str, float] | None = None,
        stark_ramptime: float | dict[str, float] | None = None,
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

        if stark_detuning is None:
            stark_detuning = {target: 0.15 for target in targets}
        elif isinstance(stark_detuning, float):
            detuning = stark_detuning
            if abs(detuning) > 0.2:
                raise ValueError(
                    "Detuning of a stark tone must not exceed 0.2 GHz: the guard-banded AWG baseband limit."
                )
            stark_detuning = {target: detuning for target in targets}
        else:
            for target in targets:
                detuning = stark_detuning[target]
                if abs(detuning) > 0.2:
                    raise ValueError(
                        "Detuning of a stark tone must not exceed 0.2 GHz: the guard-banded AWG baseband limit."
                    )

        if stark_amplitude is None:
            stark_amplitude = {target: 0.1 for target in targets}
        elif isinstance(stark_amplitude, float):
            stark_amplitude = {target: stark_amplitude for target in targets}

        if stark_ramptime is None:
            stark_ramptime = {target: 10 for target in targets}
        elif isinstance(stark_ramptime, float):
            stark_ramptime = {target: stark_ramptime for target in targets}

        self.validate_rabi_params(targets)

        if time_range is None:
            time_range = np.logspace(
                np.log10(100),
                np.log10(200 * 1000),
                51,
            )
        time_range = self.util.discretize_time_range(np.asarray(time_range))

        data: dict[str, T1Data] = {}

        for target in targets:
            power = self.calc_control_amplitude(
                target=target, rabi_rate=stark_amplitude[target]
            )
            if power > 1:
                raise ValueError("Drive amplitude of a stark tone must not exceed 1")
            ramptime = stark_ramptime[target]
            detuning = stark_detuning[target]

            def stark_t1_sequence(T: int) -> PulseSchedule:
                with PulseSchedule([target]) as ps:
                    ps.add(target, self.get_hpi_pulse(target).repeated(2))
                    ps.add(
                        target,
                        FlatTop(
                            duration=T + ramptime * 2,
                            amplitude=power,
                            tau=ramptime,
                        ).detuned(detuning=detuning),
                    )
                return ps

            sweep_result = self.sweep_parameter(
                sequence=stark_t1_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
                title="Stark-driven T1 decay",
                xlabel="Time (μs)",
                ylabel="Measured value",
                xaxis_type=xaxis_type,
            )

            for qubit, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_exp_decay(
                    target=qubit,
                    x=sweep_data.sweep_range,
                    y=0.5 * (1 - sweep_data.normalized),
                    plot=plot,
                    title="Stark-driven T1",
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
                    data[qubit] = t1_data

                    fig = fit_result["fig"]

                    if save_image:
                        viz.save_figure_image(
                            fig,
                            name=f"t1_{qubit}",
                        )

        return ExperimentResult(data=data)

    def _stark_ramsey_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        stark_detuning: float | dict[str, float] | None = None,
        stark_amplitude: float | dict[str, float] | None = None,
        stark_ramptime: float | dict[str, float] | None = None,
        time_range: ArrayLike | None = None,
        second_rotation_axis: Literal["X", "Y"] = "Y",
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        envelope_region: Literal["full", "flat"] = "full",
        plot: bool = True,
        save_image: bool = False,
    ) -> ExperimentResult[RamseyData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if stark_detuning is None:
            stark_detuning = {target: 0.15 for target in targets}
        elif isinstance(stark_detuning, float):
            detuning = stark_detuning
            if abs(detuning) > 0.2:
                raise ValueError(
                    "Detuning of a stark tone must not exceed 0.2 GHz: the guard-banded AWG baseband limit."
                )
            stark_detuning = {target: detuning for target in targets}
        else:
            for target in targets:
                detuning = stark_detuning[target]
                if abs(detuning) > 0.2:
                    raise ValueError(
                        "Detuning of a stark tone must not exceed 0.2 GHz: the guard-banded AWG baseband limit."
                    )

        if stark_amplitude is None:
            stark_amplitude = {target: 0.1 for target in targets}
        elif isinstance(stark_amplitude, float):
            stark_amplitude = {target: stark_amplitude for target in targets}

        if stark_ramptime is None:
            stark_ramptime = {target: 10 for target in targets}
        elif isinstance(stark_ramptime, float):
            stark_ramptime = {target: stark_ramptime for target in targets}

        if time_range is None:
            time_range = np.arange(0, 401, 4)
        else:
            time_range = self.util.discretize_time_range(time_range)

        self.validate_rabi_params(targets)

        data: dict[str, RamseyData] = {}

        for target in targets:
            power = self.calc_control_amplitude(
                target=target, rabi_rate=stark_amplitude[target]
            )
            if power > 1:
                raise ValueError("Drive amplitude of a stark tone must not exceed 1")
            ramptime = stark_ramptime[target]
            detuning = stark_detuning[target]

            def stark_ramsey_sequence(T: int) -> PulseSchedule:
                x90 = self.get_hpi_pulse(target=target)
                with PulseSchedule([target]) as ps:
                    ps.add(target, x90)
                    if envelope_region == "full":
                        ps.add(
                            target,
                            FlatTop(
                                duration=T + ramptime * 2,
                                amplitude=power,
                                tau=ramptime,
                            ).detuned(detuning=detuning),
                        )
                        if second_rotation_axis == "X":
                            ps.add(target, x90.shifted(np.pi))
                        else:
                            ps.add(target, x90.shifted(-np.pi / 2))
                    else:
                        ps.add(
                            target,
                            FlatTop(
                                duration=ramptime * 2,
                                amplitude=power,
                                tau=ramptime,
                            ).detuned(detuning=detuning),
                        )
                        ps.add(target, x90.repeated(2))
                        ps.add(
                            target,
                            FlatTop(
                                duration=T + ramptime * 2,
                                amplitude=power,
                                tau=ramptime,
                            ).detuned(detuning=detuning),
                        )
                        if second_rotation_axis == "X":
                            ps.add(target, VirtualZ(theta=-np.pi))
                            ps.add(target, x90)
                        else:
                            ps.add(target, VirtualZ(theta=np.pi / 2))
                            ps.add(target, x90)
                return ps

            sweep_result = self.sweep_parameter(
                sequence=stark_ramsey_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
            )

            for qubit, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_ramsey(
                    target=qubit,
                    times=sweep_data.sweep_range,
                    data=sweep_data.normalized,
                    title="Stark-driven Ramsey fringe",
                    amplitude_est=1.0,
                    offset_est=0.0,
                    plot=plot,
                )
                if fit_result["status"] == "success":
                    f = self.qubits[qubit].frequency
                    t2 = fit_result["tau"]
                    ramsey_freq = fit_result["f"]
                    if stark_detuning[qubit] > 0:
                        dressed_freq = f - ramsey_freq
                    else:
                        dressed_freq = f + ramsey_freq

                    r2 = fit_result["r2"]
                    ramsey_data = RamseyData.new(
                        sweep_data=sweep_data,
                        t2=t2,
                        ramsey_freq=ramsey_freq,
                        bare_freq=dressed_freq,
                        r2=r2,
                    )
                    data[qubit] = ramsey_data

                    sign = 1 if stark_detuning[qubit] > 0 else -1
                    ac_stark_shift = sign * ramsey_data.ramsey_freq

                    print("AC stark shift :")
                    print(f"{qubit}: {ac_stark_shift:.6f}")
                    print("")

                    fig = fit_result["fig"]

                    if save_image:
                        viz.save_figure_image(
                            fig,
                            name=f"stark_ramsey_{qubit}",
                        )

        return ExperimentResult(data=data)

    def obtain_effective_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        detuning: float = 0.001,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        if time_range is None:
            time_range = np.arange(0, 10001, 100)

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

        return Result(
            data={
                "effective_freq": effective_freq,
                "result_0": result_0,
                "result_1": result_1,
            }
        )

    def jazz_experiment(
        self,
        target_qubit: str,
        spectator_qubit: str,
        *,
        time_range: ArrayLike | None = None,
        x90: Waveform | TargetMap[Waveform] | None = None,
        x180: Waveform | TargetMap[Waveform] | None = None,
        second_rotation_axis: Literal["X", "Y"] = "Y",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        rotation_frequency: float = 0.0002,
        plot: bool = True,
    ) -> Result:
        if time_range is None:
            time_range = np.arange(0, 20001, 400)

        if x90 is None:
            x90 = {
                target_qubit: self.get_hpi_pulse(target_qubit),
            }
        elif isinstance(x90, Waveform):
            x90 = {
                target_qubit: x90,
            }

        if x180 is None:
            x180 = {
                target_qubit: self.get_hpi_pulse(target_qubit).repeated(2),
                spectator_qubit: self.get_hpi_pulse(spectator_qubit).repeated(2),
            }
        elif isinstance(x180, Waveform):
            x180 = {
                target_qubit: x180,
                spectator_qubit: x180,
            }

        # Raise an error when rotation_frequency is negative
        if rotation_frequency < 0:
            raise ValueError("rotation_frequency must be non-negative.")

        def jazz_sequence(tau: float) -> PulseSchedule:
            with PulseSchedule([target_qubit, spectator_qubit]) as ps:
                ps.add(target_qubit, x90[target_qubit])
                ps.add(target_qubit, Blank(tau))
                ps.barrier()
                ps.add(target_qubit, x180[target_qubit])
                ps.add(spectator_qubit, x180[spectator_qubit])
                ps.add(target_qubit, Blank(tau))
                if second_rotation_axis == "X":
                    ps.add(
                        target_qubit,
                        x90[target_qubit].shifted(
                            np.pi - rotation_frequency * 2 * tau * 2 * np.pi
                        ),
                    )
                else:
                    ps.add(
                        target_qubit,
                        x90[target_qubit].shifted(
                            -np.pi / 2 - rotation_frequency * 2 * tau * 2 * np.pi
                        ),
                    )
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
            time_range * 2e-3,
            result.data[target_qubit].normalized,
            is_damped=True,
            amplitude_est=1.0,
            offset_est=0.0,
            plot=plot,
            title=f"JAZZ experiment: {target_qubit}-{spectator_qubit}",
            xlabel="Wait time (μs)",
            ylabel=f"Normalized value : {target_qubit}",
        )

        if fit_result["status"] != "success":
            raise RuntimeError("Fitting failed in JAZZ experiment.")

        xi = fit_result["f"] * 1e-3 - rotation_frequency
        zeta = 2 * xi

        print(f"ξ: {xi * 1e6:.2f} kHz")
        print(f"ζ: {zeta * 1e6:.2f} kHz")

        return Result(
            data={
                "xi": xi,
                "zeta": zeta,
                **fit_result,
            }
        )

    def obtain_coupling_strength(
        self,
        target_qubit: str,
        spectator_qubit: str,
        *,
        time_range: ArrayLike | None = None,
        x90: Waveform | TargetMap[Waveform] | None = None,
        x180: Waveform | TargetMap[Waveform] | None = None,
        second_rotation_axis: Literal["X", "Y"] = "Y",
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        rotation_frequency: float = 0.0002,
        plot: bool = True,
    ) -> Result:
        qubit_1 = target_qubit
        qubit_2 = spectator_qubit

        result = self.jazz_experiment(
            target_qubit=qubit_1,
            spectator_qubit=qubit_2,
            time_range=time_range,
            x90=x90,
            x180=x180,
            second_rotation_axis=second_rotation_axis,
            shots=shots,
            interval=interval,
            rotation_frequency=rotation_frequency,
            plot=plot,
        )

        xi = result["xi"]

        f_1 = self.qubits[qubit_1].frequency
        f_2 = self.qubits[qubit_2].frequency

        a_1 = self.qubits[qubit_1].anharmonicity
        a_2 = self.qubits[qubit_2].anharmonicity

        Delta_12 = f_1 - f_2

        g = np.sqrt(np.abs((xi * (Delta_12 + a_1) * (Delta_12 - a_2)) / (a_1 + a_2)))

        print(f"frequency_1: {f_1:.5f} GHz")
        print(f"frequency_2: {f_2:.5f} GHz")
        print(f"Delta_12: {Delta_12 * 1e3:.2f} MHz")
        print(f"anharmonicity_1: {a_1 * 1e3:.2f} MHz")
        print(f"anharmonicity_2: {a_2 * 1e3:.2f} MHz")
        print(f"g: {g * 1e3:.2f} MHz")

        return Result(
            data={
                "g": g,
                **result.data,
            }
        )

    @deprecated("Use `measure_electrical_delay` instead.")
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
        ssb = self.targets[read_label].sideband

        if amplitude is None:
            amplitude = 1.0

        if frequency_range is None:
            if read_box.type == BoxType.QUEL1SE_R8:
                frequency_range = np.arange(5.90, 5.95, 0.001)
            else:
                frequency_range = np.arange(9.90, 9.95, 0.001)
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
            cnco_center = CNCO_CETNER_READ_R8
        else:
            cnco_center = CNCO_CETNER_READ

        for subrange in subranges:
            # change LO/NCO frequency to the center of the subrange
            f_center = (subrange[0] + subrange[-1]) / 2
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_center * 1e9,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            with self.system_manager.modified_device_settings(
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

    def measure_electrical_delay(
        self,
        target: str,
        *,
        f_start: float | None = None,
        df: float | None = None,
        n_samples: int | None = None,
        readout_amplitude: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        confirm: bool = True,
    ) -> float:
        read_label = Target.read_label(target)
        qubit_label = Target.qubit_label(target)
        mux = self.experiment_system.get_mux_by_qubit(qubit_label)
        ssb = self.targets[read_label].sideband
        read_box = self.experiment_system.get_readout_box_for_qubit(qubit_label)
        f_nco = self.targets[read_label].fine_frequency

        if df is None:
            df = 0.0001  # 100 kHz step
        if n_samples is None:
            n_samples = 50
        if readout_amplitude is None:
            readout_amplitude = 1.0
        if f_start is None:
            f_start = f_nco
        frequency_range = np.arange(f_start, f_start + df * n_samples, df)

        def _execute():
            self.reset_awg_and_capunits(box_ids=[read_box.id])
            phases = []
            for freq in frequency_range:
                with self.modified_frequencies({read_label: freq}):
                    result = self.measure(
                        {qubit_label: np.zeros(0)},
                        mode="avg",
                        readout_amplitudes={qubit_label: readout_amplitude},
                        shots=shots,
                        interval=interval,
                        plot=False,
                        reset_awg_and_capunits=False,
                    )
                    signal = result.data[target].kerneled
                    phase = -np.angle(signal)
                    phases.append(phase)
            return phases

        if abs(f_start - f_nco) > 0.2:
            # if the frequency is far from the NCO frequency, we need to change the LO/NCO frequency
            if confirm:
                confirmed = Confirm.ask(
                    "You are about to change the NCO frequencies. Do you want to proceed?"
                )
                if not confirmed:
                    print("Operation cancelled.")
                    return  # type: ignore

            if read_box.type == BoxType.QUEL1SE_R8:
                cnco_center = CNCO_CETNER_READ_R8
            else:
                cnco_center = CNCO_CETNER_READ
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_start * 1e9,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            with self.system_manager.modified_device_settings(
                label=read_label,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                logger.debug(f"LO: {lo}, CNCO: {cnco}")
                phases = _execute()
        else:
            # if the frequency is close to the NCO frequency, we can use the current settings
            phases = _execute()

        unwrapped = np.unwrap(phases)

        x, y = frequency_range, unwrapped
        coefficients = np.polyfit(x, y, 1)
        y_fit = np.polyval(coefficients, x)
        tau = coefficients[0] / (2 * np.pi)

        if plot:
            fig = go.Figure()
            fig.add_scatter(name="data", mode="markers", x=x, y=y)
            fig.add_scatter(name="fit", mode="lines", x=x, y=y_fit)
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.95,
                y=0.05,
                text=f"τ = {tau:.1f} ns",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
            )
            fig.update_layout(
                title=f"Electrical delay : {mux.label}",
                xaxis_title="Frequency (GHz)",
                yaxis_title="Unwrapped phase (rad)",
                showlegend=True,
            )
            fig.show()

        return tau

    def scan_resonator_frequencies(
        self,
        target: str | None = None,
        *,
        frequency_range: ArrayLike | None = None,
        readout_amplitude: float | None = None,
        readout_duration: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        electrical_delay: float | None = None,
        subrange_width: float = 0.3,
        peak_height: float | None = None,
        peak_distance: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = False,
        filter: Literal["gaussian", "savgol"] | None = None,
    ) -> Result:
        if target is None:
            target = self.qubit_labels[0]

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

        if readout_amplitude is None:
            readout_amplitude = self.params.readout_amplitude[qubit_label]

        if electrical_delay is None:
            # measure electrical delay if not provided
            tau = self.measure_electrical_delay(
                target,
                f_start=frequency_range[0],
                shots=shots,
                plot=plot,
                confirm=False,
            )
        else:
            tau = electrical_delay

        if readout_duration is None:
            readout_duration = 8192
        if readout_ramptime is None:
            readout_ramptime = 128
        if readout_ramp_type is None:
            readout_ramp_type = "Bump"
        if interval is None:
            interval = 0

        # split frequency range to avoid the frequency sweep range limit
        frequency_range = np.array(frequency_range)
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=subrange_width,
        )

        bounds = [
            subranges[0][0],
        ] + [subrange[-1] for subrange in subranges]

        signals = []

        idx = 0
        phase_offset = 0.0

        if read_box.type == BoxType.QUEL1SE_R8:
            ssb = "L"
            cnco_center = CNCO_CETNER_READ_R8
        else:
            ssb = "U"
            cnco_center = CNCO_CETNER_READ

        for subrange in subranges:
            self.reset_awg_and_capunits(box_ids=[read_box.id])

            f_center = (subrange[0] + subrange[-1]) / 2
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_center * 1e9,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            with self.system_manager.modified_device_settings(
                label=read_label,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                for sub_idx, freq in enumerate(subrange):
                    if idx > 0 and sub_idx == 0:
                        prev_freq = frequency_range[idx - 1]
                        with self.modified_frequencies({read_label: prev_freq}):
                            result = self.measure(
                                {qubit_label: np.zeros(0)},
                                mode="avg",
                                readout_amplitudes={qubit_label: readout_amplitude},
                                shots=shots,
                                interval=interval,
                                reset_awg_and_capunits=False,
                            )
                            raw = result.data[target].kerneled
                            phase_adjust = 2 * np.pi * prev_freq * tau - phase_offset
                            signal = raw * np.exp(1j * phase_adjust)
                            phase_offset += np.angle(signal) - np.angle(signals[-1])

                    with self.modified_frequencies({read_label: freq}):
                        result = self.measure(
                            {qubit_label: np.zeros(0)},
                            mode="avg",
                            readout_amplitudes={qubit_label: readout_amplitude},
                            readout_duration=readout_duration,
                            readout_ramptime=readout_ramptime,
                            readout_drag_coeff=readout_drag_coeff,
                            readout_ramp_type=readout_ramp_type,
                            shots=shots,
                            interval=interval,
                            reset_awg_and_capunits=False,
                        )
                        raw = result.data[target].kerneled
                        phase_adjust = 2 * np.pi * freq * tau - phase_offset
                        signal = raw * np.exp(1j * phase_adjust)
                        signals.append(signal)
                        idx += 1

        signals = np.array(signals)
        amplitudes = np.abs(signals)
        phases = np.angle(signals)
        phases -= phases[0] - np.pi
        phases %= 2 * np.pi
        phases -= np.pi
        phases_unwrap = np.unwrap(phases)
        phases_diff = np.diff(phases_unwrap)
        if filter == "gaussian":
            from scipy.ndimage import gaussian_filter1d

            phases_unwrap_for_peak = gaussian_filter1d(phases_unwrap, sigma=2.0)
            phases_diff_for_peak = np.diff(phases_unwrap_for_peak)
            peaks, props = find_peaks(
                np.abs(phases_diff_for_peak),
                distance=peak_distance or 10,
                prominence=0.05,
            )
            num_resonators = 4
            sorted_peaks = sorted(zip(props["prominences"], peaks), reverse=True)
            top_peaks = sorted(sorted_peaks[:num_resonators], key=lambda x: x[1])
            peaks = [idx for _, idx in top_peaks]
        elif filter == "savgol":
            from scipy.signal import savgol_filter

            # window_length: around 5% of the data length
            window_frac = 0.05
            window_length = int(len(phases) * window_frac)
            window_length = max(7, window_length // 2 * 2 + 1)  # minimum 7, odd number
            polyorder = 3
            phases_unwrap_for_peak = savgol_filter(
                phases_unwrap, window_length=window_length, polyorder=polyorder
            )
            phases_diff_for_peak = np.diff(phases_unwrap_for_peak)
            peaks, props = find_peaks(
                np.abs(phases_diff_for_peak),
                distance=peak_distance or 10,
                prominence=0.05,
            )
            num_resonators = 4
            sorted_peaks = sorted(zip(props["prominences"], peaks), reverse=True)
            top_peaks = sorted(sorted_peaks[:num_resonators], key=lambda x: x[1])
            peaks = [idx for _, idx in top_peaks]
        else:
            peaks, _ = find_peaks(
                np.abs(phases_diff),
                height=peak_height or 0.5,
                distance=peak_distance or 10,
            )
        peak_freqs = frequency_range[peaks]

        fig1 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
        fig1.add_scatter(
            row=1,
            col=1,
            x=frequency_range,
            y=amplitudes,
            name=target,
            mode="markers+lines",
        )
        fig1.add_scatter(
            row=2,
            col=1,
            x=frequency_range,
            y=phases,
            name=target,
            mode="markers+lines",
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
        fig1.update_yaxes(title_text="Amplitude (arb. units)", row=1, col=1)
        fig1.update_yaxes(title_text="Phase (rad)", row=2, col=1)
        fig1.update_layout(
            title=dict(
                text=f"Resonator frequency scan : {mux.label}",
                subtitle=dict(
                    text=f"readout_amplitude={readout_amplitude:.6g}",
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
        fig2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
        fig2.add_scatter(
            row=1,
            col=1,
            x=frequency_range,
            y=phases_unwrap,
            name=target,
            mode="markers+lines",
        )
        fig2.add_scatter(
            row=2,
            col=1,
            x=frequency_range,
            y=phases_diff,
            name=target,
            mode="markers+lines",
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
        fig2.update_xaxes(title_text="Readout frequency (GHz)", row=2, col=1)
        fig2.update_yaxes(title_text="Unwrapped phase (rad)", row=1, col=1)
        fig2.update_yaxes(title_text="Phase diff (rad)", row=2, col=1)
        fig2.update_layout(
            title=dict(
                text=f"Resonator frequency scan : {mux.label}",
                subtitle=dict(
                    text=f"readout_amplitude={readout_amplitude:.6g}",
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
                width=600,
                height=450,
            )

        return Result(
            data={
                "peaks": peak_freqs,
                "frequency_range": frequency_range,
                "subranges": subranges,
                "signals": signals,
                "phases_unwrap": phases_unwrap,
                "phases_diff": phases_diff,
                "fig_phase": fig1,
                "fig_phase_diff": fig2,
            }
        )

    def resonator_spectroscopy(
        self,
        target: str | None = None,
        *,
        frequency_range: ArrayLike | None = None,
        power_range: ArrayLike = np.arange(-60, 5, 5),
        phase_shift: float | None = None,  # deprecated
        electrical_delay: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if target is None:
            target = self.qubit_labels[0]

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

        # measure electrical shift if not provided
        if electrical_delay is None:
            electrical_delay = self.measure_electrical_delay(
                target,
                shots=shots,
                interval=interval,
                plot=False,
            )

        result = []
        for power in tqdm(power_range):
            power_linear = 10 ** (power / 10)
            amplitude = np.sqrt(power_linear)
            phases_diff = self.scan_resonator_frequencies(
                target,
                frequency_range=frequency_range,
                electrical_delay=electrical_delay,
                readout_amplitude=amplitude,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )["phases_diff"]
            abs_phases_diff = np.abs(phases_diff)
            abs_phases_diff = np.append(abs_phases_diff, abs_phases_diff[-1])
            result.append(abs_phases_diff)

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=frequency_range,
                y=power_range,
                z=result,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(
                        text="Abs. phase shift (rad)",
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

        return Result(
            data={
                "frequency_range": frequency_range,
                "power_range": power_range,
                "data": np.array(result),
                "fig": fig,
            }
        )

    def measure_reflection_coefficient(
        self,
        target: str,
        *,
        center_frequency: float | None = None,
        df: float | None = None,
        frequency_width: float | None = None,
        readout_amplitude: float | None = None,
        electrical_delay: float | None = None,
        qubit_state: str = "0",
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        qubit_label = Target.qubit_label(target)
        read_label = Target.read_label(target)

        if center_frequency is None:
            center_frequency = self.targets[read_label].frequency
        if df is None:
            df = 0.0005  # 500 kHz step
        if frequency_width is None:
            frequency_width = 0.05
        if readout_amplitude is None:
            readout_amplitude = self.params.readout_amplitude[qubit_label]
        if electrical_delay is None:
            electrical_delay = self.measure_electrical_delay(
                target,
                f_start=(center_frequency - frequency_width / 2) // df * df,
                df=0.00005,
                n_samples=50,
                shots=128,
                interval=1024,
                plot=plot,
                confirm=False,
            )

        freq_range = np.arange(
            center_frequency - frequency_width / 2,
            center_frequency + frequency_width / 2,
            df,
        )

        signals = []

        initialize_pulse = self.get_pulse_for_state(
            target=qubit_label,
            state=qubit_state,
        )

        self.reset_awg_and_capunits(qubits=[qubit_label])

        for freq in freq_range:
            with self.modified_frequencies({read_label: freq}):
                result = self.measure(
                    {qubit_label: initialize_pulse},
                    mode="avg",
                    readout_amplitudes={qubit_label: readout_amplitude},
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                )
                signal = result.data[target].kerneled
                signal = signal * np.exp(1j * 2 * np.pi * freq * electrical_delay)
                signals.append(signal)

        signals = np.array(signals)
        amplitudes = np.abs(signals)
        # amplitudes -= (
        #     (amplitudes[-1] - amplitudes[0])
        #     / (freq_range[-1] - freq_range[0])
        #     * (freq_range - freq_range[0])
        # )
        phases = np.angle(signals)
        phases -= phases[0]
        signals = amplitudes * np.exp(1j * phases)

        fit_result = fitting.fit_reflection_coefficient(
            target=target,
            freq_range=freq_range,
            data=signals,
            plot=plot,
            title=f"Reflection coefficient of {target} : |{qubit_state}〉",
        )

        if plot:
            print(f"{target} : |{qubit_state}〉")
            print(f"f_r : {fit_result['f_r']:.6f} GHz")
            print(f"κ_e : {fit_result['kappa_ex'] * 1e3:.6f} MHz")
            print(f"κ_i : {fit_result['kappa_in'] * 1e3:.6f} MHz")

        fig = fit_result["fig"]

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"reflection_coefficient_{target}",
                width=800,
                height=450,
            )

        return Result(
            data={
                "frequency_range": freq_range,
                "reflection_coefficients": signals,
                **fit_result,
            }
        )

    def scan_qubit_frequencies(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        control_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        readout_frequency: float | None = None,
        subrange_width: float | None = None,
        peak_height: float | None = None,
        peak_distance: int | None = None,
        simultaneous_drive: bool = True,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = False,
    ) -> Result:
        # control and readout pulses
        qubit = Target.qubit_label(target)
        resonator = Target.read_label(target)
        ctrl_box = self.experiment_system.get_control_box_for_qubit(qubit)

        if control_amplitude is None:
            control_amplitude = self.params.control_amplitude[qubit]

        if readout_amplitude is None:
            readout_amplitude = self.params.readout_amplitude[qubit]

        # split frequency range to avoid the frequency sweep range limit
        if frequency_range is None:
            if ctrl_box.type == BoxType.QUEL1SE_R8:
                frequency_range = np.arange(3.0, 5.0, 0.005)
            else:
                frequency_range = np.arange(6.5, 9.5, 0.005)
        else:
            frequency_range = np.array(frequency_range)

        if subrange_width is None:
            subrange_width = 0.3
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=subrange_width,
        )

        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = 1024

        bounds = [
            subranges[0][0],
        ] + [subrange[-1] for subrange in subranges]

        # readout frequency
        readout_frequency = readout_frequency or self.targets[resonator].frequency

        # result buffer
        signals = []

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
            with self.system_manager.modified_device_settings(
                label=qubit,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                self.reset_awg_and_capunits(qubits=[qubit])
                for control_frequency in subrange:
                    with self.modified_frequencies(
                        {
                            qubit: control_frequency,
                            resonator: readout_frequency,
                        }
                    ):
                        with PulseSchedule([qubit, resonator]) as ps:
                            ps.add(
                                qubit,
                                Gaussian(
                                    duration=1024,
                                    amplitude=control_amplitude,
                                    sigma=128,
                                ),
                            )
                            if not simultaneous_drive:
                                ps.barrier()
                            ps.add(
                                resonator,
                                FlatTop(
                                    duration=1024,
                                    amplitude=readout_amplitude,
                                    tau=128,
                                ),
                            )
                        result = self.execute(
                            schedule=ps,
                            mode="avg",
                            shots=shots,
                            interval=interval,
                            reset_awg_and_capunits=False,
                        )
                        signal = result.data[qubit][-1].kerneled
                        signals.append(signal)
                        idx += 1

        signals = np.array(signals)
        amplitudes = np.abs(signals)
        phases = np.angle(signals)
        phases -= np.median(phases) - np.pi
        phases %= 2 * np.pi
        phases -= np.pi
        phases_std = np.std(phases)
        # phases[phases > 3 * phases_std] -= 2 * np.pi

        peaks, _ = find_peaks(
            np.abs(phases),
            height=peak_height or 3 * phases_std,
            distance=peak_distance or 10,
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
            y=amplitudes,
        )
        fig.add_scatter(
            name=target,
            mode="markers+lines",
            row=2,
            col=1,
            x=frequency_range,
            y=phases,
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
        fig.update_yaxes(title_text="Amplitude (arb. units)", row=1, col=1)
        fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)
        fig.update_layout(
            title=dict(
                text=f"Control frequency scan : {qubit}",
                subtitle=dict(
                    text=f"control_amplitude={control_amplitude:.6g}, readout_amplitude={readout_amplitude:.6g}",
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

        return Result(
            data={
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
                "signals": signals,
                "amplitudes": amplitudes,
                "phases": phases,
                "fig": fig,
            }
        )

    @deprecated("Use `measure_qubit_resonance` instead.")
    def estimate_control_amplitude(
        self,
        target: str,
        *,
        frequency_range: ArrayLike,
        control_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        target_rabi_rate: float = DEFAULT_RABI_FREQUENCY,
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
                        text=f"readout_amplitude={readout_amplitude:.6g}",
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

    def measure_qubit_resonance(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        control_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        target_rabi_rate: float = DEFAULT_RABI_FREQUENCY,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        qubit_label = Target.qubit_label(target)
        qubit_frequency = self.qubits[qubit_label].frequency

        if frequency_range is None:
            frequency_range = np.arange(
                qubit_frequency - 0.1,
                qubit_frequency + 0.1,
                0.001,
            )
        else:
            frequency_range = np.asarray(frequency_range)

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
        )
        data = np.unwrap(data["phases"])

        fit_result = fitting.fit_sqrt_lorentzian(
            target=target,
            x=frequency_range,
            y=data,
            plot=False,
            title="Qubit resonance fit",
        )

        rabi_rate = fit_result.get("Omega")
        if rabi_rate is None:
            return Result(
                data={
                    "frequency_range": frequency_range,
                    "phases": data,
                    "rabi_rate": None,
                    "estimated_amplitude": None,
                    "fig": None,
                }
            )
        estimated_amplitude = target_rabi_rate / rabi_rate * control_amplitude

        if plot:
            fig = fit_result["fig"]
            fig.update_layout(
                title=dict(
                    text=f"Control amplitude estimation : {target}",
                    subtitle=dict(
                        text=f"control_amplitude={control_amplitude:.6g}, readout_amplitude={readout_amplitude:.6g}",
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

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"measure_qubit_resonance{target}",
                width=600,
                height=300,
            )
        return Result(
            data={
                "frequency_range": frequency_range,
                "phases": data,
                "rabi_rate": rabi_rate,
                "estimated_amplitude": estimated_amplitude,
                "fig": fig,
                **fit_result,
            }
        )

    def qubit_spectroscopy(
        self,
        target: str,
        frequency_range: ArrayLike | None = None,
        power_range: ArrayLike = np.arange(-60, 0, 5),
        readout_amplitude: float | None = None,
        readout_frequency: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
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
            phases = result1d["phases"]
            result2d.append(phases)

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
                    text=f"readout_amplitude={result1d['readout_amplitude']:.6g}",
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

        return Result(
            data={
                "frequency_range": result1d["frequency_range"],
                "power_range": power_range,
                "data": np.array(result2d),
                "fig": fig,
            }
        )

    def measure_dispersive_shift(
        self,
        target: str,
        *,
        df: float | None = None,
        frequency_width: float | None = None,
        readout_amplitude: float | None = None,
        electrical_delay: float | None = None,
        threshold: float = 0.5,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        result_0 = self.measure_reflection_coefficient(
            target,
            df=df,
            frequency_width=frequency_width,
            readout_amplitude=readout_amplitude,
            electrical_delay=electrical_delay,
            qubit_state="0",
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=False,
        )
        result_1 = self.measure_reflection_coefficient(
            target,
            df=df,
            frequency_width=frequency_width,
            readout_amplitude=readout_amplitude,
            electrical_delay=electrical_delay,
            qubit_state="1",
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=False,
        )

        frequency_range = result_0["frequency_range"]
        signals_0 = result_0["reflection_coefficients"]
        signals_1 = result_1["reflection_coefficients"]
        phases_0 = np.angle(signals_0)
        phases_0 -= phases_0[0]
        phases_diff_0 = np.diff(phases_0)
        phases_diff_0[phases_diff_0 > threshold] -= 2 * np.pi
        phases_0 = np.concatenate([[0], np.cumsum(phases_diff_0)])
        phases_1 = np.angle(signals_1)
        phases_1 -= phases_1[0]
        phases_diff_1 = np.diff(phases_1)
        phases_diff_1[phases_diff_1 > threshold] -= 2 * np.pi
        phases_1 = np.concatenate([[0], np.cumsum(phases_diff_1)])
        f_0 = result_0["f_r"]
        f_1 = result_1["f_r"]
        dispersive_shift = (f_1 - f_0) / 2

        fig1 = go.Figure()
        fig1.add_scatter(
            x=frequency_range,
            y=phases_0,
            name="0",
            mode="lines+markers",
        )
        fig1.add_scatter(
            x=frequency_range,
            y=phases_1,
            name="1",
            mode="lines+markers",
        )
        fig1.add_vline(
            x=f_0,
            line_width=2,
            line_color="red",
            opacity=0.6,
        )
        fig1.add_vline(
            x=f_1,
            line_width=2,
            line_color="red",
            opacity=0.6,
        )
        fig1.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text=f"χ: {dispersive_shift * 1e3:.3f} MHz",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
        )
        fig1.update_layout(
            title=f"Dispersive shift : {target}",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Phase (rad)",
            width=600,
            height=300,
        )

        distance = np.abs(signals_1 - signals_0)
        optimal_frequency = frequency_range[np.argmax(distance)]
        fig2 = go.Figure()
        fig2.add_scatter(
            x=frequency_range,
            y=distance,
            name="State distance",
            mode="lines+markers",
        )
        fig2.add_vline(
            x=optimal_frequency,
            line_width=2,
            line_color="red",
            opacity=0.6,
        )
        fig2.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text=f"f_opt: {optimal_frequency:.4f} GHz",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
        )
        fig2.update_layout(
            title=f"State separation : {target}",
            xaxis_title="Frequency (GHz)",
            yaxis_title="State distance",
            width=600,
            height=300,
        )

        if plot:
            fig1.show()
            fig2.show()
            print(f"f_0  : {f_0:.4f} GHz")
            print(f"f_1  : {f_1:.4f} GHz")
            print(f"χ    : {dispersive_shift * 1e3:.3f} MHz")
            print(f"f_opt: {optimal_frequency:.4f} GHz")

        if save_image:
            viz.save_figure_image(
                fig1,
                name=f"dispersive_shift_{target}",
                width=600,
                height=300,
            )

        return Result(
            data={
                "f_0": f_0,
                "f_1": f_1,
                "dispersive_shift": dispersive_shift,
                "optimal_frequency": optimal_frequency,
                "frequency_range": frequency_range,
                "signals_0": signals_0,
                "signals_1": signals_1,
                "phases_0": phases_0,
                "phases_1": phases_1,
                "fig": fig1,
            }
        )

    def find_optimal_readout_frequency(
        self,
        target: str,
        *,
        df: float | None = None,
        frequency_width: float | None = None,
        readout_amplitude: float | None = None,
        electrical_delay: float | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if df is None:
            df = 0.0004
        if frequency_width is None:
            frequency_width = 0.02
        result_0 = self.measure_reflection_coefficient(
            target,
            df=df,
            frequency_width=frequency_width,
            readout_amplitude=readout_amplitude,
            electrical_delay=electrical_delay,
            qubit_state="0",
            shots=shots,
            interval=interval,
            plot=False,
            save_image=False,
        )
        result_1 = self.measure_reflection_coefficient(
            target,
            df=df,
            frequency_width=frequency_width,
            readout_amplitude=readout_amplitude,
            electrical_delay=electrical_delay,
            qubit_state="1",
            shots=shots,
            interval=interval,
            plot=False,
            save_image=False,
        )

        frequency_range = result_0["frequency_range"]
        signals_0 = result_0["reflection_coefficients"]
        signals_1 = result_1["reflection_coefficients"]

        distance = np.abs(signals_1 - signals_0)
        optimal_frequency = frequency_range[np.argmax(distance)]
        fig = go.Figure()
        fig.add_scatter(
            x=frequency_range,
            y=distance,
            name="State distance",
            mode="lines+markers",
        )
        fig.add_vline(
            x=optimal_frequency,
            line_width=2,
            line_color="red",
            opacity=0.6,
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text=f"f_opt: {optimal_frequency:.4f} GHz",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
        )
        fig.update_layout(
            title=f"State separation : {target}",
            xaxis_title="Frequency (GHz)",
            yaxis_title="State distance",
            width=600,
            height=300,
        )

        if plot:
            fig.show()
            print(f"f_opt: {optimal_frequency:.4f} GHz")

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"optimal_readout_frequency_{target}",
                width=600,
                height=300,
            )

        return Result(
            data={
                "optimal_frequency": optimal_frequency,
                "frequency_range": frequency_range,
                "signals_0": signals_0,
                "signals_1": signals_1,
                "fig": fig,
            }
        )

    def find_optimal_readout_amplitude(
        self,
        target: str,
        *,
        amplitude_range: ArrayLike | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if amplitude_range is None:
            amplitude_range = np.arange(0.01, 0.26, 0.01)
        else:
            amplitude_range = np.array(amplitude_range)

        buffer_0 = []
        buffer_1 = []
        for amplitude in tqdm(amplitude_range):
            result_0 = self.measure_state(
                {target: "0"},
                mode="avg",
                readout_amplitudes={target: amplitude},
                shots=shots,
                interval=interval,
            )
            result_1 = self.measure_state(
                {target: "1"},
                mode="avg",
                readout_amplitudes={target: amplitude},
                shots=shots,
                interval=interval,
            )
            buffer_0.append(result_0.data[target].kerneled)
            buffer_1.append(result_1.data[target].kerneled)
        signals_0 = np.array(buffer_0)
        signals_1 = np.array(buffer_1)

        distance = np.abs(signals_1 - signals_0)
        optimal_amplitude = amplitude_range[np.argmax(distance)]
        fig = go.Figure()
        fig.add_scatter(
            x=amplitude_range,
            y=distance,
            name="State distance",
            mode="lines+markers",
        )
        fig.add_vline(
            x=optimal_amplitude,
            line_width=2,
            line_color="red",
            opacity=0.6,
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text=f"amp_opt: {optimal_amplitude:.4f}",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
        )
        fig.update_layout(
            title=f"State separation : {target}",
            xaxis_title="Amplitude (arb. units)",
            yaxis_title="State distance",
            width=600,
            height=300,
        )

        if plot:
            viz.scatter_iq_data(
                {
                    "0": signals_0,
                    "1": signals_1,
                }
            )
            fig.show()
            print(f"amp_opt: {optimal_amplitude:.4f}")

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"optimal_readout_amplitude_{target}",
                width=600,
                height=300,
            )

        return Result(
            data={
                "optimal_amplitude": optimal_amplitude,
                "amplitude_range": amplitude_range,
                "signals_0": signals_0,
                "signals_1": signals_1,
                "fig": fig,
            }
        )

    def ckp_sequence(
        self,
        target: str,
        qubit_initial_state: str | None = None,
        qubit_drive_detuning: float | None = None,
        qubit_pi_pulse: Waveform | None = None,
        qubit_drive_scale: float | None = None,
        resonator_drive_detuning: float | None = None,
        resonator_drive_amplitude: float | None = None,
        resonator_drive_duration: float | None = None,
        resonator_drive_ramptime: float | None = None,
    ) -> PulseSchedule:
        qubit = self.qubits[target].label
        resonator = self.resonators[target].label

        if qubit_initial_state is None:
            qubit_initial_state = "0"
        if qubit_drive_detuning is None:
            qubit_drive_detuning = 0.0
        if qubit_pi_pulse is None:
            qubit_pi_pulse = self.get_hpi_pulse(target).repeated(2)
        if qubit_drive_scale is None:
            qubit_drive_scale = 0.8
        if resonator_drive_detuning is None:
            resonator_drive_detuning = 0.0
        if resonator_drive_amplitude is None:
            resonator_drive_amplitude = 0.1
        if resonator_drive_duration is None:
            resonator_drive_duration = 1024
        if resonator_drive_ramptime is None:
            resonator_drive_ramptime = 32

        resonetor_drive_pulse = FlatTop(
            duration=resonator_drive_duration,
            amplitude=resonator_drive_amplitude,
            tau=resonator_drive_ramptime,
        ).detuned(resonator_drive_detuning)
        qubit_drive_pulse = (
            qubit_pi_pulse.padded(
                resonator_drive_duration,
                pad_side="left",
            )
            .scaled(qubit_drive_scale)
            .detuned(qubit_drive_detuning)
        )
        resonator_readout_pulse = self.readout(target)
        with PulseSchedule() as seq:
            if qubit_initial_state == "1":
                seq.add(qubit, qubit_pi_pulse)
            seq.barrier()
            seq.add(qubit, qubit_drive_pulse)
            seq.add(resonator, Blank(64))
            seq.add(resonator, resonetor_drive_pulse)
            seq.add(resonator, Blank(1024))
            seq.add(resonator, resonator_readout_pulse)
        return seq

    def ckp_measurement(
        self,
        target: str,
        qubit_initial_state: str,
        qubit_detuning_range: ArrayLike | None = None,
        qubit_pi_pulse: Waveform | None = None,
        qubit_drive_scale: float | None = None,
        resonator_detuning_range: ArrayLike | None = None,
        resonator_drive_amplitude: float | None = None,
        resonator_drive_duration: float | None = None,
        plot: bool = True,
        verbose: bool = False,
        save_image: bool = True,
    ) -> Result:
        if qubit_detuning_range is None:
            qubit_detuning_range = np.linspace(-0.03, 0.01, 30)
        else:
            qubit_detuning_range = np.asarray(qubit_detuning_range)
        if resonator_detuning_range is None:
            resonator_detuning_range = np.linspace(-0.01, 0.01, 30)
        else:
            resonator_detuning_range = np.asarray(resonator_detuning_range)

        qubit_label = Target.qubit_label(target)
        read_label = Target.read_label(target)
        f_qubit = self.targets[qubit_label].frequency
        f_resonator = self.targets[read_label].frequency
        qubit_frequency_range = qubit_detuning_range + f_qubit
        resonator_frequency_range = resonator_detuning_range + f_resonator

        result2d = []
        qubit_resonance_frequencies = []
        self.reset_awg_and_capunits(qubits=[qubit_label])
        for resonator_detuning in tqdm(resonator_detuning_range):
            result1d = []
            for qubit_detuning in qubit_detuning_range:
                result = self.execute(
                    self.ckp_sequence(
                        target=target,
                        qubit_initial_state=qubit_initial_state,
                        qubit_drive_scale=qubit_drive_scale,
                        qubit_pi_pulse=qubit_pi_pulse,
                        qubit_drive_detuning=qubit_detuning,
                        resonator_drive_detuning=resonator_detuning,
                        resonator_drive_duration=resonator_drive_duration,
                        resonator_drive_amplitude=resonator_drive_amplitude,
                    ),
                    reset_awg_and_capunits=False,
                )
                data = result.data[target][-1]
                result1d.append(data.kerneled)

            result1d = self.rabi_params[target].normalize(np.array(result1d))

            f0 = fitting.fit_lorentzian(
                x=qubit_frequency_range,
                y=result1d,
                plot=True if verbose else False,
            ).get("f0", np.nan)
            qubit_resonance_frequencies.append(f0)
            result2d.append(result1d)

        data = np.array(result2d)
        if qubit_initial_state == "1":
            data *= -1

        fig = go.Figure()
        fig.add_heatmap(
            z=data.T,
            x=resonator_frequency_range,
            y=qubit_frequency_range,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(
                    text="Normalized signal",
                    side="right",
                ),
            ),
        )
        fig.update_layout(
            title=dict(
                text=f"CKP measurement : {target} : |{qubit_initial_state}〉",
                subtitle=dict(
                    text=f"resonator_drive_amplitude={resonator_drive_amplitude:.6g}",
                    font=dict(size=11, family="monospace"),
                ),
            ),
            xaxis_title="Resonator drive frequency (GHz)",
            yaxis_title="Qubit drive frequency (GHz)",
            width=600,
            height=400,
        )
        if plot:
            fig.show()

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"ckp_measurement_{target}_{qubit_initial_state}",
                width=600,
                height=400,
            )

        qubit_resonance_frequencies = np.array(qubit_resonance_frequencies)
        fit_result = fitting.fit_lorentzian(
            x=resonator_frequency_range,
            y=qubit_resonance_frequencies,
            plot=plot,
            title=f"CKP experiment fit : {target} : |{qubit_initial_state}〉",
            xlabel="Resonator drive frequency (GHz)",
            ylabel="Qubit drive frequency (GHz)",
        )

        return Result(
            data={
                "qubit_frequency_range": qubit_frequency_range,
                "resonator_frequency_range": resonator_frequency_range,
                "qubit_detuning_range": qubit_detuning_range,
                "resonator_detuning_range": resonator_detuning_range,
                "qubit_resonance_frequencies": qubit_resonance_frequencies,
                "qubit_initial_state": qubit_initial_state,
                "data": data,
                "fit_result": fit_result,
            }
        )

    def ckp_experiment(
        self,
        target: str,
        qubit_detuning_range: ArrayLike | None = None,
        qubit_pi_pulse: Waveform | None = None,
        qubit_drive_scale: float | None = None,
        resonator_detuning_range: ArrayLike | None = None,
        resonator_drive_amplitude: float | None = None,
        resonator_drive_duration: float | None = None,
        plot: bool = True,
        verbose: bool = False,
        save_image: bool = True,
    ) -> Result:
        if resonator_drive_amplitude is None:
            resonator_drive_amplitude = self.params.get_readout_amplitude(
                Target.qubit_label(target)
            )

        if qubit_pi_pulse is None:
            duration = 128
            ramptime = 64
            calib_result = self.calibrate_default_pulse(
                target,
                pulse_type="pi",
                duration=duration,
                ramptime=ramptime,
                update_params=False,
            )
            amplitude = calib_result.data[target].calib_value
            qubit_pi_pulse = FlatTop(
                duration=duration,
                amplitude=amplitude,
                tau=ramptime,
            )

        result_0 = self.ckp_measurement(
            target=target,
            qubit_initial_state="0",
            qubit_detuning_range=qubit_detuning_range,
            qubit_pi_pulse=qubit_pi_pulse,
            qubit_drive_scale=qubit_drive_scale,
            resonator_detuning_range=resonator_detuning_range,
            resonator_drive_amplitude=resonator_drive_amplitude,
            resonator_drive_duration=resonator_drive_duration,
            plot=plot,
            verbose=verbose,
            save_image=save_image,
        )
        result_1 = self.ckp_measurement(
            target=target,
            qubit_initial_state="1",
            qubit_detuning_range=qubit_detuning_range,
            qubit_pi_pulse=qubit_pi_pulse,
            qubit_drive_scale=qubit_drive_scale,
            resonator_detuning_range=resonator_detuning_range,
            resonator_drive_amplitude=resonator_drive_amplitude,
            resonator_drive_duration=resonator_drive_duration,
            plot=plot,
            verbose=verbose,
            save_image=save_image,
        )

        x_data = result_0["resonator_frequency_range"]
        fit_result_0 = result_0["fit_result"]
        fit_result_1 = result_1["fit_result"]
        y_data_0 = result_0["qubit_resonance_frequencies"]
        y_data_1 = result_1["qubit_resonance_frequencies"]
        x_fit = np.linspace(
            x_data[0],
            x_data[-1],
            1000,
        )
        popt_0 = fit_result_0["popt"]
        popt_1 = fit_result_1["popt"]
        gamma_0 = fit_result_0["gamma"]
        gamma_1 = fit_result_1["gamma"]
        gamma = (gamma_0 + gamma_1) / 2
        A_0 = fit_result_0["A"]
        A_1 = fit_result_1["A"]
        A = (A_0 + A_1) / 2
        C_0 = fit_result_0["C"]
        C_1 = fit_result_1["C"]
        f_q = (C_0 + C_1) / 2
        y_fit_0 = fitting.func_lorentzian(x_fit, *popt_0)
        y_fit_1 = fitting.func_lorentzian(x_fit, *popt_1)
        f_r_0 = fit_result_0["f0"]
        f_r_1 = fit_result_1["f0"]
        f_r = (f_r_0 + f_r_1) / 2
        delta = f_r - f_q
        chi = (f_r_1 - f_r_0) / 2
        kappa = gamma * 2
        power = kappa * A / (8 * chi)
        n_mean = 4 * power / kappa
        n_crit = np.abs(delta / (4 * chi))

        fig = go.Figure()
        fig.add_scatter(
            x=x_fit,
            y=y_fit_0,
            name="|0⟩ fit",
            mode="lines",
            line=dict(color=COLORS[0]),
        )
        fig.add_scatter(
            x=x_data,
            y=y_data_0,
            name="|0⟩ data",
            mode="markers",
            marker=dict(color=COLORS[0]),
        )
        fig.add_scatter(
            x=x_fit,
            y=y_fit_1,
            name="|1⟩ fit",
            mode="lines",
            line=dict(color=COLORS[1]),
        )
        fig.add_scatter(
            x=x_data,
            y=y_data_1,
            name="|1⟩ data",
            mode="markers",
            marker=dict(color=COLORS[1]),
        )
        fig.add_vline(
            x=f_r_0,
            line_width=2,
            line_color="red",
            opacity=0.6,
        )
        fig.add_vline(
            x=f_r_1,
            line_width=2,
            line_color="red",
            opacity=0.6,
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.05,
            text=f"χ : {chi * 1e3:.3f} MHz<br>"
            f"κ : {kappa * 1e3:.3f} MHz<br>"
            f"|A|² : {power * 1e3:.3f} MHz<br>",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
        )
        fig.update_layout(
            title=dict(
                text=f"CKP Experiment : {target}",
                subtitle=dict(
                    text=f"resonator_drive_amplitude={resonator_drive_amplitude:.6g}",
                    font=dict(size=11, family="monospace"),
                ),
            ),
            xaxis_title="Resonator drive frequency (GHz)",
            yaxis_title="Qubit drive frequency (GHz)",
            width=600,
            height=300,
        )
        if plot:
            fig.show()
            print(f"f_r_0 : {f_r_0:.6f} GHz")
            print(f"f_r_1 : {f_r_1:.6f} GHz")
            print(f"f_r   : {f_r:.6f} GHz")
            print(f"f_q   : {f_q:.6f} GHz")
            print(f"Δ     : {delta * 1e3:.3f} MHz")
            print(f"χ     : {chi * 1e3:.3f} MHz")
            print(f"κ     : {kappa * 1e3:.3f} MHz")
            print(f"|A|²  : {power * 1e3:.3f} MHz")
            print(f"n̅*    : {n_mean:.3f}")
            print(f"n_c   : {n_crit:.3f}")

        if save_image:
            viz.save_figure_image(
                fig,
                name=f"ckp_experiment_{target}",
                width=600,
                height=300,
            )

        return Result(
            data={
                "f_r_0": f_r_0,
                "f_r_1": f_r_1,
                "f_r": f_r,
                "f_q": f_q,
                "delta": delta,
                "chi": chi,
                "kappa": kappa,
                "power": power,
                "n": n_mean,
                "n_crit": n_crit,
                "result_0": result_0,
                "result_1": result_1,
            }
        )

    def characterize_1q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        data = {
            "t1_experiment": {},
            "t2_experiment": {},
            "ramsey_experiment": {},
        }

        for target in targets:
            try:
                result = self.t1_experiment(
                    target,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                    save_image=save_image,
                )
                data["t1_experiment"][target] = result.data[target]

                result = self.t2_experiment(
                    target,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                    save_image=save_image,
                )
                data["t2_experiment"][target] = result.data[target]

                result = self.ramsey_experiment(
                    target,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                    save_image=save_image,
                )
                data["ramsey_experiment"][target] = result.data[target]

            except Exception as e:
                print(f"Characterization failed for {target}: {e}")
                continue

        if plot:
            print()
            print("T1 (µs):")
            for target in targets:
                try:
                    t1 = data["t1_experiment"][target].t1
                    print(f"  {target}: {t1 * 1e-3:.6f}")
                except Exception:
                    print(f"  {target}: null")
            print()
            print("T2 echo (µs):")
            for target in targets:
                try:
                    t2_echo = data["t2_experiment"][target].t2
                    print(f"  {target}: {t2_echo * 1e-3:.6f}")
                except Exception:
                    print(f"  {target}: null")
            print()
            print("T2* (µs):")
            for target in targets:
                try:
                    t2_star = data["ramsey_experiment"][target].t2
                    print(f"  {target}: {t2_star * 1e-3:.6f}")
                except Exception:
                    print(f"  {target}: null")
            print()
            print("Qubit frequency (GHz):")
            for target in targets:
                try:
                    bare_freq = data["ramsey_experiment"][target].bare_freq
                    print(f"  {target}: {bare_freq:.6f}")
                except Exception:
                    print(f"  {target}: null")

        return Result(data=data)

    def characterize_2q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if targets is None:
            targets = self.edge_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        data = {
            "obtain_coupling_strength": {},
        }

        for target in targets:
            try:
                pair = target.split("-")
                result = self.obtain_coupling_strength(
                    *pair,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                )
                data["obtain_coupling_strength"][target] = result.data

            except Exception as e:
                print(f"Characterization failed for {target}: {e}")
                continue

        if plot:
            print()
            print("Qubit-qubit coupling strength g (MHz):")
            for target in targets:
                try:
                    g = data["obtain_coupling_strength"][target]["g"]
                    print(f"  {target}: {g * 1e3:.6f}")
                except Exception:
                    print(f"  {target}: null")
            print()
            print("ZZ coefficient ξ (kHz):")
            for target in targets:
                try:
                    xi = data["obtain_coupling_strength"][target]["xi"]
                    print(f"  {target}: {xi * 1e6:.6f}")
                except Exception:
                    print(f"  {target}: null")

        return Result(data=data)
