"""Crosstalk check experiment utilities."""

from __future__ import annotations

import math
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from qubex import visualization as viz
from qubex.analysis import fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.models.result import Result
from qubex.pulse import FlatTop, PulseSchedule
from qubex.system import LatticeGraph, SystemManager
from qubex.system.quel1 import MixingUtil
from tqdm import tqdm

DEFAULT_SSB = "L"
DEFAULT_CNCO_CENTER = 2_250_000_000
DEFAULT_TIME_RANGES: tuple[range, ...] = (
    range(0, 1201, 40),
    range(0, 12001, 400),
    range(0, 120001, 4000),
)


def crosstalk_check(
    exp: Experiment,
    control_qubit: str,
    target_qubits: list[str],
    time_ranges: list[range] | None = None,
    shots: int = DEFAULT_SHOTS * 2,
    ramptime: float = 0.0,
    fft_threshold: float = 0.1,
    plot: bool = True,
    fft_plot: bool = False,
    save: bool = False,
    overwrite: bool = False,
    ssb: Literal["L", "U"] = DEFAULT_SSB,
    cnco_center: int = DEFAULT_CNCO_CENTER,
) -> Result:
    """Perform a crosstalk check using a control-qubit Rabi drive against target qubits."""
    if time_ranges is None:
        time_ranges = list(DEFAULT_TIME_RANGES)

    with suppress(ValueError):
        target_qubits.remove(control_qubit)

    rabi_result = exp.obtain_rabi_params(control_qubit, plot=False)
    rabi_params = rabi_result.rabi_params
    params = rabi_params.get(control_qubit) if rabi_params is not None else None

    if (
        params is None
        or not np.isfinite(params.frequency)
        or not np.isfinite(params.r2)
    ):
        raise RuntimeError(
            f"Failed to obtain valid Rabi parameters for {control_qubit}."
        )

    max_rabi_freq = params.frequency * 1e3 / exp.params.control_amplitude[control_qubit]
    print(f"Max Rabi frequency for {control_qubit}: {max_rabi_freq} MHz")

    sweep_results = {}
    crosstalk_results = {}

    for target_qubit in tqdm(target_qubits):
        crosstalk_value = None  # Initialize crosstalk_value for each target qubit

        def rabi_sequence(T: float, target_qubit: str = target_qubit) -> PulseSchedule:
            with PulseSchedule([control_qubit, target_qubit]) as ps:
                ps.add(
                    control_qubit,
                    FlatTop(
                        duration=T + 2 * ramptime,
                        amplitude=1.0,
                        tau=ramptime,
                    ),
                )
            return ps

        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            exp.targets[target_qubit].frequency * 1e9,
            ssb=ssb,
            cnco_center=cnco_center,
        )

        resonator_label = exp.targets[target_qubit].read_label(target_qubit)
        port = exp.targets[resonator_label].channel.port
        labels = [
            t.label
            for t in exp.experiment_system.read_out_targets
            if port.id == t.channel.port.id and t.label != resonator_label
        ]

        exp.system_manager.load(
            chip_id=exp.chip_id,
            config_dir=exp.config_path,
            params_dir=exp.params_path,
            targets_to_exclude=labels,
        )
        exp.system_manager.push(box_ids=exp.box_ids, confirm=False)

        with exp.system_manager.modified_backend_settings(
            label=control_qubit,
            lo_freq=lo,
            cnco_freq=cnco,
            fnco_freq=0,
        ):
            for time_range in time_ranges:
                time_range = np.array(time_range, dtype=np.float64)
                sweep_result = exp.sweep_parameter(
                    sequence=rabi_sequence,
                    sweep_range=time_range,
                    frequencies={control_qubit: exp.targets[target_qubit].frequency},
                    shots=shots,
                    interval=DEFAULT_INTERVAL,
                    plot=plot,
                )
                for target, data in sweep_result.data.items():
                    if target != target_qubit:
                        continue
                    data_fft = np.fft.fft(
                        (data.data - np.mean(data.data)) / np.std(data.data)
                    )[: data.data.size // 2]
                    if fft_plot:
                        viz.plot_fft(
                            time_range,
                            (data.data - np.mean(data.data)) / np.std(data.data),
                            title=f"FFT of {target_qubit} response to {control_qubit} Rabi drive",
                            xlabel="Frequency (GHz)",
                            ylabel="Amplitude",
                        )
                    if (
                        np.abs(data_fft[1:])[np.argmax(np.abs(data_fft[1:]))]
                        / np.std(data.data)
                        / data.data.size
                        > fft_threshold
                    ):
                        fit_result = fitting.fit_rabi(
                            target=data.target,
                            times=time_range,
                            data=data.data,
                            reference_point=exp.obtain_reference_points(target)[
                                "iq"
                            ].get(target),
                            plot=plot,
                            is_damped=True,
                        )
                        if fit_result.status.value != "success":
                            crosstalk_value = None
                        else:
                            crosstalk_value = 10 * np.log10(
                                (fit_result.data["frequency"] * 1e3 / max_rabi_freq)
                                ** 2
                            )
                        break  # Exit the loop after the first successful fit
                    print(
                        f"Warning: No significant peak found in FFT for {target_qubit} at time range {np.min(time_range)} to {np.max(time_range)} ns. Skipping fitting."
                    )
                else:
                    continue
                break

        crosstalk_results[target_qubit] = crosstalk_value
        sweep_results[target_qubit] = sweep_result

        exp.system_manager.load(
            chip_id=exp.chip_id, config_dir=exp.config_path, params_dir=exp.params_path
        )
        exp.system_manager.push(box_ids=exp.box_ids, confirm=False)

    print("Crosstalk results (dB):")

    for target_qubit, crosstalk_value in crosstalk_results.items():
        if crosstalk_value is not None:
            print(f"{target_qubit}: {crosstalk_value:.2f} dB")
        else:
            print(f"{target_qubit}: Fit failed, crosstalk value is None")

    if save:

        def load_yaml(file_path: str | Path) -> dict[str, Any]:
            with open(file_path) as file:
                data = yaml.safe_load(file)
            return data if data is not None else {}

        def write_yaml(data: dict[str, Any], file_path: str | Path) -> None:
            with open(file_path, "w") as file:
                yaml.safe_dump(data, file, default_flow_style=False)

        data_dict = {
            "meta": {
                "description": "Crosstalk",
                "unit": "dB",
            },
            "data": {
                f"{control_qubit}": {
                    f"{target_qubit}": float(crosstalk_value)
                    if crosstalk_value is not None
                    else None
                    for target_qubit, crosstalk_value in crosstalk_results.items()
                }
            },
        }

        yaml_path = Path(exp.params_path) / "crosstalk.yaml"
        yaml_data = load_yaml(yaml_path) if yaml_path.exists() else {}

        if yaml_data is None:
            yaml_data = {}
        yaml_data.setdefault("meta", data_dict["meta"])
        yaml_data.setdefault("data", {})
        yaml_data["data"].setdefault(f"{control_qubit}", {})

        if overwrite:
            yaml_data["data"][f"{control_qubit}"].update(
                data_dict["data"][f"{control_qubit}"]
            )
        else:
            for target_qubit, crosstalk_value in crosstalk_results.items():
                if crosstalk_value is not None:
                    yaml_data["data"][f"{control_qubit}"][f"{target_qubit}"] = float(
                        crosstalk_value
                    )

        write_yaml(yaml_data, yaml_path)

    return Result(
        data={"sweep_results": sweep_results, "crosstalk_results": crosstalk_results},
    )


def _is_valid(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


def init_yaml_file(path, description="Crosstalk", unit="dB"):
    data = {
        "meta": {
            "description": description,
            "unit": unit,
        },
        "data": {
            f"{'Q' + str(i).zfill(2)}": {
                f"{'Q' + str(j).zfill(2)}": None for j in range(64)
            }
            for i in range(64)
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)


def plot_crosstalk(
    exp: Experiment,
    control_qubit: str,
    save_image: bool = False,
) -> None:
    """Plot crosstalk values on the lattice graph for a control qubit."""
    system_manager = SystemManager.shared()
    chip = exp.chip
    loader = system_manager.config_loader
    graph = LatticeGraph(chip.n_qubits)
    if values := loader.load_param_data("crosstalk"):
        graph.plot_lattice_data(
            title="Estimate crosstalk (dB)",
            values=list(values[control_qubit].values()),
            texts=[
                f"{qubit}<br>control<br>qubit"
                if qubit == control_qubit
                else f"{qubit}<br>{value:.3f}<br>dB"
                if _is_valid(value)
                else "N/A"
                for qubit, value in values[control_qubit].items()
            ],
            hovertexts=[
                f"{qubit}: control qubit"
                if qubit == control_qubit
                else f"{qubit}: {value:.3f} dB"
                if _is_valid(value)
                else f"{qubit}: N/A"
                for qubit, value in values[control_qubit].items()
            ],
            save_image=save_image,
            image_name="crosstalk_dB",
        )
