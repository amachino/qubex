from __future__ import annotations

import numpy as np

from qubex.backend import DeviceController, ExperimentSystem, RawResult, SystemManager
from qubex.typing import TargetMap

from .classifiers import StateClassifier
from .models import MeasureData, MeasureMode, MeasureResult, MultipleMeasureResult


class MeasurementExecutor:
    def __init__(
        self,
        *,
        system_manager: SystemManager,
        device_controller: DeviceController,
        experiment_system: ExperimentSystem,
        classifiers: TargetMap[StateClassifier],
    ) -> None:
        self._system_manager = system_manager
        self._device_controller = device_controller
        self._experiment_system = experiment_system
        self._classifiers = classifiers

    def execute_measurement(
        self,
        *,
        sequencer,
        measure_mode: MeasureMode,
        shots: int,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool = False,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> MeasureResult:
        backend_result = self._device_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=shots,
            integral_mode=measure_mode.integral_mode,
            dsp_demodulation=enable_dsp_demodulation,
            enable_sum=enable_dsp_sum,
            enable_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
        result = self._create_measure_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=shots,
        )
        self._save_if_needed(result)
        return result

    def execute_schedule(
        self,
        *,
        sequencer,
        measure_mode: MeasureMode,
        shots: int,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool = False,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> MultipleMeasureResult:
        backend_result = self._device_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=shots,
            integral_mode=measure_mode.integral_mode,
            dsp_demodulation=enable_dsp_demodulation,
            enable_sum=enable_dsp_sum,
            enable_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
        result = self._create_multiple_measure_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=shots,
        )
        self._save_if_needed(result)
        return result

    def _save_if_needed(self, result: MeasureResult | MultipleMeasureResult) -> None:
        rawdata_dir = self._system_manager.rawdata_dir
        if rawdata_dir is not None:
            result.save(data_dir=rawdata_dir)

    def _create_measure_result(
        self,
        *,
        backend_result: RawResult,
        measure_mode: MeasureMode,
        shots: int,
    ) -> MeasureResult:
        label_slice = slice(1, None)  # remove the resonator prefix "R"
        norm_factor = 2 ** (-32)  # normalization factor for 32-bit data
        capture_index = -1

        iq_data: dict[str, np.ndarray] = {}
        for target, iqs in sorted(backend_result.data.items()):
            sideband = self._experiment_system.get_target(target).sideband
            if sideband == "L":
                iq_data[target] = np.conjugate(iqs)
            else:
                iq_data[target] = iqs

        if measure_mode == MeasureMode.SINGLE:
            backend_data = {
                # iqs[capture_index]: ndarray[shots, duration]
                target[label_slice]: iqs[capture_index] * norm_factor
                for target, iqs in iq_data.items()
            }
            measure_data = {
                qubit: MeasureData(
                    target=qubit,
                    mode=measure_mode,
                    raw=iq,
                    classifier=self._classifiers.get(qubit),
                )
                for qubit, iq in backend_data.items()
            }
        elif measure_mode == MeasureMode.AVG:
            backend_data = {
                # iqs[capture_index]: ndarray[1, duration]
                target[label_slice]: iqs[capture_index].squeeze() * norm_factor / shots
                for target, iqs in iq_data.items()
            }
            measure_data = {
                qubit: MeasureData(
                    target=qubit,
                    mode=measure_mode,
                    raw=iq,
                )
                for qubit, iq in backend_data.items()
            }
        else:
            raise ValueError(f"Invalid measure mode: {measure_mode}")

        return MeasureResult(
            mode=measure_mode,
            data=measure_data,
            config=self._device_controller.box_config,
        )

    def _create_multiple_measure_result(
        self,
        *,
        backend_result: RawResult,
        measure_mode: MeasureMode,
        shots: int,
    ) -> MultipleMeasureResult:
        label_slice = slice(1, None)  # remove the resonator prefix "R"
        norm_factor = 2 ** (-32)  # normalization factor for 32-bit data

        iq_data: dict[str, list[np.ndarray]] = {}
        for target, iqs in sorted(backend_result.data.items()):
            sideband = self._experiment_system.get_target(target).sideband
            if sideband == "L":
                iq_data[target] = [np.conjugate(iq) for iq in iqs]
            else:
                iq_data[target] = iqs

        measure_data: dict[str, list[MeasureData]] = {}
        if measure_mode == MeasureMode.SINGLE:
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                measure_data[qubit] = []
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(
                        MeasureData(
                            target=qubit,
                            mode=measure_mode,
                            raw=iq * norm_factor,
                            classifier=self._classifiers.get(qubit),
                        )
                    )
        elif measure_mode == MeasureMode.AVG:
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                measure_data[qubit] = []
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(
                        MeasureData(
                            target=qubit,
                            mode=measure_mode,
                            raw=iq.squeeze() * norm_factor / shots,
                        )
                    )
        else:
            raise ValueError(f"Invalid measure mode: {measure_mode}")

        return MultipleMeasureResult(
            mode=measure_mode,
            data=measure_data,
            config=self._device_controller.box_config,
        )
