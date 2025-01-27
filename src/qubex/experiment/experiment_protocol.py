from __future__ import annotations

from typing import Collection, ContextManager, Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Protocol, deprecated

from ..analysis import RabiParam
from ..backend import (
    Box,
    ControlParams,
    ControlSystem,
    DeviceController,
    ExperimentSystem,
    QuantumSystem,
    Qubit,
    Resonator,
    StateManager,
    Target,
)
from ..clifford import Clifford, CliffordGenerator
from ..measurement import (
    Measurement,
    MeasureResult,
    StateClassifier,
)
from ..measurement.measurement import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from ..pulse import (
    PulseSchedule,
    Waveform,
)
from ..typing import IQArray, ParametricPulseSchedule, ParametricWaveformDict, TargetMap
from .experiment_constants import (
    CALIBRATION_SHOTS,
    RABI_TIME_RANGE,
)
from .experiment_note import ExperimentNote
from .experiment_record import ExperimentRecord
from .experiment_result import (
    ExperimentResult,
    RabiData,
    SweepData,
)
from .experiment_util import ExperimentUtil


class ExperimentProtocol(Protocol):
    def _create_qubit_labels(
        self,
        chip_id: str,
        muxes: Collection[str | int] | None,
        qubits: Collection[str | int] | None,
        exclude_qubits: Collection[str | int] | None,
        config_dir: str,
        params_dir: str,
    ) -> list[str]: ...

    def _validate(self): ...

    @property
    def tool(self): ...

    @property
    def util(self) -> type[ExperimentUtil]: ...

    @property
    def measurement(self) -> Measurement: ...

    @property
    def state_manager(self) -> StateManager: ...

    @property
    def experiment_system(self) -> ExperimentSystem: ...

    @property
    def quantum_system(self) -> QuantumSystem: ...

    @property
    def control_system(self) -> ControlSystem: ...

    @property
    def device_controller(self) -> DeviceController: ...

    @property
    def params(self) -> ControlParams: ...

    @property
    def chip_id(self) -> str: ...

    @property
    def qubit_labels(self) -> list[str]: ...

    @property
    def mux_labels(self) -> list[str]: ...

    @property
    def qubits(self) -> dict[str, Qubit]: ...

    @property
    def resonators(self) -> dict[str, Resonator]: ...

    @property
    def targets(self) -> dict[str, Target]: ...

    @property
    def available_targets(self) -> dict[str, Target]: ...

    @property
    def ge_targets(self) -> dict[str, Target]: ...

    @property
    def ef_targets(self) -> dict[str, Target]: ...

    @property
    def cr_targets(self) -> dict[str, Target]: ...

    @property
    def boxes(self) -> dict[str, Box]: ...

    @property
    def box_ids(self) -> list[str]: ...

    @property
    def config_path(self) -> str: ...

    @property
    def params_path(self) -> str: ...

    @property
    def system_note(self) -> ExperimentNote: ...

    @property
    def note(self) -> ExperimentNote: ...

    @property
    def hpi_pulse(self) -> dict[str, Waveform]: ...

    @property
    def pi_pulse(self) -> dict[str, Waveform]: ...

    @property
    def drag_hpi_pulse(self) -> dict[str, Waveform]: ...

    @property
    def drag_pi_pulse(self) -> dict[str, Waveform]: ...

    @property
    def ef_hpi_pulse(self) -> dict[str, Waveform]: ...

    @property
    def ef_pi_pulse(self) -> dict[str, Waveform]: ...

    @property
    def rabi_params(self) -> dict[str, RabiParam]: ...

    @property
    def ge_rabi_params(self) -> dict[str, RabiParam]: ...

    @property
    def ef_rabi_params(self) -> dict[str, RabiParam]: ...

    @property
    def classifier_type(self) -> Literal["kmeans", "gmm"]: ...

    @property
    def classifiers(self) -> TargetMap[StateClassifier]: ...

    @property
    def state_centers(self) -> dict[str, dict[int, complex]]: ...

    @property
    def clifford_generator(self) -> CliffordGenerator: ...

    @property
    def clifford(self) -> dict[str, Clifford]: ...

    def _validate_rabi_params(
        self,
        targets: Collection[str] | None = None,
    ): ...

    def store_rabi_params(self, rabi_params: dict[str, RabiParam]): ...

    def get_pulse_for_state(
        self,
        target: str,
        state: str,  # Literal["0", "1", "+", "-", "+i", "-i"],
    ) -> Waveform: ...

    def get_spectators(
        self,
        qubit: str,
        in_same_mux: bool = False,
    ) -> list[Qubit]: ...

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray: ...

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray: ...

    def print_environment(self, verbose: bool = False): ...

    def print_boxes(self): ...

    def check_status(self): ...

    def linkup(
        self,
        box_ids: Optional[list[str]] = None,
        noise_threshold: int = 500,
    ) -> None: ...

    def resync_clocks(
        self,
        box_ids: Optional[list[str]] = None,
    ) -> None: ...

    def configure(
        self,
        box_ids: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ): ...

    def reload(self): ...

    @deprecated("This method is tentative. It may be removed in the future.")
    def register_custom_target(
        self,
        *,
        label: str,
        frequency: float,
        box_id: str,
        port_number: int,
        channel_number: int,
        update_lsi: bool = False,
    ): ...

    def modified_frequencies(
        self, frequencies: dict[str, float] | None
    ) -> ContextManager: ...

    def print_defaults(self): ...

    def save_defaults(self): ...

    def clear_defaults(self): ...

    def delete_defaults(self): ...

    def load_record(
        self,
        name: str,
    ) -> ExperimentRecord: ...

    def execute(
        self,
        schedule: PulseSchedule,
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> MeasureResult: ...

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        initial_states: dict[str, str] | None = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        plot: bool = False,
    ) -> MeasureResult: ...

    def _measure_batch(
        self,
        sequences: Sequence[TargetMap[IQArray]],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
    ): ...

    def measure_state(
        self,
        states: dict[
            str, Literal["0", "1", "+", "-", "+i", "-i"] | Literal["g", "e", "f"]
        ],
        *,
        mode: Literal["single", "avg"] = "single",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        plot: bool = False,
    ) -> MeasureResult: ...

    def measure_readout_snr(
        self,
        targets: Collection[str] | None = None,
        *,
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict: ...

    def sweep_readout_amplitude(
        self,
        targets: Collection[str] | None = None,
        *,
        amplitude_range: ArrayLike = np.linspace(0.0, 0.1, 21),
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict: ...

    def sweep_readout_duration(
        self,
        targets: Collection[str] | None = None,
        *,
        time_range: ArrayLike = np.arange(128, 2048, 128),
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_margin: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict: ...

    def check_noise(
        self,
        targets: Collection[str] | None = None,
        *,
        duration: int = 10240,
        plot: bool = True,
    ) -> MeasureResult: ...

    def check_waveform(
        self,
        targets: Collection[str] | None = None,
        *,
        plot: bool = True,
    ) -> MeasureResult: ...

    def check_rabi(
        self,
        targets: Collection[str] | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        store_params: bool = True,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]: ...

    def obtain_rabi_params(
        self,
        targets: Collection[str] | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        amplitudes: dict[str, float] | None = None,
        frequencies: dict[str, float] | None = None,
        is_damped: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = True,
        simultaneous: bool = False,
    ) -> ExperimentResult[RabiData]: ...

    def obtain_ef_rabi_params(
        self,
        targets: Collection[str] | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        is_damped: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]: ...

    def rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike = RABI_TIME_RANGE,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]: ...

    def ef_rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]: ...

    def sweep_parameter(
        self,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        *,
        sweep_range: ArrayLike,
        repetitions: int = 1,
        frequencies: Optional[dict[str, float]] = None,
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        plot: bool = True,
        title: str = "Sweep result",
        xaxis_title: str = "Sweep value",
        yaxis_title: str = "Measured value",
        xaxis_type: Literal["linear", "log"] = "linear",
        yaxis_type: Literal["linear", "log"] = "linear",
    ) -> ExperimentResult[SweepData]: ...

    def repeat_sequence(
        self,
        sequence: TargetMap[Waveform] | PulseSchedule,
        *,
        repetitions: int = 20,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[SweepData]: ...
