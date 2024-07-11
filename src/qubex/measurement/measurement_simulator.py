from __future__ import annotations

import typing
from contextlib import contextmanager
from typing import Literal

import numpy as np

from ..config import Config, LatticeChipGraph, Target
from ..simulator import Control, Coupling, Simulator, System, Transmon
from ..typing import IQArray, TargetMap
from .measurement import Measurement
from .measurement_result import MeasureData, MeasureMode, MeasureResult
from .qube_backend import QubeBackend

DEFAULT_CONFIG_DIR = "./config"
DEFAULT_SHOTS = 1024
DEFAULT_INTERVAL = 150 * 1024  # ns
DEFAULT_CONTROL_WINDOW = 1024  # ns
DEFAULT_CAPTURE_WINDOW = 1024  # ns
DEFAULT_READOUT_DURATION = 512  # ns
INTERVAL_STEP = 10240  # ns


class MeasurementSimulator(Measurement):
    def __init__(
        self,
        chip_id: str,
        qubits: list[str],
        *,
        config_dir: str = DEFAULT_CONFIG_DIR,
    ):
        self._chip_id = chip_id
        self._qubits = qubits
        config = Config(config_dir)
        self._config = config
        config_path = config.get_system_settings_path(chip_id)
        self._backend = QubeBackend(config_path)
        self._simulator = self._create_simulator(config)

    def _create_simulator(self, config: Config) -> Simulator:
        if self._chip_id == "64Q":
            chip_graph = LatticeChipGraph(4, 4)
        elif self._chip_id == "16Q":
            chip_graph = LatticeChipGraph(2, 2)
        else:
            raise ValueError(f"Invalid chip ID: {self._chip_id}")

        transmons = []
        for label in self._qubits:
            qubit = config.get_qubit(self._chip_id, label)
            transmon = Transmon(
                label=label,
                dimension=3,
                frequency=qubit.frequency,
                anharmonicity=qubit.anharmonicity,
                decay_rate=0.0,
                dephasing_rate=0.0,
            )
            transmons.append(transmon)

        couplings = []
        for edge in chip_graph.qubit_edges:
            if edge[0] in self._qubits and edge[1] in self._qubits:
                coupling = Coupling(
                    pair=edge,
                    strength=0.01,
                )
                couplings.append(coupling)

        system = System(
            transmons=transmons,
            couplings=couplings,
        )
        return Simulator(system)

    @property
    def chip_id(self) -> str:
        """Get the chip ID."""
        return self._chip_id

    @property
    def targets(self) -> dict[str, Target]:
        """Get the targets."""
        all_targets = self._config.get_all_targets(self._chip_id)
        targets = {}
        for target in all_targets:
            if target.qubit in self._qubits:
                targets[target.label] = target
        return targets

    def check_link_status(self, box_list: list[str]) -> dict:
        raise NotImplementedError

    def check_clock_status(self, box_list: list[str]) -> dict:
        raise NotImplementedError

    def linkup(self, box_list: list[str]):
        raise NotImplementedError

    def relinkup(self, box_list: list[str]):
        raise NotImplementedError

    @contextmanager
    def modified_frequencies(self, target_frequencies: dict[str, float]):
        yield

    def measure_noise(
        self,
        targets: list[str],
        duration: int,
    ) -> MeasureResult:
        raise NotImplementedError

    def measure(
        self,
        waveforms: TargetMap[IQArray],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ) -> MeasureResult:
        if len(waveforms) != 1:
            raise ValueError("Simulator only supports single qubit control")

        target = next(iter(waveforms.keys()))
        waveform = next(iter(waveforms.values()))

        control = Control(
            target=target,
            frequency=self.targets[target].frequency,
            waveform=waveform,
        )

        initial_state = self._simulator.system.state(
            {label: "0" for label in self._qubits},
        )

        result = self._simulator.simulate(
            control=control,
            initial_state=initial_state,
        )

        measure_mode = MeasureMode(mode)

        measure_data = {}

        for qubit in self._qubits:
            if len(result.states) < 2:
                kerneled = np.array(1.0)
            else:
                state = result.substates(qubit)[-1]
                kerneled = abs(state[0][0][0])
            measure_data[qubit] = MeasureData(
                target=qubit,
                mode=measure_mode,
                raw=np.array([]),
                kerneled=kerneled,
                classified=None,
            )

        return MeasureResult(
            mode=measure_mode,
            data=measure_data,
            config={},
        )

    def measure_batch(
        self,
        waveforms_list: typing.Sequence[TargetMap[IQArray]],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ):
        for waveforms in waveforms_list:
            yield self.measure(
                waveforms,
                mode=mode,
                shots=shots,
                interval=interval,
                control_window=control_window,
                capture_window=capture_window,
                readout_duration=readout_duration,
            )
