"""Factory for constructing measurement results from backend output."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from qubex.backend import ExperimentSystem
from qubex.backend.quel1 import RawResult

from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult


class MeasurementResultFactory:
    """Create `MeasurementResult` instances from backend raw results."""

    def __init__(self, *, experiment_system: ExperimentSystem) -> None:
        self._experiment_system = experiment_system

    def create(
        self,
        *,
        backend_result: RawResult,
        measurement_config: MeasurementConfig,
        device_config: dict,
    ) -> MeasurementResult:
        """
        Build a measurement result from backend output and runtime config.

        Parameters
        ----------
        backend_result : RawResult
            Raw status/data/config returned by backend execution.
        measurement_config : MeasurementConfig
            Configuration used for the run.
        device_config : dict
            Device configuration snapshot to store in result.

        Returns
        -------
        MeasurementResult
            Canonical measurement result.
        """
        measure_mode = measurement_config.mode
        label_slice = slice(1, None)  # remove the resonator prefix "R"
        norm_factor = 2 ** (-32)  # normalization factor for 32-bit data

        iq_data = {}
        for target, iqs in sorted(backend_result.data.items()):
            sideband = self._experiment_system.get_target(target).sideband
            if sideband == "L":
                iq_data[target] = [np.conjugate(iq) for iq in iqs]
            else:
                iq_data[target] = iqs

        measure_data = defaultdict(list)
        if measure_mode == "single":
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(iq * norm_factor)
        elif measure_mode == "avg":
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(
                        iq.squeeze() * norm_factor / measurement_config.shots
                    )
        else:
            raise ValueError(f"Invalid measure mode: {measure_mode}")

        return MeasurementResult(
            mode=measure_mode,
            data=dict(measure_data),
            device_config=device_config,
            measurement_config=measurement_config.to_dict(),
        )
