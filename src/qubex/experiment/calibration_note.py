from __future__ import annotations

from pathlib import Path

from .experiment_constants import (
    CALIBRATION_DIR,
    CR_PARAMS,
    DRAG_HPI_PARAMS,
    DRAG_PI_PARAMS,
    HPI_PARAMS,
    PI_PARAMS,
    RABI_PARAMS,
    STATE_CENTERS,
)
from .experiment_note import ExperimentNote


class CalibrationNote(ExperimentNote):
    def __init__(
        self,
        chip_id: str,
        dir_path: str = CALIBRATION_DIR,
    ):
        file_path = Path(dir_path) / f"{chip_id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(str(file_path))

    @property
    def rabi_params(self) -> dict:
        return self.get(RABI_PARAMS)

    @rabi_params.setter
    def rabi_params(self, value: dict):
        self.put(RABI_PARAMS, value)

    @property
    def hpi_params(self) -> dict:
        return self.get(HPI_PARAMS)

    @hpi_params.setter
    def hpi_params(self, value: dict):
        self.put(HPI_PARAMS, value)

    @property
    def pi_params(self) -> dict:
        return self.get(PI_PARAMS)

    @pi_params.setter
    def pi_params(self, value: dict):
        self.put(PI_PARAMS, value)

    @property
    def drag_hpi_params(self) -> dict:
        return self.get(DRAG_HPI_PARAMS)

    @drag_hpi_params.setter
    def drag_hpi_params(self, value: dict):
        self.put(DRAG_HPI_PARAMS, value)

    @property
    def drag_pi_params(self) -> dict:
        return self.get(DRAG_PI_PARAMS)

    @drag_pi_params.setter
    def drag_pi_params(self, value: dict):
        self.put(DRAG_PI_PARAMS, value)

    @property
    def state_centers(self) -> dict:
        return self.get(STATE_CENTERS)

    @state_centers.setter
    def state_centers(self, value: dict):
        self.put(STATE_CENTERS, value)

    @property
    def cr_params(self) -> dict:
        return self.get(CR_PARAMS)

    @cr_params.setter
    def cr_params(self, value: dict):
        self.put(CR_PARAMS, value)
