from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, TypedDict

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

logger = getLogger(__name__)


class Property(TypedDict, total=False):
    timestamp: str


class RabiParam(Property):
    target: str
    amplitude: float
    frequency: float
    phase: float
    offset: float
    noise: float
    angle: float


class FlatTopParam(Property):
    target: str
    amplitude: float
    duration: float
    tau: float


class DragParam(Property):
    target: str
    amplitude: float
    duration: float
    beta: float


class StateCenter(Property):
    target: str
    states: dict[str, list[float]]


class CrossResonanceParam(Property):
    target: str
    duration: float
    ramptime: float
    cr_amplitude: float
    cr_phase: float
    cancel_amplitude: float
    cancel_phase: float
    cr_cancel_ratio: float


class CalibrationNote(ExperimentNote):
    def __init__(
        self,
        chip_id: str,
        file_path: Path | str | None = None,
    ):
        if file_path is None:
            file_path = Path(CALIBRATION_DIR) / f"{chip_id}.json"
        else:
            file_path = Path(file_path)
        super().__init__(file_path)

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        return self.get(RABI_PARAMS)

    @rabi_params.setter
    def rabi_params(self, value: dict[str, RabiParam]):
        self.put(RABI_PARAMS, value)

    @property
    def hpi_params(self) -> dict[str, FlatTopParam]:
        return self.get(HPI_PARAMS)

    @hpi_params.setter
    def hpi_params(self, value: dict[str, FlatTopParam]):
        self.put(HPI_PARAMS, value)

    @property
    def pi_params(self) -> dict[str, FlatTopParam]:
        return self.get(PI_PARAMS)

    @pi_params.setter
    def pi_params(self, value: dict[str, FlatTopParam]):
        self.put(PI_PARAMS, value)

    @property
    def drag_hpi_params(self) -> dict[str, DragParam]:
        return self.get(DRAG_HPI_PARAMS)

    @drag_hpi_params.setter
    def drag_hpi_params(self, value: dict[str, DragParam]):
        self.put(DRAG_HPI_PARAMS, value)

    @property
    def drag_pi_params(self) -> dict[str, DragParam]:
        return self.get(DRAG_PI_PARAMS)

    @drag_pi_params.setter
    def drag_pi_params(self, value: dict[str, DragParam]):
        self.put(DRAG_PI_PARAMS, value)

    @property
    def state_centers(self) -> dict[str, StateCenter]:
        return self.get(STATE_CENTERS)

    @state_centers.setter
    def state_centers(self, value: dict[str, StateCenter]):
        self.put(STATE_CENTERS, value)

    @property
    def cr_params(self) -> dict[str, CrossResonanceParam]:
        return self.get(CR_PARAMS)

    @cr_params.setter
    def cr_params(self, value: dict[str, CrossResonanceParam]):
        self.put(CR_PARAMS, value)

    def get_rabi_param(
        self,
        target: str,
        cutoff: int | None = None,
    ) -> RabiParam | None:
        return self.get_property(RABI_PARAMS, target, cutoff)

    def get_hpi_param(
        self,
        target: str,
        cutoff: int | None = None,
    ) -> FlatTopParam | None:
        return self.get_property(HPI_PARAMS, target, cutoff)

    def get_pi_param(
        self,
        target: str,
        cutoff: int | None = None,
    ) -> FlatTopParam | None:
        return self.get_property(PI_PARAMS, target, cutoff)

    def get_drag_hpi_param(
        self,
        target: str,
        cutoff: int | None = None,
    ) -> DragParam | None:
        return self.get_property(DRAG_HPI_PARAMS, target, cutoff)

    def get_drag_pi_param(
        self,
        target: str,
        cutoff: int | None = None,
    ) -> DragParam | None:
        return self.get_property(DRAG_PI_PARAMS, target, cutoff)

    def get_state_center(
        self,
        target: str,
        cutoff: int | None = None,
    ) -> StateCenter | None:
        return self.get_property(STATE_CENTERS, target, cutoff)

    def get_cr_param(
        self,
        target: str,
        cutoff: int | None = None,
    ) -> CrossResonanceParam | None:
        return self.get_property(CR_PARAMS, target, cutoff)

    def get_property(
        self,
        key: str,
        target: str,
        cutoff: int | None = None,
    ) -> Any:
        property = self.get(key)
        if property is None:
            return None
        value = property.get(target)
        if value is None:
            return None
        if cutoff is None:
            return value
        timestamp = value["timestamp"]
        time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        days_passed = (datetime.now() - time).days
        if days_passed >= cutoff:
            logger.warning(
                f"{key}['{target}'] is outdated and ignored ({days_passed} days passed)."
            )
            return None
        return value

    def get(
        self,
        key: str,
        cutoff: int | None = None,
    ) -> dict[str, Any]:
        property = super().get(key)
        if property is None:
            raise KeyError(f"Key '{key}' not found.")
        if cutoff is None:
            return property
        value = {}
        for k, v in property.items():
            timestamp = v["timestamp"]
            time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - time).days < cutoff:
                value[k] = v
        return value

    def put(self, key: str, value: dict[str, Any]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for v in value.values():
            v["timestamp"] = timestamp
        super().put(key, value)
