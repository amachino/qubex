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
    STATE_PARAMS,
)
from .experiment_note import ExperimentNote

logger = getLogger(__name__)


class Parameter(TypedDict, total=False):
    timestamp: str


class RabiParam(Parameter):
    target: str
    amplitude: float
    frequency: float
    phase: float
    offset: float
    noise: float
    angle: float
    r2: float


class StateParam(Parameter):
    target: str
    centers: dict[str, list[float]]


class FlatTopParam(Parameter):
    target: str
    amplitude: float
    duration: float
    tau: float


class DragParam(Parameter):
    target: str
    amplitude: float
    duration: float
    beta: float


class CrossResonanceParam(Parameter):
    target: str
    duration: float
    ramptime: float
    cr_amplitude: float
    cr_phase: float
    cr_beta: float
    cancel_amplitude: float
    cancel_phase: float
    cancel_beta: float
    rotary_amplitude: float
    zx_rotation_rate: float


class CalibrationNote(ExperimentNote):
    def __init__(
        self,
        chip_id: str,
        calibration_dir: Path | str = CALIBRATION_DIR,
        file_path: Path | str | None = None,
    ):
        self._chip_id = chip_id
        if file_path is None:
            file_path = Path(calibration_dir) / f"{chip_id}.json"
        else:
            file_path = Path(file_path)
        super().__init__(file_path)

    @property
    def chip_id(self) -> str:
        return self._chip_id

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
    def state_params(self) -> dict[str, StateParam]:
        return self.get(STATE_PARAMS)

    @state_params.setter
    def state_params(self, value: dict[str, StateParam]):
        self.put(STATE_PARAMS, value)

    @property
    def cr_params(self) -> dict[str, CrossResonanceParam]:
        return self.get(CR_PARAMS)

    @cr_params.setter
    def cr_params(self, value: dict[str, CrossResonanceParam]):
        self.put(CR_PARAMS, value)

    def get_rabi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> RabiParam | None:
        return self.get_property(RABI_PARAMS, target, valid_days)

    def update_rabi_param(
        self,
        target: str,
        value: RabiParam,
    ):
        self.put_property(RABI_PARAMS, target, value)

    def remove_rabi_param(
        self,
        target: str,
    ):
        self.remove_property(RABI_PARAMS, target)

    def get_hpi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> FlatTopParam | None:
        return self.get_property(HPI_PARAMS, target, valid_days)

    def update_hpi_param(
        self,
        target: str,
        value: FlatTopParam,
    ):
        self.put_property(HPI_PARAMS, target, value)

    def remove_hpi_param(
        self,
        target: str,
    ):
        self.remove_property(HPI_PARAMS, target)

    def get_pi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> FlatTopParam | None:
        return self.get_property(PI_PARAMS, target, valid_days)

    def update_pi_param(
        self,
        target: str,
        value: FlatTopParam,
    ):
        self.put_property(PI_PARAMS, target, value)

    def remove_pi_param(
        self,
        target: str,
    ):
        self.remove_property(PI_PARAMS, target)

    def get_drag_hpi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> DragParam | None:
        return self.get_property(DRAG_HPI_PARAMS, target, valid_days)

    def update_drag_hpi_param(
        self,
        target: str,
        value: DragParam,
    ):
        self.put_property(DRAG_HPI_PARAMS, target, value)

    def remove_drag_hpi_param(
        self,
        target: str,
    ):
        self.remove_property(DRAG_HPI_PARAMS, target)

    def get_drag_pi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> DragParam | None:
        return self.get_property(DRAG_PI_PARAMS, target, valid_days)

    def update_drag_pi_param(
        self,
        target: str,
        value: DragParam,
    ):
        self.put_property(DRAG_PI_PARAMS, target, value)

    def remove_drag_pi_param(
        self,
        target: str,
    ):
        self.remove_property(DRAG_PI_PARAMS, target)

    def get_state_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> StateParam | None:
        return self.get_property(STATE_PARAMS, target, valid_days)

    def update_state_param(
        self,
        target: str,
        value: StateParam,
    ):
        self.put_property(STATE_PARAMS, target, value)

    def remove_state_param(
        self,
        target: str,
    ):
        self.remove_property(STATE_PARAMS, target)

    def get_cr_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> CrossResonanceParam | None:
        return self.get_property(CR_PARAMS, target, valid_days)

    def update_cr_param(
        self,
        target: str,
        value: CrossResonanceParam,
    ):
        self.put_property(CR_PARAMS, target, value)

    def remove_cr_param(
        self,
        target: str,
    ):
        self.remove_property(CR_PARAMS, target)

    def get_property(
        self,
        key: str,
        target: str,
        valid_days: int | None = None,
    ) -> Any:
        property = self.get(key)
        value = property.get(target)
        if value is None:
            return None
        if valid_days is None:
            return value
        timestamp = value["timestamp"]
        time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        days_passed = (datetime.now() - time).days
        if days_passed >= valid_days:
            logger.warning(
                f"{key}['{target}'] is outdated and ignored ({days_passed} days passed)."
            )
            return None
        return value

    def put_property(
        self,
        key: str,
        target: str,
        value: Any,
    ):
        self.put(key, {target: value})

    def remove_property(
        self,
        key: str,
        target: str,
    ):
        property = self.get(key)
        if property is None:
            print(f"Key '{key}' not found.")
            return
        if target not in property:
            print(f"Key '{target}' not found.")
            return
        del property[target]
        print(f"Key '{target}' removed.")

    def get(
        self,
        key: str,
        valid_days: int | None = None,
    ) -> dict[str, Any]:
        property = super().get(key)
        if property is None:
            return {}
        if valid_days is None:
            return property
        value = {}
        for k, v in property.items():
            timestamp = v["timestamp"]
            time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - time).days < valid_days:
                value[k] = v
        return value

    def put(
        self,
        key: str,
        value: dict[str, Any],
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for v in value.values():
            if v.get("timestamp") is None:
                v["timestamp"] = timestamp
        super().put(key, value)
