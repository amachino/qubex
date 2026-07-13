"""Calibration note storage and access helpers."""

from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, TypedDict

from qubex.experiment.experiment_constants import (
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
    """Base schema for calibration parameters."""

    timestamp: str


class RabiParam(Parameter):
    """Rabi calibration parameters for a target."""

    target: str
    amplitude: float
    frequency: float
    phase: float
    offset: float
    noise: float
    angle: float
    distance: float
    r2: float
    reference_phase: float


class StateParam(Parameter):
    """State classification parameters for a target."""

    target: str
    centers: dict[str, list[float]]
    reference_phase: float


class FlatTopParam(Parameter):
    """Flat-top pulse calibration parameters for a target."""

    target: str
    amplitude: float
    duration: float
    tau: float


class DragParam(Parameter):
    """DRAG pulse calibration parameters for a target."""

    target: str
    amplitude: float
    duration: float
    beta: float


class CrossResonanceParam(Parameter):
    """Cross-resonance calibration parameters for a target."""

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
    """Calibration note data store with typed helpers."""

    def __init__(
        self,
        chip_id: str,
        calibration_dir: Path | str = CALIBRATION_DIR,
        file_path: Path | str | None = None,
    ):
        """Initialize a calibration note for a chip."""
        self._chip_id = chip_id
        self._reference_phases = {}
        if file_path is None:
            file_path = Path(calibration_dir) / f"{chip_id}.json"
        else:
            file_path = Path(file_path)
        super().__init__(file_path)

    @property
    def chip_id(self) -> str:
        """Return the chip identifier."""
        return self._chip_id

    @property
    def reference_phases(self) -> dict[str, float]:
        """Return reference phases by target."""
        return self._reference_phases

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        """Return stored Rabi parameters."""
        return self.get(RABI_PARAMS)

    @rabi_params.setter
    def rabi_params(self, value: dict[str, RabiParam]):
        self.put(RABI_PARAMS, value)

    @property
    def hpi_params(self) -> dict[str, FlatTopParam]:
        """Return stored half-pi pulse parameters."""
        return self.get(HPI_PARAMS)

    @hpi_params.setter
    def hpi_params(self, value: dict[str, FlatTopParam]):
        self.put(HPI_PARAMS, value)

    @property
    def pi_params(self) -> dict[str, FlatTopParam]:
        """Return stored pi pulse parameters."""
        return self.get(PI_PARAMS)

    @pi_params.setter
    def pi_params(self, value: dict[str, FlatTopParam]):
        self.put(PI_PARAMS, value)

    @property
    def drag_hpi_params(self) -> dict[str, DragParam]:
        """Return stored DRAG half-pi parameters."""
        return self.get(DRAG_HPI_PARAMS)

    @drag_hpi_params.setter
    def drag_hpi_params(self, value: dict[str, DragParam]):
        self.put(DRAG_HPI_PARAMS, value)

    @property
    def drag_pi_params(self) -> dict[str, DragParam]:
        """Return stored DRAG pi parameters."""
        return self.get(DRAG_PI_PARAMS)

    @drag_pi_params.setter
    def drag_pi_params(self, value: dict[str, DragParam]):
        self.put(DRAG_PI_PARAMS, value)

    @property
    def state_params(self) -> dict[str, StateParam]:
        """Return stored state classification parameters."""
        return self.get(STATE_PARAMS)

    @state_params.setter
    def state_params(self, value: dict[str, StateParam]):
        self.put(STATE_PARAMS, value)

    @property
    def cr_params(self) -> dict[str, CrossResonanceParam]:
        """Return stored cross-resonance parameters."""
        return self.get(CR_PARAMS)

    @cr_params.setter
    def cr_params(self, value: dict[str, CrossResonanceParam]):
        self.put(CR_PARAMS, value)

    def get_rabi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> RabiParam | None:
        """Return a Rabi parameter entry if valid."""
        return self.get_property(RABI_PARAMS, target, valid_days)

    def update_rabi_param(
        self,
        target: str,
        value: RabiParam,
    ) -> None:
        """Update a Rabi parameter entry."""
        self.put_property(RABI_PARAMS, target, value)

    def remove_rabi_param(
        self,
        target: str,
    ) -> None:
        """Remove a Rabi parameter entry."""
        self.remove_property(RABI_PARAMS, target)

    def get_hpi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> FlatTopParam | None:
        """Return a half-pi parameter entry if valid."""
        return self.get_property(HPI_PARAMS, target, valid_days)

    def update_hpi_param(
        self,
        target: str,
        value: FlatTopParam,
    ) -> None:
        """Update a half-pi parameter entry."""
        self.put_property(HPI_PARAMS, target, value)

    def remove_hpi_param(
        self,
        target: str,
    ) -> None:
        """Remove a half-pi parameter entry."""
        self.remove_property(HPI_PARAMS, target)

    def get_pi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> FlatTopParam | None:
        """Return a pi parameter entry if valid."""
        return self.get_property(PI_PARAMS, target, valid_days)

    def update_pi_param(
        self,
        target: str,
        value: FlatTopParam,
    ) -> None:
        """Update a pi parameter entry."""
        self.put_property(PI_PARAMS, target, value)

    def remove_pi_param(
        self,
        target: str,
    ) -> None:
        """Remove a pi parameter entry."""
        self.remove_property(PI_PARAMS, target)

    def get_drag_hpi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> DragParam | None:
        """Return a DRAG half-pi parameter entry if valid."""
        return self.get_property(DRAG_HPI_PARAMS, target, valid_days)

    def update_drag_hpi_param(
        self,
        target: str,
        value: DragParam,
    ) -> None:
        """Update a DRAG half-pi parameter entry."""
        self.put_property(DRAG_HPI_PARAMS, target, value)

    def remove_drag_hpi_param(
        self,
        target: str,
    ) -> None:
        """Remove a DRAG half-pi parameter entry."""
        self.remove_property(DRAG_HPI_PARAMS, target)

    def get_drag_pi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> DragParam | None:
        """Return a DRAG pi parameter entry if valid."""
        return self.get_property(DRAG_PI_PARAMS, target, valid_days)

    def update_drag_pi_param(
        self,
        target: str,
        value: DragParam,
    ) -> None:
        """Update a DRAG pi parameter entry."""
        self.put_property(DRAG_PI_PARAMS, target, value)

    def remove_drag_pi_param(
        self,
        target: str,
    ) -> None:
        """Remove a DRAG pi parameter entry."""
        self.remove_property(DRAG_PI_PARAMS, target)

    def get_state_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> StateParam | None:
        """Return a state parameter entry if valid."""
        return self.get_property(STATE_PARAMS, target, valid_days)

    def update_state_param(
        self,
        target: str,
        value: StateParam,
    ) -> None:
        """Update a state parameter entry."""
        self.put_property(STATE_PARAMS, target, value)

    def remove_state_param(
        self,
        target: str,
    ) -> None:
        """Remove a state parameter entry."""
        self.remove_property(STATE_PARAMS, target)

    def get_cr_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> CrossResonanceParam | None:
        """Return a cross-resonance parameter entry if valid."""
        return self.get_property(CR_PARAMS, target, valid_days)

    def update_cr_param(
        self,
        target: str,
        value: CrossResonanceParam,
    ) -> None:
        """Update a cross-resonance parameter entry."""
        self.put_property(CR_PARAMS, target, value)

    def remove_cr_param(
        self,
        target: str,
    ) -> None:
        """Remove a cross-resonance parameter entry."""
        self.remove_property(CR_PARAMS, target)

    def get_property(
        self,
        key: str,
        target: str,
        valid_days: int | None = None,
    ) -> Any:
        """
        Return a stored property entry if still valid.

        Parameters
        ----------
        key
            Parameter category key.
        target
            Target identifier.
        valid_days
            Validity window in days.
        """
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
            logger.info(
                f"{key}['{target}'] is outdated and ignored ({days_passed} days passed)."
            )
            return None
        return value

    def put_property(
        self,
        key: str,
        target: str,
        value: Any,
    ) -> None:
        """Store a property entry under a category key."""
        self.put(key, {target: value})

    def remove_property(
        self,
        key: str,
        target: str,
    ) -> None:
        """Remove a property entry if it exists."""
        property = self.get(key)
        if property is None:
            logger.warning(f"Key '{key}' not found.")
            return
        if target not in property:
            logger.warning(f"Key '{target}' not found.")
            return
        del property[target]
        logger.info(f"Key '{target}' removed.")

    def get(
        self,
        key: str,
        valid_days: int | None = None,
    ) -> dict[str, Any]:
        """Return stored entries for a key, filtered by age."""
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
    ) -> None:
        """Store entries and add a timestamp if missing."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for v in value.values():
            if v.get("timestamp") is None:
                v["timestamp"] = timestamp
        super().put(key, value)
