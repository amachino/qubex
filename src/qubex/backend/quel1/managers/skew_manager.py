"""Skew calibration manager for QuEL-1 backend controller."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from qubex.backend.quel1.quel1_backend_constants import RELAXED_NOISE_THRESHOLD
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        Quel1BoxCommonProtocol as Quel1Box,
        Quel1SystemProtocol as Quel1System,
        SkewRuntimeProtocol,
    )


class Quel1SkewManager:
    """Handle skew YAML loading and skew measurement operations for QuEL-1."""

    _WAIT_MIN = 0
    _WAIT_MAX_EXCLUSIVE = 128

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context

    def load_skew_yaml(self, file_path: str | Path) -> None:
        """Load skew calibration YAML into qubecalib system database."""
        path = Path.cwd() / Path(file_path)
        with path.open(encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        self._validate_wait_range(payload)
        self._runtime_context.qubecalib.sysdb.load_skew_yaml(str(file_path))

    @classmethod
    def _validate_wait_range(cls, payload: object) -> None:
        """Validate `box_setting.*.wait` range in skew YAML payload."""
        if not isinstance(payload, dict):
            raise TypeError("skew yaml must be a mapping")
        box_setting = payload.get("box_setting")
        if not isinstance(box_setting, dict):
            raise TypeError("skew yaml must contain `box_setting` mapping")
        for box_name, setting in box_setting.items():
            if not isinstance(setting, dict):
                raise TypeError(f"box_setting.{box_name} must be a mapping")
            if "wait" not in setting:
                raise ValueError(f"box_setting.{box_name}.wait is required")
            wait = setting["wait"]
            if not isinstance(wait, int) or isinstance(wait, bool):
                raise TypeError(f"box_setting.{box_name}.wait must be an integer")
            if wait < cls._WAIT_MIN or wait >= cls._WAIT_MAX_EXCLUSIVE:
                raise ValueError(
                    f"wait must satisfy 0 <= wait < 128 (box={box_name}, wait={wait})"
                )

    def run_skew_measurement(
        self,
        *,
        skew_yaml_path: str | Path,
        box_yaml_path: str | Path,
        clockmaster_ip: str,
        box_names: list[str],
        estimate: bool = True,
    ) -> tuple[SkewRuntimeProtocol, Any]:
        """Run skew measurement workflow and return skew runtime and plot figure."""
        resolved_box_names = list(dict.fromkeys(box_names))
        system = self._build_skew_system(
            box_names=resolved_box_names,
            clockmaster_ip=clockmaster_ip,
        )
        skew = self._runtime_context.driver.Skew.from_yaml(
            str(skew_yaml_path),
            box_yaml=str(box_yaml_path),
            clockmaster_ip=clockmaster_ip,
            system=system,
            boxes=[],
        )
        skew.system.resync()
        skew.measure()
        if estimate:
            skew.estimate()
        fig = skew.plot()
        return skew, fig

    def _build_skew_system(
        self,
        *,
        box_names: list[str],
        clockmaster_ip: str,
    ) -> Quel1System:
        """Build a temporary `Quel1System` for skew measurement without db reconnect path."""
        driver = self._runtime_context.driver
        existing_boxes: dict[str, Quel1Box] = {}
        if self._runtime_context.is_connected:
            connected_system = self._runtime_context.quel1system
            clockmaster = connected_system._clockmaster  # noqa: SLF001
            existing_boxes = dict(connected_system.boxes)
        else:
            clockmaster = driver.QuBEMasterClient(clockmaster_ip)

        db = self._runtime_context.qubecalib.system_config_database
        named_boxes = []
        for box_name in box_names:
            self._runtime_context.validate_box_availability(box_name)
            box = existing_boxes.get(box_name)
            if box is None:
                box = db.create_box(box_name, reconnect=False)
                box.reconnect(background_noise_threshold=RELAXED_NOISE_THRESHOLD)
            named_boxes.append(driver.NamedBox(name=box_name, box=box))
        return driver.Quel1System.create(
            clockmaster=clockmaster,
            boxes=named_boxes,
            update_copnfig_cache=False,
        )
