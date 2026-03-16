"""Skew calibration manager for QuEL-1 backend controller."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
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

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context

    def load_skew_yaml(self, file_path: str | Path) -> None:
        """Load skew calibration YAML into qubecalib system database."""
        path = self._resolve_path(file_path)
        payload = self._load_yaml_payload(path)
        self._validate_wait_values(payload)
        self._runtime_context.qubecalib.sysdb.load_skew_yaml(str(file_path))

    def update_skew(
        self,
        *,
        file_path: str | Path,
        wait: int,
        box_names: list[str] | None = None,
        backup: bool = False,
    ) -> dict[str, object]:
        """
        Update skew waits in one YAML file and reload the runtime sysdb.

        Parameters
        ----------
        file_path : str | Path
            Path to the skew calibration YAML file.
        wait : int
            New skew wait value applied to the selected boxes.
        box_names : list[str] | None, optional
            Box names to update. When omitted, all boxes in `box_setting` are
            updated.
        backup : bool, optional
            Whether to save the original file as `*.bak` before overwriting it.

        Returns
        -------
        dict[str, object]
            Summary containing the updated file path, optional backup path,
            selected box names, and applied wait value.
        """
        self._validate_wait_value(wait, box_name="<requested>")
        path = self._resolve_path(file_path)
        payload = self._load_yaml_payload(path)
        box_setting = self._require_box_setting(payload)

        resolved_box_names = (
            list(dict.fromkeys(box_names))
            if box_names is not None
            else list(box_setting.keys())
        )
        unknown_box_names = [
            name for name in resolved_box_names if name not in box_setting
        ]
        if unknown_box_names:
            names = ", ".join(unknown_box_names)
            raise ValueError(f"Unknown box names in skew yaml: {names}")

        backup_path: Path | None = None
        if backup:
            backup_path = path.with_suffix(f"{path.suffix}.bak")
            copy2(path, backup_path)

        for box_name in resolved_box_names:
            box_setting[box_name]["wait"] = wait

        self._validate_wait_values(payload)
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False)
        self.load_skew_yaml(path)
        return {
            "file_path": path,
            "backup_path": backup_path,
            "box_names": resolved_box_names,
            "wait": wait,
        }

    @staticmethod
    def _resolve_path(file_path: str | Path) -> Path:
        """Resolve one skew-file path from cwd semantics."""
        return Path.cwd() / Path(file_path)

    @staticmethod
    def _load_yaml_payload(path: Path) -> dict[str, Any]:
        """Load one skew YAML payload from disk."""
        with path.open(encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        if not isinstance(payload, dict):
            raise TypeError("skew yaml must be a mapping")
        return payload

    @classmethod
    def _require_box_setting(cls, payload: object) -> dict[str, Any]:
        """Return the `box_setting` mapping from one skew YAML payload."""
        if not isinstance(payload, dict):
            raise TypeError("skew yaml must be a mapping")
        box_setting = payload.get("box_setting")
        if not isinstance(box_setting, dict):
            raise TypeError("skew yaml must contain `box_setting` mapping")
        return box_setting

    @classmethod
    def _validate_wait_values(cls, payload: object) -> None:
        """Validate `box_setting.*.wait` values in one skew YAML payload."""
        box_setting = cls._require_box_setting(payload)
        for box_name, setting in box_setting.items():
            if not isinstance(setting, dict):
                raise TypeError(f"box_setting.{box_name} must be a mapping")
            if "wait" not in setting:
                raise ValueError(f"box_setting.{box_name}.wait is required")
            cls._validate_wait_value(setting["wait"], box_name=box_name)

    @classmethod
    def _validate_wait_value(cls, wait: object, *, box_name: str) -> None:
        """Validate one skew wait value."""
        if not isinstance(wait, int) or isinstance(wait, bool):
            raise TypeError(f"box_setting.{box_name}.wait must be an integer")
        if wait < cls._WAIT_MIN:
            raise ValueError(f"wait must be non-negative (box={box_name}, wait={wait})")

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
            update_copnfig_cache=True,
        )
