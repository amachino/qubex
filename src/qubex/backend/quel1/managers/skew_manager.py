"""Skew calibration manager for QuEL-1 backend controller."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, Any, cast

import yaml
from rich.console import Console

from qubex.backend.quel1.quel1_backend_constants import (
    DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT,
)
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

if TYPE_CHECKING:
    from typing import Protocol

    from qubex.backend.quel1.compat.qubecalib_protocols import (
        Quel1BoxCommonProtocol as Quel1Box,
        Quel1SystemProtocol as Quel1System,
        SkewRuntimeProtocol,
    )

    class _SkewTargetPortMutable(Protocol):
        _target_port: set[Any]


console = Console()


class Quel1SkewManager:
    """Handle skew YAML loading and skew measurement operations for QuEL-1."""

    _WAIT_MIN = 0

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context
        self._last_skew: SkewRuntimeProtocol | None = None

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
            Target skew index. Existing `port_wait` values are shifted by
            `wait - measured_idx` for each measured port.
        box_names : list[str] | None, optional
            Box names to update. When omitted, all boxes in `box_setting` are
            updated.
        backup : bool, optional
            Whether to save the original file as `*.bak.YYYYMMDD_HHMMSS`
            before overwriting it.

        Returns
        -------
        dict[str, object]
            Summary containing the updated file path, optional backup path,
            selected box names, and target wait value.
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
            copy2(path, backup_path)

        estimated_indices = self._require_estimated_indices(
            box_names=resolved_box_names,
        )
        for box_name, port, idx in estimated_indices:
            port_wait = box_setting[box_name].setdefault("port_wait", {})
            if not isinstance(port_wait, dict):
                raise TypeError(f"box_setting.{box_name}.port_wait must be a mapping")
            current_wait = port_wait.get(port, self._WAIT_MIN)
            self._validate_wait_value(current_wait, box_name=box_name)
            port_wait[port] = max(self._WAIT_MIN, current_wait + (wait - idx))

        self._validate_wait_values(payload)
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False)
        self.load_skew_yaml(path)
        self._last_skew = None
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
        """Validate `box_setting.*.wait` and `port_wait.*` values in one skew YAML payload."""
        box_setting = cls._require_box_setting(payload)
        for box_name, setting in box_setting.items():
            if not isinstance(setting, dict):
                raise TypeError(f"box_setting.{box_name} must be a mapping")
            cls._validate_box_wait(setting, box_name=box_name)
            port_wait = cls._require_port_wait(setting, box_name=box_name)
            for port, wait in port_wait.items():
                cls._validate_port_wait_key(port, box_name=box_name)
                cls._validate_wait_value(wait, box_name=box_name)

    @classmethod
    def _validate_box_wait(cls, setting: dict[Any, Any], *, box_name: str) -> None:
        """Validate one `box_setting.<box>.wait` value required by the driver."""
        if "wait" not in setting:
            raise KeyError(f"box_setting.{box_name}.wait is required")
        wait = setting["wait"]
        if not isinstance(wait, int) or isinstance(wait, bool):
            raise TypeError(f"box_setting.{box_name}.wait must be an integer")
        if wait < cls._WAIT_MIN:
            raise ValueError(f"wait must be non-negative (box={box_name}, wait={wait})")

    @staticmethod
    def _require_port_wait(
        setting: dict[Any, Any],
        *,
        box_name: str,
    ) -> dict[Any, Any]:
        """Return the `port_wait` mapping from one box skew setting."""
        port_wait = setting.get("port_wait")
        if port_wait is None:
            # TODO: Replace this fallback when the full port_wait initialization
            # path is defined.
            return {}
        if not isinstance(port_wait, dict):
            raise TypeError(f"box_setting.{box_name}.port_wait must be a mapping")
        return port_wait

    @staticmethod
    def _validate_port_wait_key(port: object, *, box_name: str) -> None:
        """Validate one `port_wait` port key."""
        if not isinstance(port, int) or isinstance(port, bool):
            raise TypeError(f"box_setting.{box_name}.port_wait keys must be integers")

    @classmethod
    def _validate_wait_value(cls, wait: object, *, box_name: str) -> None:
        """Validate one skew wait value."""
        if not isinstance(wait, int) or isinstance(wait, bool):
            raise TypeError(
                f"box_setting.{box_name}.port_wait value must be an integer"
            )
        if wait < cls._WAIT_MIN:
            raise ValueError(f"wait must be non-negative (box={box_name}, wait={wait})")

    def _require_estimated_indices(
        self,
        *,
        box_names: list[str],
    ) -> list[tuple[str, int, int]]:
        """Return measured skew indices for selected boxes from the last scan."""
        if self._last_skew is None:
            raise RuntimeError("Run check_skew before update_skew.")

        estimated = getattr(self._last_skew, "_estimated", None)
        if not isinstance(estimated, dict):
            raise TypeError("The last skew measurement has no estimated indices.")

        selected_boxes = set(box_names)
        estimated_indices: list[tuple[str, int, int]] = []
        for port, estimated_params in estimated.items():
            if (
                not isinstance(port, tuple)
                or len(port) != 2
                or not isinstance(port[0], str)
                or not isinstance(port[1], int)
                or port[0] not in selected_boxes
            ):
                continue
            idx = getattr(estimated_params, "idx", None)
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise TypeError(f"estimated index for {port} must be an integer")
            estimated_indices.append((port[0], port[1], idx))

        if not estimated_indices:
            names = ", ".join(box_names)
            raise RuntimeError(
                f"No estimated skew indices are available for selected boxes: {names}"
            )
        return estimated_indices

    def run_skew_measurement(
        self,
        *,
        skew_yaml_path: str | Path,
        box_yaml_path: str | Path,
        clockmaster_ip: str,
        box_names: list[str],
        target_box_names: list[str] | None = None,
        estimate: bool = True,
    ) -> tuple[SkewRuntimeProtocol, Any]:
        """Run skew measurement workflow and return skew runtime and plot figure."""
        resolved_box_names = list(dict.fromkeys(box_names))
        resolved_target_box_names = (
            list(dict.fromkeys(target_box_names))
            if target_box_names is not None
            else resolved_box_names
        )
        console.print(
            f"Preparing skew measurement system for boxes: {resolved_box_names}"
        )
        system = self._build_skew_system(
            box_names=resolved_box_names,
            clockmaster_ip=clockmaster_ip,
        )
        console.print("Loading skew settings...")
        skew = self._runtime_context.driver.Skew.from_yaml(
            str(skew_yaml_path),
            box_yaml=str(box_yaml_path),
            clockmaster_ip=clockmaster_ip,
            system=system,
            boxes=[],
        )
        self._restrict_skew_targets(
            skew=skew,
            target_box_names=resolved_target_box_names,
        )
        prepare = getattr(skew, "prepare", None)
        if callable(prepare):
            console.print("Applying RF switch settings...")
            prepare()
        console.print("Resyncing skew system...")
        skew.system.resync()
        console.print("Measuring skew targets...")
        skew.measure()
        if estimate:
            console.print("Estimating skew...")
            skew.estimate()
        self._last_skew = skew
        console.print("Rendering skew plot...")
        fig = skew.plot()
        return skew, fig

    @staticmethod
    def _restrict_skew_targets(
        *,
        skew: SkewRuntimeProtocol,
        target_box_names: list[str],
    ) -> None:
        """Restrict scan targets while keeping reference/monitor boxes available."""
        target_boxes = set(target_box_names)
        target_ports = getattr(skew, "_target_port", None)
        if not isinstance(target_ports, set):
            return
        filtered_target_ports = {
            target_port
            for target_port in target_ports
            if isinstance(target_port, tuple)
            and len(target_port) >= 2
            and target_port[0] in target_boxes
        }
        target_port_skew = cast("_SkewTargetPortMutable", skew)
        target_port_skew._target_port = filtered_target_ports  # noqa: SLF001
        console.print(
            f"Skew target ports: {len(filtered_target_ports)} "
            f"from boxes {target_box_names}"
        )

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
                box.reconnect(
                    background_noise_threshold=DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT
                )
            named_boxes.append(driver.NamedBox(name=box_name, box=box))
        return driver.Quel1System.create(
            clockmaster=clockmaster,
            boxes=named_boxes,
            update_copnfig_cache=True,
        )
