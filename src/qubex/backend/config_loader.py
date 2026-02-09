"""Configuration loading helpers for backend systems."""

from __future__ import annotations

import logging
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from typing_extensions import deprecated

from qubex.constants import (
    BOX_FILE,
    CHIP_FILE,
    DEFAULT_CONFIG_DIR,
    PARAMS_FILE,
    PROPS_FILE,
    WIRING_FILE,
)
from qubex.typing import ConfigurationMode

from .control_system import Box, ControlSystem
from .experiment_system import ControlParams, ExperimentSystem, WiringInfo
from .quantum_system import Chip, QuantumSystem

logger = logging.getLogger(__name__)

PARAMS_MAP = {
    "qubit_frequency": ("qubit_frequency", "props"),
    "qubit_anharmonicity": ("anharmonicity", "props"),
    "resonator_frequency": ("resonator_frequency", "props"),
    "control_frequency": ("control_frequency", "props"),
    "control_frequency_ef": (None, "props"),
    "readout_frequency": (None, "props"),
    "control_amplitude": ("control_amplitude", "params"),
    "readout_amplitude": ("readout_amplitude", "params"),
    "control_vatt": ("control_vatt", "params"),
    "readout_vatt": ("readout_vatt", "params"),
    "pump_vatt": ("pump_vatt", "params"),
    "control_fsc": ("control_fsc", "params"),
    "readout_fsc": ("readout_fsc", "params"),
    "pump_fsc": ("pump_fsc", "params"),
    "capture_delay": ("capture_delay", "params"),
    "capture_delay_word": ("capture_delay_word", "params"),
    "jpa_params": ("jpa_params", "params"),
    "t1": ("t1", "props"),
    "t2_echo": ("t2_echo", "props"),
    "t2_star": ("t2_star", "props"),
    "average_readout_fidelity": ("average_readout_fidelity", "props"),
    "x90_gate_fidelity": ("x90_gate_fidelity", "props"),
    "x180_gate_fidelity": ("x180_gate_fidelity", "props"),
    "zx90_gate_fidelity": ("zx90_gate_fidelity", "props"),
    "static_zz_interaction": ("static_zz_interaction", "props"),
    "qubit_qubit_coupling_strength": ("qubit_qubit_coupling_strength", "props"),
    "resonator_external_linewidth": ("external_loss_rate", "props"),
    "resonator_internal_linewidth": ("internal_loss_rate", "props"),
}


class ConfigLoader:
    """
    Single-chip configuration loader that builds an ExperimentSystem.

    Summary
    -------
    ConfigLoader loads configuration and parameter YAML files for a specific
    quantum chip and constructs the corresponding ExperimentSystem. It prefers
    structured per-file parameter files under `params/<name>.yaml` with the
    shape `{"meta": ..., "data": ...}`. When a per-file file exists, its
    `data` is MERGED over the legacy maps in `params.yaml`/`props.yaml`
    (per-file entries override legacy values; missing keys fall back to legacy).
    If the per-file is completely absent, the legacy maps are used as-is. When
    `meta.unit` is provided, numeric values in per-file `data` are converted
    to the internal base units (GHz for frequency-like quantities, ns for
    time-like quantities). `meta.unit` must be a string and is applied
    uniformly to the values in per-file `data` only.

    The loader passes through `jpa_params` as-is; it does not perform key
    normalization. Downstream consumers should handle optional keys.

    Parameters
    ----------
    chip_id : str
        The quantum chip identifier (e.g., "64Q"). All configuration is loaded
        for this specific chip.
    config_dir : Path | str | None, optional
        Directory containing configuration files (`chip.yaml`, `box.yaml`,
        `wiring.yaml`). Defaults to `DEFAULT_CONFIG_DIR/<chip_id>/config`.
    params_dir : Path | str | None, optional
        Directory containing parameter files (`params.yaml` and per-section
        files under `params/`). Defaults to
        `DEFAULT_CONFIG_DIR/<chip_id>/params`.
    chip_file, box_file, wiring_file, props_file, params_file : str, optional
        Filenames for the respective YAMLs. Usually left as defaults.
    targets_to_exclude : list[str] | None, optional
        Qubit/resonator labels to exclude when assembling the ExperimentSystem.
    configuration_mode : ConfigurationMode | None, optional
        Control configuration style. Defaults to "ge-cr-cr" if not provided.

    Notes
    -----
    - Per-file parameter YAML must be structured as `{"meta": ..., "data": ...}`.
        - `meta.unit` is a string (applied to all numeric values in per-file `data`).
            Supported units are case-insensitive and include Hz/kHz/MHz/GHz (converted
            to GHz) and s/ms/us/µs/ns (converted to ns).
    - `get_experiment_system(chip_id)` accepts an optional argument for backward
      compatibility, but the argument is deprecated and ignored; call
      `get_experiment_system()` with no arguments.

    Examples
    --------
    >>> from qubex.backend.config_loader import ConfigLoader
    >>> cfg = ConfigLoader(chip_id="64Q")
    >>> system = cfg.get_experiment_system()
    """

    def __init__(
        self,
        *,
        chip_id: str,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        chip_file: str = CHIP_FILE,
        box_file: str = BOX_FILE,
        wiring_file: str = WIRING_FILE,
        props_file: str = PROPS_FILE,
        params_file: str = PARAMS_FILE,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: ConfigurationMode | None = None,
    ):
        if config_dir is None:
            config_dir = Path(DEFAULT_CONFIG_DIR) / chip_id / "config"
        if params_dir is None:
            params_dir = Path(DEFAULT_CONFIG_DIR) / chip_id / "params"
        if configuration_mode is None:
            configuration_mode = "ge-cr-cr"
        self._chip_id = chip_id
        self._config_dir = config_dir
        self._params_dir = params_dir

        self._chip_dict = self._load_config_file(chip_file)
        self._box_dict = self._load_config_file(box_file)
        self._wiring_dict = self._load_config_file(wiring_file)
        self._props_dict = self._load_legacy_params_file(props_file)  # legacy
        self._params_dict = self._load_legacy_params_file(params_file)  # legacy

        self._quantum_system = self._load_quantum_system()
        self._control_system = self._load_control_system()
        self._wiring_info = self._load_wiring_info()
        self._control_params = self._load_control_params()
        self._experiment_system = self._load_experiment_system(
            targets_to_exclude=targets_to_exclude,
            configuration_mode=configuration_mode,
        )

    @property
    def config_path(self) -> Path:
        """Returns the absolute path to the configuration directory."""
        return Path(self._config_dir).resolve()

    @property
    def params_path(self) -> Path:
        """Returns the absolute path to the parameters directory."""
        return Path(self._params_dir).resolve()

    def get_experiment_system(self, chip_id: str | None = None) -> ExperimentSystem:
        """
        Return the ExperimentSystem for the configured chip.

        Parameters
        ----------
        chip_id : str, optional
            Deprecated and ignored. Use `get_experiment_system()` without
            arguments. A `DeprecationWarning` is emitted when provided.

        Returns
        -------
        ExperimentSystem
            The ExperimentSystem for this loader's chip.

        Notes
        -----
        The `chip_id` parameter is kept for backward compatibility and will be
        removed in a future minor release.

        Examples
        --------
        >>> cfg = ConfigLoader(chip_id="64Q")
        >>> cfg.get_experiment_system()
        """
        if chip_id is not None:
            warnings.warn(
                "get_experiment_system(chip_id) is deprecated; the argument is ignored. "
                "Use get_experiment_system() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self._experiment_system is None:
            raise RuntimeError(
                f"ExperimentSystem is not available for chip: {self._chip_id}"
            )
        return self._experiment_system

    def _load_config_file(self, file_name: str) -> dict:
        path = Path(self._config_dir) / file_name
        try:
            with open(path) as file:
                result = yaml.safe_load(file)
        except FileNotFoundError:
            logger.exception(f"Configuration file not found: {path}")
            raise
        except yaml.YAMLError:
            logger.exception(f"Error loading configuration file: {path}")
            raise
        return result

    def _load_legacy_params_file(self, file_name: str) -> dict:
        path = Path(self._params_dir) / file_name
        try:
            with open(path) as file:
                result = yaml.safe_load(file)
        except FileNotFoundError:
            # Tolerate missing legacy params files (e.g., props.yaml, params.yaml)
            # so that per-file structured params can be used without requiring
            # the monolithic files. Return an empty mapping (no warning).
            logger.debug("Legacy parameter file not found; treating as empty: %s", path)
            return {}
        except yaml.YAMLError:
            logger.exception(f"Error loading parameter file: {path}")
            raise
        return result

    def _load_structured_params_yaml(self, path: Path) -> dict[str, dict]:
        """
        Load a YAML file that may have a structured shape {"meta": ..., "data": ...}.

        Returns
        -------
        dict[str, dict]
            A mapping with keys "meta" and "data". Both are dicts (empty if absent).

        Notes
        -----
        The on-disk format is preserved ("meta"/"data" keys). Downstream consumers
        should typically use the returned["data"]. The "meta" is returned for
        tooling/annotation but not merged into data.
        """
        try:
            with open(path) as f:
                payload = yaml.safe_load(f)
        except FileNotFoundError:
            raise
        except yaml.YAMLError:
            logger.exception(f"Error loading parameter file: {path}")
            raise

        if payload is None:
            return {"data": {}, "meta": {}}
        if isinstance(payload, dict) and ("data" in payload or "meta" in payload):
            data = payload.get("data") or {}
            meta = payload.get("meta") or {}
            if not isinstance(data, dict):
                raise TypeError(f"'data' must be a mapping in {path}")
            if not isinstance(meta, dict):
                raise TypeError(f"'meta' must be a mapping in {path}")
            return {"data": data, "meta": meta}
        raise TypeError(
            f"Per-file params must be structured with 'meta'/'data' mappings: {path}"
        )

    def _unit_scale_to_internal(self, unit: str) -> float:
        """
        Return a scale factor to convert values to internal base units.

        Internal base units are GHz for frequency-like values and ns for time-like values.
        If unit is already base or unrecognized/dimensionless, returns 1.0.

        Supported units (case-insensitive):
        - Frequency: Hz, kHz, MHz, GHz → internal GHz
        - Time: s, ms, us/µs, ns → internal ns
        """
        if not unit:
            return 1.0
        u = unit.strip().lower()
        # Frequency family → to GHz
        if u in {"hz", "khz", "mhz", "ghz"}:
            return {
                "hz": 1e-9,
                "khz": 1e-6,
                "mhz": 1e-3,
                "ghz": 1.0,
            }[u]
        # Time family → to ns
        if u in {"s", "ms", "us", "µs", "ns"}:
            return {
                "s": 1e9,
                "ms": 1e6,
                "us": 1e3,
                "µs": 1e3,
                "ns": 1.0,
            }[u]
        return 1.0

    def _convert_units_in_data(self, data: dict, unit: str | None) -> dict:
        """
        Convert numeric values in `data` according to `unit`.

        Returns a new dict; does not mutate the input.
        """

        def apply(value: Any, key: str | None = None) -> Any:
            if isinstance(unit, str):
                scale = self._unit_scale_to_internal(unit)
            else:
                scale = 1.0

            if isinstance(value, (int, float)) and scale != 1.0:
                return float(value) * scale
            else:
                return value

        if not isinstance(data, dict):
            return data
        return {k: apply(v, k) for k, v in data.items()}

    @deprecated("Use load_param_data() instead.")
    def _load_param_data(self, param_name: str, use_default: bool = True) -> dict:
        return self.load_param_data(param_name, use_default=use_default)

    def load_param_data(self, param_name: str, use_default: bool = True) -> dict:
        """
        Load a parameter dictionary with per-file preference and legacy fallback.

        Behavior
        --------
        - If a per-file YAML exists, load and unit-convert its `data` section, then
          merge over the legacy map (from props.yaml or params.yaml), so missing keys
          are filled by legacy values. Per-file keys override legacy.
        - If a per-file YAML is absent, return the legacy map as-is (or empty).
        """
        legacy_name, legacy_file = PARAMS_MAP[param_name]
        file_path = Path(self._params_dir) / f"{param_name}.yaml"

        legacy_root = self._params_dict if legacy_file == "params" else self._props_dict
        legacy_map = legacy_root.get(self._chip_id, {}) or {}
        legacy_key = legacy_name or param_name
        legacy_data = legacy_map.get(legacy_key, {}) or {}

        if file_path.exists():
            payload = self._load_structured_params_yaml(file_path)
            data = payload.get("data", {}) or {}
            meta = payload.get("meta", {}) or {}
            unit = meta.get("unit")
            default = meta.get("default")
            if use_default and default is not None:
                data = {
                    k: (
                        default
                        if (v is None or (isinstance(v, float) and math.isnan(v)))
                        else v
                    )
                    for k, v in data.items()
                }

            converted_data = self._convert_units_in_data(data, unit)

            if not converted_data:
                # Per-file exists but empty; return legacy as-is
                logger.info(
                    "Param `%s` for chip `%s`: per-file %s has no data; using legacy (%s).",
                    param_name,
                    self._chip_id,
                    file_path.name,
                    PARAMS_FILE if legacy_file == "params" else PROPS_FILE,
                )
                return legacy_data

            # Merge legacy -> per-file (per-file wins)
            merged = {**legacy_data, **converted_data}

            # Detect overrides (keys present in both with different values)
            overridden_keys = [
                k
                for k, v in converted_data.items()
                if k in legacy_data and legacy_data.get(k) != v
            ]

            # Logging: indicate source(s), warning on overrides, info otherwise.
            if legacy_data:
                if overridden_keys:
                    preview = ", ".join(sorted(overridden_keys)[:5])
                    more = (
                        ""
                        if len(overridden_keys) <= 5
                        else f" (+{len(overridden_keys) - 5} more)"
                    )
                    logger.warning(
                        "Param `%s` for chip `%s`: per-file (%s) overrides legacy (%s) for keys: %s%s.",
                        param_name,
                        self._chip_id,
                        file_path.name,
                        PARAMS_FILE if legacy_file == "params" else PROPS_FILE,
                        preview,
                        more,
                    )
                else:
                    logger.debug(
                        "Param `%s` for chip `%s`: merged per-file (%s) over legacy (%s)%s.",
                        param_name,
                        self._chip_id,
                        file_path.name,
                        PARAMS_FILE if legacy_file == "params" else PROPS_FILE,
                        f" with unit={unit!r}" if unit else "",
                    )
            else:
                logger.debug(
                    "Param `%s` for chip `%s`: loaded per-file from %s%s.",
                    param_name,
                    self._chip_id,
                    file_path.name,
                    f" with unit={unit!r}" if unit else "",
                )

            return merged
        else:
            return legacy_data

    def _load_quantum_system(self) -> QuantumSystem | None:
        chip_id = self._chip_id
        chip_info = self._chip_dict.get(chip_id)
        if chip_info is None:
            logger.warning(f"Chip `{chip_id}` is missing in `{CHIP_FILE}`. ")
            return None
        chip = Chip.new(
            id=chip_id,
            name=chip_info["name"],
            n_qubits=chip_info["n_qubits"],
        )
        qubit_frequency_dict = self.load_param_data("qubit_frequency")
        qubit_anharmonicity_dict = self.load_param_data("qubit_anharmonicity")
        resonator_frequency_dict = self.load_param_data("resonator_frequency")
        control_frequency_dict = self.load_param_data("control_frequency")
        control_frequency_ef_dict = self.load_param_data("control_frequency_ef")
        readout_frequency_dict = self.load_param_data("readout_frequency")

        # TODO: Fix SLF001
        for qubit in chip.qubits:
            qubit._bare_frequency = qubit_frequency_dict.get(qubit.label)  # noqa: SLF001
            qubit._anharmonicity = qubit_anharmonicity_dict.get(qubit.label)  # noqa: SLF001
            qubit._control_frequency_ge = control_frequency_dict.get(qubit.label)  # noqa: SLF001
            qubit._control_frequency_ef = control_frequency_ef_dict.get(qubit.label)  # noqa: SLF001
        for resonator in chip.resonators:
            resonator._frequency_g = resonator_frequency_dict.get(resonator.qubit)  # noqa: SLF001
            resonator._readout_frequency = readout_frequency_dict.get(resonator.qubit)  # noqa: SLF001
        return QuantumSystem(chip=chip)

    def _load_control_system(self) -> ControlSystem | None:
        chip_id = self._chip_id
        box_ports = defaultdict(list)
        wirings = self._wiring_dict.get(chip_id)
        if wirings is None:
            logger.warning(f"Chip `{chip_id}` is missing in `{WIRING_FILE}`. ")
            return None
        for wiring in wirings:
            box, port = wiring["read_out"].split("-")
            box_ports[box].append(int(port))
            box, port = wiring["read_in"].split("-")
            box_ports[box].append(int(port))
            for ctrl in wiring["ctrl"]:
                box, port = ctrl.split("-")
                box_ports[box].append(int(port))
        boxes = [
            Box.new(
                id=id,
                name=box["name"],
                type=box["type"],
                address=box["address"],
                adapter=box["adapter"],
                port_numbers=box_ports[id],
                options=box.get("options"),
            )
            for id, box in self._box_dict.items()
            if id in box_ports
        ]
        return ControlSystem(
            boxes=boxes,
            clock_master_address=self._chip_dict[chip_id].get("clock_master"),
        )

    def _load_wiring_info(self) -> WiringInfo | None:
        chip_id = self._chip_id
        try:
            wirings = self._wiring_dict[chip_id]
            quantum_system = self._quantum_system
            control_system = self._control_system
        except KeyError:
            return None
        if quantum_system is None or control_system is None:
            return None

        def get_gen_port(specifier: str | None):
            if specifier is None:
                return None
            box_id = specifier.split("-")[0]
            port_num = int(specifier.split("-")[1])
            port = control_system.get_gen_port(box_id, port_num)
            return port

        def get_cap_port(specifier: str | None):
            if specifier is None:
                return None
            box_id = specifier.split("-")[0]
            port_num = int(specifier.split("-")[1])
            port = control_system.get_cap_port(box_id, port_num)
            return port

        ctrl = []
        read_out = []
        read_in = []
        pump = []
        for wiring in wirings:
            mux_num = int(wiring["mux"])
            mux = quantum_system.get_mux(mux_num)
            qubits = quantum_system.get_qubits_in_mux(mux_num)
            for identifier, qubit in zip(wiring["ctrl"], qubits, strict=True):
                ctrl_port = get_gen_port(identifier)
                if ctrl_port is not None:
                    ctrl.append((qubit, ctrl_port))
            read_out_port = get_gen_port(wiring["read_out"])
            if read_out_port is not None:
                read_out.append((mux, read_out_port))
            read_in_port = get_cap_port(wiring["read_in"])
            if read_in_port is not None:
                read_in.append((mux, read_in_port))
            pump_port = get_gen_port(wiring.get("pump"))
            if pump_port is not None:
                pump.append((mux, pump_port))

        wiring_info = WiringInfo(
            ctrl=ctrl,
            read_out=read_out,
            read_in=read_in,
            pump=pump,
        )
        return wiring_info

    def _load_control_params(self) -> ControlParams | None:
        # Build control params primarily from per-file param names when present;
        # fall back to monolithic params.yaml where needed. Do not require the
        # legacy monolithic file to exist.
        control_params = ControlParams(
            control_amplitude=self.load_param_data("control_amplitude"),
            readout_amplitude=self.load_param_data("readout_amplitude"),
            control_vatt=self.load_param_data("control_vatt"),
            readout_vatt=self.load_param_data("readout_vatt"),
            pump_vatt=self.load_param_data("pump_vatt"),
            control_fsc=self.load_param_data("control_fsc"),
            readout_fsc=self.load_param_data("readout_fsc"),
            pump_fsc=self.load_param_data("pump_fsc"),
            capture_delay=self.load_param_data("capture_delay"),
            capture_delay_word=self.load_param_data("capture_delay_word"),
            jpa_params=self.load_param_data("jpa_params"),
        )
        return control_params

    def _load_experiment_system(
        self,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: ConfigurationMode = "ge-cr-cr",
    ) -> ExperimentSystem | None:
        if (
            self._quantum_system is None
            or self._control_system is None
            or self._wiring_info is None
            or self._control_params is None
        ):
            return None
        return ExperimentSystem(
            quantum_system=self._quantum_system,
            control_system=self._control_system,
            wiring_info=self._wiring_info,
            control_params=self._control_params,
            targets_to_exclude=targets_to_exclude,
            configuration_mode=configuration_mode,
        )
