"""Configuration loading helpers for experiment-system assembly."""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from typing_extensions import deprecated

from qubex.backend.backend_controller import (
    BACKEND_KIND_QUEL1,
    BACKEND_KIND_QUEL3,
    BackendKind,
    normalize_backend_kind,
)
from qubex.constants import (
    BOX_FILE,
    CHIP_FILE,
    DEFAULT_CONFIG_DIR,
    PARAMS_FILE,
    PROPS_FILE,
    SYSTEM_FILE,
    WIRING_FILE,
    WIRING_V2_FILE,
)
from qubex.system.quel1.quel1_system_loader import Quel1SystemLoader
from qubex.system.quel3.quel3_system_loader import Quel3SystemLoader
from qubex.system.wiring import (
    normalize_wiring_v2_rows,
)
from qubex.typing import ConfigurationMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.system.control_system import ControlSystem
    from qubex.system.experiment_system import (
        ControlParams,
        ExperimentSystem,
        WiringInfo,
    )
    from qubex.system.quantum_system import QuantumSystem

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
        Quantum chip identifier (e.g., "64Q"). All configuration is loaded

        for this specific chip.
    config_dir : Path | str | None, optional
        Directory containing configuration files (`chip.yaml`, `box.yaml`,
        `wiring.yaml`). Defaults to `DEFAULT_CONFIG_DIR/<chip_id>/config`.
    params_dir : Path | str | None, optional
        Directory containing parameter files (`params.yaml` and per-section
        files under `params/`). Defaults to
        `DEFAULT_CONFIG_DIR/<chip_id>/params`.
    chip_file, system_file, box_file, props_file, params_file : str, optional
        Filenames for the respective YAMLs. Usually left as defaults.
    wiring_file : str | None, optional
        Wiring filename override. If omitted, the loader selects a wiring file
        from resolved backend kind and config files.
    targets_to_exclude : list[str] | None, optional
        Qubit/resonator labels to exclude when assembling the ExperimentSystem.
    configuration_mode : ConfigurationMode | None, optional
        Control configuration style. Defaults to "ge-cr-cr" if not provided.
    autoload : bool, optional
        If `True`, configuration is loaded during initialization.

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
    >>> from qubex.system import ConfigLoader
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
        system_file: str = SYSTEM_FILE,
        box_file: str = BOX_FILE,
        wiring_file: str | None = None,
        props_file: str = PROPS_FILE,
        params_file: str = PARAMS_FILE,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: ConfigurationMode | None = None,
        autoload: bool = True,
    ):
        """
        Initialize configuration loader state.

        Parameters
        ----------
        chip_id : str
            Chip identifier.
        config_dir : Path | str | None, optional
            Path to configuration directory.
        params_dir : Path | str | None, optional
            Path to parameters directory.
        chip_file : str, optional
            Chip YAML filename.
        system_file : str, optional
            System YAML filename.
        box_file : str, optional
            Box YAML filename.
        wiring_file : str | None, optional
            Wiring YAML filename override.
        props_file : str, optional
            Legacy props YAML filename.
        params_file : str, optional
            Legacy params YAML filename.
        targets_to_exclude : list[str] | None, optional
            Target labels to exclude in experiment-system assembly.
        configuration_mode : ConfigurationMode | None, optional
            Configuration mode used for experiment-system assembly.
        autoload : bool, optional
            If `True`, load configuration during initialization.
        """
        if config_dir is None:
            config_dir = Path(DEFAULT_CONFIG_DIR) / chip_id / "config"
        if params_dir is None:
            params_dir = Path(DEFAULT_CONFIG_DIR) / chip_id / "params"
        if configuration_mode is None:
            configuration_mode = cast(ConfigurationMode, "ge-cr-cr")
        self._chip_id = chip_id
        self._config_dir = config_dir
        self._params_dir = params_dir
        self._chip_file = chip_file
        self._system_file = system_file
        self._box_file = box_file
        self._wiring_file = wiring_file
        self._resolved_wiring_file = wiring_file or WIRING_FILE
        self._props_file = props_file
        self._params_file = params_file
        self._default_targets_to_exclude = targets_to_exclude
        self._default_configuration_mode = configuration_mode
        self._is_loaded = False
        self._is_loading = False

        self._chip_dict: dict = {}
        self._system_dict: dict = {}
        self._box_dict: dict = {}
        self._wiring_dict: dict = {}
        self._props_dict: dict = {}
        self._params_dict: dict = {}
        self._quantum_system: QuantumSystem | None = None
        self._wiring_rows: list[dict[str, Any]] | None = None
        self._control_system: ControlSystem | None = None
        self._wiring_info: WiringInfo | None = None
        self._control_params: ControlParams | None = None
        self._experiment_system: ExperimentSystem | None = None
        self._system_loader: Quel1SystemLoader | Quel3SystemLoader | None = None
        self._backend_kind: BackendKind = BACKEND_KIND_QUEL1

        if autoload:
            self.load(
                targets_to_exclude=targets_to_exclude,
                configuration_mode=configuration_mode,
            )

    @staticmethod
    def resolve_wiring_file(
        *,
        chip_id: str,
        config_dir: Path | str | None,
        backend_kind: BackendKind,
    ) -> str:
        """
        Resolve wiring file name from backend kind and config directory.

        Parameters
        ----------
        chip_id : str
            Chip identifier.
        config_dir : Path | str | None
            Configuration directory. If `None`, default config path is used.
        backend_kind : BackendKind
            Backend family (`"quel1"` or `"quel3"`).

        Returns
        -------
        str
            Resolved wiring file name.
        """
        if backend_kind != BACKEND_KIND_QUEL3:
            return WIRING_FILE
        config_path = (
            Path(config_dir)
            if config_dir is not None
            else Path(DEFAULT_CONFIG_DIR) / chip_id / "config"
        )
        if (config_path / WIRING_V2_FILE).exists():
            return WIRING_V2_FILE
        return WIRING_FILE

    def load(
        self,
        *,
        targets_to_exclude: list[str] | None = None,
        configuration_mode: ConfigurationMode | None = None,
        backend_kind: BackendKind | None = None,
    ) -> None:
        """
        Load configuration files and build runtime system objects.

        Parameters
        ----------
        targets_to_exclude : list[str] | None, optional
            Target labels to exclude in experiment-system assembly.
        configuration_mode : ConfigurationMode | None, optional
            Configuration mode used for experiment-system assembly.
        backend_kind : BackendKind | None, optional
            Backend family override. If omitted, backend is resolved from
            `system.yaml`.
        """
        if targets_to_exclude is None:
            targets_to_exclude = self._default_targets_to_exclude
        if configuration_mode is None:
            configuration_mode = cast(
                ConfigurationMode | None,
                self._default_configuration_mode,
            )
        if configuration_mode is None:
            configuration_mode = cast(ConfigurationMode, "ge-cr-cr")
        self._is_loaded = False
        self._is_loading = True
        try:
            self._chip_dict = self._load_config_file(self._chip_file)
            self._system_dict = self._load_optional_config_file(self._system_file)
            self._backend_kind = self._resolve_loaded_backend_kind(
                backend_kind=backend_kind
            )
            self._resolved_wiring_file = self._resolve_wiring_file(
                backend_kind=self._backend_kind
            )
            self._box_dict = self._load_config_file(self._box_file)
            self._wiring_dict = self._load_config_file(self._resolved_wiring_file)
            self._props_dict = self._load_legacy_params_file(self._props_file)  # legacy
            self._params_dict = self._load_legacy_params_file(
                self._params_file
            )  # legacy
            self._system_loader = self._create_system_loader(self._backend_kind)

            self._quantum_system = self._load_quantum_system()
            self._wiring_rows = self._load_wiring_rows()
            self._control_system = self._load_control_system()
            self._wiring_info = self._load_wiring_info()
            self._control_params = self._load_control_params()
            self._experiment_system = self._load_experiment_system(
                targets_to_exclude=targets_to_exclude,
                configuration_mode=configuration_mode,
            )
            self._is_loaded = True
        finally:
            self._is_loading = False

    @property
    def config_path(self) -> Path:
        """Returns the absolute path to the configuration directory."""
        return Path(self._config_dir).resolve()

    @property
    def params_path(self) -> Path:
        """Returns the absolute path to the parameters directory."""
        return Path(self._params_dir).resolve()

    @property
    def backend_kind(self) -> BackendKind:
        """Return backend family used for this loaded configuration."""
        self._ensure_loaded()
        return self._backend_kind

    @property
    def wiring_file(self) -> str:
        """Return effective wiring file name selected for this load."""
        self._ensure_loaded()
        return self._resolved_wiring_file

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
            ExperimentSystem for this loader's chip.


        Notes
        -----
        `chip_id` is kept for backward compatibility and will be
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
        self._ensure_loaded()
        if self._experiment_system is None:
            raise RuntimeError(
                f"ExperimentSystem is not available for chip: {self._chip_id}"
            )
        return self._experiment_system

    def _ensure_loaded(self) -> None:
        """Ensure configuration payloads are loaded before access."""
        if not (self._is_loaded or self._is_loading):
            raise RuntimeError(
                "ConfigLoader is not loaded. Please call `load()` first."
            )

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

    def _load_optional_config_file(self, file_name: str) -> dict:
        """
        Load an optional configuration YAML file as a mapping.

        Parameters
        ----------
        file_name : str
            YAML file name under the config directory.

        Returns
        -------
        dict
            Parsed mapping. Returns an empty mapping when the file is missing.
        """
        path = Path(self._config_dir) / file_name
        try:
            with open(path) as file:
                result = yaml.safe_load(file)
        except FileNotFoundError:
            logger.debug("Optional config file not found; treating as empty: %s", path)
            return {}
        except yaml.YAMLError:
            logger.exception(f"Error loading configuration file: {path}")
            raise
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise TypeError(f"`{file_name}` must be a mapping at top level.")
        return result

    def _validate_loaded_system_chip_id(self) -> None:
        """Validate loaded system chip id against loader chip id."""
        configured_chip_id = self._system_dict.get("chip_id")
        if configured_chip_id is not None and str(configured_chip_id) != self._chip_id:
            raise ValueError(
                f"`{self._system_file}` chip_id mismatch: expected `{self._chip_id}`, got `{configured_chip_id}`."
            )

    def _resolve_loaded_backend_kind(
        self,
        *,
        backend_kind: BackendKind | None = None,
    ) -> BackendKind:
        """Resolve backend kind from explicit override or loaded system mapping."""
        self._validate_loaded_system_chip_id()
        if backend_kind is not None:
            normalized_override = normalize_backend_kind(backend_kind)
            if normalized_override is None:
                raise ValueError(
                    f"Unsupported backend for chip `{self._chip_id}` in explicit argument: {backend_kind!r}"
                )
            return normalized_override

        value = self._system_dict.get("backend")
        if value is not None:
            normalized = normalize_backend_kind(value)
            if normalized is not None:
                return normalized
            raise ValueError(
                f"Unsupported backend for chip `{self._chip_id}` in `{self._system_file}`: {value!r}"
            )
        return BACKEND_KIND_QUEL1

    def _resolve_wiring_file(self, *, backend_kind: BackendKind) -> str:
        """Resolve effective wiring file name for the current load."""
        if self._wiring_file is not None:
            return self._wiring_file
        return self.resolve_wiring_file(
            chip_id=self._chip_id,
            config_dir=self._config_dir,
            backend_kind=backend_kind,
        )

    def _create_system_loader(
        self,
        backend_kind: BackendKind,
    ) -> Quel1SystemLoader | Quel3SystemLoader:
        """Create a backend-specific system loader."""
        if backend_kind == BACKEND_KIND_QUEL3:
            return Quel3SystemLoader()
        return Quel1SystemLoader()

    def _resolve_system_loader(self) -> Quel1SystemLoader | Quel3SystemLoader:
        """Return active system loader for the current loaded configuration."""
        if self._system_loader is None:
            backend_kind = self._resolve_loaded_backend_kind()
            self._system_loader = self._create_system_loader(backend_kind)
        return self._system_loader

    def _resolve_clock_master_address(self) -> str | None:
        """
        Resolve clock-master address from loaded config via backend-specific loader.

        Returns
        -------
        str | None
            Resolved clock-master address. Returns `None` when unresolved.
        """
        system_loader = self._resolve_system_loader()
        return system_loader.resolve_clock_master_address(
            chip_id=self._chip_id,
            chip_dict=self._chip_dict,
            system_dict=self._system_dict,
        )

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
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise TypeError(f"`{file_name}` must be a mapping at top level.")
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
        On-disk format is preserved ("meta"/"data" keys). Downstream consumers
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
        self._ensure_loaded()
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
        from qubex.system.quantum_system import Chip, QuantumSystem

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
        system_loader = self._resolve_system_loader()
        return system_loader.load_control_system(
            chip_id=self._chip_id,
            box_dict=self._box_dict,
            wiring_rows=self._wiring_rows,
            wiring_file=self._resolved_wiring_file,
            clock_master_address=self._resolve_clock_master_address(),
        )

    def _load_wiring_info(self) -> WiringInfo | None:
        system_loader = self._resolve_system_loader()
        return system_loader.load_wiring_info(
            wiring_rows=self._wiring_rows,
            quantum_system=self._quantum_system,
            control_system=self._control_system,
        )

    def _load_wiring_rows(self) -> list[dict[str, Any]] | None:
        """Return wiring rows in canonical legacy shape."""
        chip_id = self._chip_id
        if chip_id in self._wiring_dict:
            wirings = self._wiring_dict[chip_id]
            if wirings is None:
                logger.warning(
                    f"Chip `{chip_id}` is missing in `{self._resolved_wiring_file}`. "
                )
                return None
            if not isinstance(wirings, list):
                raise TypeError(
                    f"`{self._resolved_wiring_file}` must map chip id to a list of wiring entries."
                )
            return [dict(wiring) for wiring in wirings]

        if self._wiring_dict.get("schema_version") != 2:
            logger.warning(
                f"Chip `{chip_id}` is missing in `{self._resolved_wiring_file}`. "
            )
            return None

        file_chip_id = self._wiring_dict.get("chip_id")
        if file_chip_id != chip_id:
            logger.warning(
                "Chip `%s` does not match `%s` in `%s`.",
                chip_id,
                file_chip_id,
                self._resolved_wiring_file,
            )
            return None

        if self._quantum_system is None:
            return []
        return normalize_wiring_v2_rows(
            wiring_dict=self._wiring_dict,
            wiring_file=self._resolved_wiring_file,
            qubit_indices={qubit.index for qubit in self._quantum_system.qubits},
            mux_to_qubit_indices={
                mux.index: tuple(
                    qubit.index
                    for qubit in self._quantum_system.get_qubits_in_mux(mux.index)
                )
                for mux in self._quantum_system.muxes
            },
        )

    def _load_control_params(self) -> ControlParams | None:
        from qubex.system.experiment_system import ControlParams

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
        from qubex.system.experiment_system import ExperimentSystem

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
