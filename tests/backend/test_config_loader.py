"""Tests for system config loader."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest
import yaml

from qubex.backend.backend_controller import BACKEND_KIND_QUEL1, BACKEND_KIND_QUEL3
from qubex.backend.quel1.quel1_backend_constants import (
    DEFAULT_CAPTURE_DELAY,
    DEFAULT_CNCO_FREQ,
    DEFAULT_FNCO_FREQ,
    DEFAULT_LO_FREQ,
)
from qubex.system.config_loader import ConfigLoader
from qubex.system.control_system import CapPort, GenPort, PortType
from qubex.system.experiment_system import DEFAULT_PUMP_FREQUENCY


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _make_minimal_files(tmp_path: Path) -> tuple[Path, Path, str]:
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"

    # chip.yaml
    chip_yaml = {
        chip_id: {"name": "Test Chip", "n_qubits": 4, "clock_master": "10.0.0.1"}
    }
    _write_yaml(config_dir / "chip.yaml", chip_yaml)

    # box.yaml (one box sufficient)
    box_yaml = {
        "BOX1": {
            "name": "Box One",
            "type": "quel1-a",
            "address": "10.0.0.2",
            "adapter": "dummy",
        }
    }
    _write_yaml(config_dir / "box.yaml", box_yaml)

    # wiring.yaml — use ports compatible with QUEL1_A mapping
    wiring_yaml = {
        chip_id: [
            {
                "mux": 0,
                "read_out": "BOX1-1",  # READ_OUT
                "read_in": "BOX1-0",  # READ_IN
                # 4 control ports for 4 qubits in mux 0
                "ctrl": ["BOX1-2", "BOX1-4", "BOX1-9", "BOX1-11"],  # CTRL ports
                "pump": "BOX1-3",  # PUMP
            }
        ]
    }
    _write_yaml(config_dir / "wiring.yaml", wiring_yaml)

    # params per-file: qubit_frequency (MHz), resonator_frequency (MHz), control_amplitude (unitless)
    _write_yaml(
        params_dir / "qubit_frequency.yaml",
        {
            "meta": {"unit": "MHz", "default": 6000},
            "data": {"Q0": 5000, "Q1": None},  # Q2, Q3 missing
        },
    )
    _write_yaml(
        params_dir / "resonator_frequency.yaml",
        {
            "meta": {"unit": "MHz"},
            "data": {"Q0": 8000},
        },
    )
    _write_yaml(
        params_dir / "control_amplitude.yaml",
        {
            "meta": {},
            "data": {"Q0": 0.05},
        },
    )
    _write_yaml(
        params_dir / "frequency_margin.yaml",
        {
            "meta": {"unit": "GHz", "default": 0.1},
            "data": {"READ": 0.2},
        },
    )

    # jpa_params per-file: pass-through
    _write_yaml(
        params_dir / "jpa_params.yaml",
        {
            "meta": {},
            "data": {
                0: {"pump_frequency": 12.3, "pump_amplitude": 0.1, "dc_voltage": 0.2},
                1: None,
            },
        },
    )

    # legacy params.yaml: readout_amplitude fallback only
    _write_yaml(
        params_dir / "params.yaml",
        {
            chip_id: {
                "readout_amplitude": {"Q0": 0.02},
            }
        },
    )

    # legacy props.yaml may be empty
    _write_yaml(params_dir / "props.yaml", {})

    return config_dir, params_dir, chip_id


def test_build_experiment_system_and_unit_conversion(tmp_path: Path):
    """Given ConfigLoader, when building ExperimentSystem, then units are converted correctly."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    system = loader.get_experiment_system()

    # Quantum system built
    assert system is not None
    qs = system.quantum_system
    # Frequencies converted: 5000 MHz -> 5.0 GHz; default 6000 MHz -> 6.0 GHz
    q0 = qs.get_qubit("Q0")
    q1 = qs.get_qubit("Q1")
    q2 = qs.get_qubit("Q2")
    assert math.isclose(q0.frequency, 5.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(q1.frequency, 6.0, rel_tol=0, abs_tol=1e-9)
    assert math.isnan(q2.frequency)

    # Anharmonicity default when not specified: factor = -1/19 * frequency
    assert math.isclose(q0.anharmonicity, -5.0 / 19.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(q1.anharmonicity, -6.0 / 19.0, rel_tol=0, abs_tol=1e-9)

    # Resonator frequency for Q0 set from per-file (8000 MHz -> 8.0 GHz)
    r0 = qs.get_resonator(0)
    assert math.isclose(r0.frequency, 8.0, rel_tol=0, abs_tol=1e-9)

    # Wiring is constructed and includes ctrl/read_out/read_in/pump
    w = system.wiring_info
    assert len(w.ctrl) == 4
    assert len(w.read_out) == 1
    assert len(w.read_in) == 1
    assert len(w.pump) == 1

    # Port numbers match wiring.yaml selections
    # ctrl entries follow qubit order in mux; first corresponds to Q0 -> port 2
    first_qubit, first_ctrl_port = w.ctrl[0]
    assert first_qubit.label == "Q0"
    assert first_ctrl_port.number == 2
    (mux_ro, ro_port) = w.read_out[0]
    (mux_ri, ri_port) = w.read_in[0]
    assert ro_port.number == 1
    assert ri_port.number == 0
    assert mux_ro.index == mux_ri.index == 0


def test_control_params_sources_and_jpa_passthrough(tmp_path: Path):
    """Given ConfigLoader, when loading control params, then JPA params are passed through."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    system = loader.get_experiment_system()
    cp = system.control_params

    # Per-file control_amplitude overrides default for Q0; others fallback to DEFAULT_CONTROL_AMPLITUDE
    assert math.isclose(cp.get_control_amplitude("Q0"), 0.05, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cp.get_control_amplitude("Q3"), 0.03, rel_tol=0, abs_tol=1e-12)

    # Legacy readout_amplitude used when per-file is absent
    assert math.isclose(cp.get_readout_amplitude("Q0"), 0.02, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cp.get_readout_amplitude("Q1"), 0.01, rel_tol=0, abs_tol=1e-12)

    # jpa_params are passed through, getters reflect provided/None/default
    assert cp.jpa_params.get(0) == {
        "pump_frequency": 12.3,
        "pump_amplitude": 0.1,
        "dc_voltage": 0.2,
    }
    assert cp.jpa_params.get(1) is None
    assert math.isclose(cp.get_pump_frequency(0), 12.3, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        cp.get_pump_frequency(1), DEFAULT_PUMP_FREQUENCY, rel_tol=0, abs_tol=1e-12
    )
    assert math.isclose(cp.get_frequency_margin("READ"), 0.2, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        cp.get_frequency_margin("CTRL_GE"), 0.1, rel_tol=0, abs_tol=1e-12
    )


def test_get_experiment_system_deprecation_warning(tmp_path: Path):
    """Given deprecated arguments, when calling get_experiment_system, then ConfigLoader emits a warning."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)
    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"get_experiment_system\(chip_id\) is deprecated; the argument is ignored\. "
            r"Use get_experiment_system\(\) instead\."
        ),
    ):
        sys = loader.get_experiment_system(chip_id="IGNORED")
    assert sys is not None


def test_merge_per_file_over_legacy(tmp_path: Path):
    """Given per-file and legacy params, when loading params, then per-file values override legacy values."""
    # Arrange: start from minimal files, then add per-file readout_amplitude only for Q1
    # and set legacy props.yaml to provide qubit_frequency for Q2.
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    # Per-file readout_amplitude provides only Q1
    _write_yaml(
        params_dir / "readout_amplitude.yaml",
        {
            "meta": {},
            "data": {"Q1": 0.04},
        },
    )

    # Legacy props.yaml supplies qubit_frequency for Q2 (internal units GHz)
    _write_yaml(
        params_dir / "props.yaml",
        {
            chip_id: {
                "qubit_frequency": {"Q2": 5.5},
            }
        },
    )

    # Act
    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    system = loader.get_experiment_system()
    cp = system.control_params

    # Assert: readout_amplitude is merged: Q0 from legacy, Q1 from per-file, others default
    assert math.isclose(cp.get_readout_amplitude("Q0"), 0.02, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cp.get_readout_amplitude("Q1"), 0.04, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cp.get_readout_amplitude("Q2"), 0.01, rel_tol=0, abs_tol=1e-12)

    # Assert: qubit_frequency merged: per-file for Q0/Q1 (with unit conversion and default),
    # legacy provides Q2 (GHz, no conversion), Q3 remains NaN
    qs = system.quantum_system
    q0 = qs.get_qubit("Q0")
    q1 = qs.get_qubit("Q1")
    q2 = qs.get_qubit("Q2")
    q3 = qs.get_qubit("Q3")
    assert math.isclose(q0.frequency, 5.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(q1.frequency, 6.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(q2.frequency, 5.5, rel_tol=0, abs_tol=1e-9)
    assert math.isnan(q3.frequency)


def test_override_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Given per-file overrides, when loading params, then ConfigLoader logs an override warning."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    # Per-file readout_amplitude overrides legacy value for Q0 (legacy is 0.02)
    _write_yaml(
        params_dir / "readout_amplitude.yaml",
        {
            "meta": {},
            "data": {"Q0": 0.03},
        },
    )

    caplog.set_level(logging.WARNING, logger="qubex.system.config_loader")

    _ = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )

    # Assert a warning was logged indicating override
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(("overrides legacy" in m and "readout_amplitude" in m) for m in messages)


def test_load_param_data_applies_default_when_requested(tmp_path: Path) -> None:
    """Given per-file default values, when defaults are requested, then default values are applied."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )

    with_default = loader.load_param_data("qubit_frequency", use_default=True)
    without_default = loader.load_param_data("qubit_frequency", use_default=False)

    assert math.isclose(with_default["Q1"], 6.0, rel_tol=0, abs_tol=1e-9)
    assert without_default["Q1"] is None


def test_load_param_data_requires_structured_yaml(tmp_path: Path) -> None:
    """Given malformed per-file params, when loading params, then ValueError is raised."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    _write_yaml(params_dir / "control_amplitude.yaml", {"unexpected": 1})

    with pytest.raises(TypeError, match="Per-file params must be structured"):
        ConfigLoader(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
        )


def test_control_system_box_options_loaded_from_box_yaml(tmp_path: Path) -> None:
    """Given box options in box.yaml, when loading config, then box options are preserved."""
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"

    _write_yaml(
        config_dir / "chip.yaml",
        {chip_id: {"name": "Test Chip", "n_qubits": 4, "clock_master": "10.0.0.1"}},
    )
    _write_yaml(
        config_dir / "box.yaml",
        {
            "BOX1": {
                "name": "Box One",
                "type": "quel1se-riken8",
                "address": "10.0.0.2",
                "adapter": "dummy",
                "options": ["se8_mxfe1_awg1331", "refclk_corrected_mxfe1"],
            }
        },
    )
    _write_yaml(
        config_dir / "wiring.yaml",
        {
            chip_id: [
                {
                    "mux": 0,
                    "read_out": "BOX1-1",
                    "read_in": "BOX1-0",
                    "ctrl": ["BOX1-3", "BOX1-6", "BOX1-7", "BOX1-8"],
                    "pump": "BOX1-2",
                }
            ]
        },
    )
    _write_yaml(params_dir / "props.yaml", {})
    _write_yaml(params_dir / "params.yaml", {})

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    system = loader.get_experiment_system()
    box = system.control_system.get_box("BOX1")

    assert box.options == ("se8_mxfe1_awg1331", "refclk_corrected_mxfe1")


def test_control_system_clock_master_prefers_system_yaml(tmp_path: Path) -> None:
    """Given system.yaml and chip.yaml clock-master values, when loading control system config, then system.yaml value is used."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)
    _write_yaml(
        config_dir / "system.yaml",
        {
            "schema_version": 1,
            "chip_id": chip_id,
            "backend": "quel1",
            "quel1": {"clock_master": "10.0.0.9"},
        },
    )

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    system = loader.get_experiment_system()

    assert system.control_system.clock_master_address == "10.0.0.9"


def test_config_loader_autoload_false_requires_explicit_load(tmp_path: Path) -> None:
    """Given autoload disabled, when accessing system before load, then RuntimeError is raised."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )

    with pytest.raises(RuntimeError, match="call `load\\(\\)` first"):
        loader.get_experiment_system()

    loader.load()

    system = loader.get_experiment_system()
    assert system is not None


def test_load_uses_wiring_yaml_for_quel3_backend(tmp_path: Path) -> None:
    """Given quel3 backend, when loading, then ConfigLoader uses wiring.yaml."""
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"

    _write_yaml(
        config_dir / "chip.yaml",
        {chip_id: {"name": "Test Chip", "n_qubits": 4, "clock_master": "10.0.0.1"}},
    )
    _write_yaml(
        config_dir / "box.yaml",
        {
            "unit-a": {
                "name": "Unit A",
                "type": "quel1-a",
                "address": "10.0.0.2",
                "adapter": "dummy",
            }
        },
    )
    _write_yaml(
        config_dir / "system.yaml",
        {"schema_version": 1, "chip_id": chip_id, "backend": BACKEND_KIND_QUEL3},
    )
    _write_yaml(config_dir / "wiring.yaml", {chip_id: []})
    _write_yaml(params_dir / "props.yaml", {})
    _write_yaml(params_dir / "params.yaml", {})

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )
    loader.load()

    assert loader.backend_kind == BACKEND_KIND_QUEL3
    assert loader.wiring_file == "wiring.yaml"


def test_load_configures_quel3_readout_without_lo(tmp_path: Path) -> None:
    """Given quel3 backend, when loading, then readout ports are configured without LO."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    _write_yaml(
        config_dir / "box.yaml",
        {
            "BOX1": {
                "name": "Box One",
                "type": "quel3",
                "address": "10.0.0.2",
                "adapter": "dummy",
            }
        },
    )
    _write_yaml(
        config_dir / "system.yaml",
        {"schema_version": 1, "chip_id": chip_id, "backend": BACKEND_KIND_QUEL3},
    )

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    experiment_system = loader.get_experiment_system()
    control_system = experiment_system.control_system
    read_out_port = control_system.get_gen_port("BOX1", 1)
    read_in_port = control_system.get_cap_port("BOX1", 0)

    assert read_out_port.sideband is None
    assert read_out_port.lo_freq is None
    assert read_in_port.lo_freq is None


def test_build_target_registry_does_not_mutate_port_state(tmp_path: Path) -> None:
    """Given configured system, when rebuilding target registry, then port state remains unchanged."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)
    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    experiment_system = loader.get_experiment_system()
    read_out_port = experiment_system.control_system.get_gen_port("BOX1", 1)
    read_in_port = experiment_system.control_system.get_cap_port("BOX1", 0)

    read_out_snapshot = (
        read_out_port.lo_freq,
        read_out_port.cnco_freq,
        read_out_port.sideband,
        read_out_port.channels[0].fnco_freq,
        read_out_port.rfswitch,
    )
    read_in_snapshot = (
        read_in_port.lo_freq,
        read_in_port.cnco_freq,
        tuple(channel.fnco_freq for channel in read_in_port.channels),
        read_in_port.rfswitch,
    )

    _ = experiment_system._build_target_registry(mode="ge-cr-cr")  # noqa: SLF001

    assert (
        read_out_port.lo_freq,
        read_out_port.cnco_freq,
        read_out_port.sideband,
        read_out_port.channels[0].fnco_freq,
        read_out_port.rfswitch,
    ) == read_out_snapshot
    assert (
        read_in_port.lo_freq,
        read_in_port.cnco_freq,
        tuple(channel.fnco_freq for channel in read_in_port.channels),
        read_in_port.rfswitch,
    ) == read_in_snapshot


def test_configure_updates_port_state_before_target_registry_rebuild(
    tmp_path: Path,
) -> None:
    """Given modified ports, when configuring, then port state is recomputed and applied."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)
    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    experiment_system = loader.get_experiment_system()
    read_out_port = experiment_system.control_system.get_gen_port("BOX1", 1)
    read_in_port = experiment_system.control_system.get_cap_port("BOX1", 0)

    read_out_port.lo_freq = None
    read_out_port.cnco_freq = None
    read_out_port.channels[0].fnco_freq = None
    read_in_port.lo_freq = None
    read_in_port.cnco_freq = None
    for channel in read_in_port.channels:
        channel.fnco_freq = None

    with pytest.warns(
        DeprecationWarning,
        match=r"Use `SystemManager\.load\(\.\.\., configuration_mode=\.\.\.\)` to rebuild configuration\.",
    ):
        experiment_system.configure(mode="ge-cr-cr")

    assert read_out_port.lo_freq is not None
    assert read_out_port.cnco_freq is not None
    assert read_out_port.channels[0].fnco_freq is not None
    assert read_in_port.lo_freq is not None
    assert read_in_port.cnco_freq is not None
    assert all(channel.fnco_freq is not None for channel in read_in_port.channels)


def test_configure_initializes_monitor_ports_for_quel1(tmp_path: Path) -> None:
    """Given QuEL-1 system init, when building ports, then monitor input has initial frequencies."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)
    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )
    experiment_system = loader.get_experiment_system()
    box = experiment_system.control_system.get_box("BOX1")
    monitor_out_port = next(
        port for port in box.ports if port.type == PortType.MNTR_OUT
    )
    monitor_in_port = next(port for port in box.ports if port.type == PortType.MNTR_IN)
    assert isinstance(monitor_out_port, GenPort)
    assert isinstance(monitor_in_port, CapPort)

    assert monitor_out_port.lo_freq is None
    assert monitor_out_port.cnco_freq is None
    assert all(channel.fnco_freq is None for channel in monitor_out_port.channels)
    assert monitor_out_port.rfswitch == "pass"
    assert monitor_in_port.lo_freq == DEFAULT_LO_FREQ
    assert monitor_in_port.cnco_freq == DEFAULT_CNCO_FREQ
    assert all(
        channel.fnco_freq == DEFAULT_FNCO_FREQ for channel in monitor_in_port.channels
    )
    assert all(
        channel.ndelay == DEFAULT_CAPTURE_DELAY for channel in monitor_in_port.channels
    )
    assert monitor_in_port.rfswitch == "open"


def test_load_raises_for_system_chip_id_mismatch(tmp_path: Path) -> None:
    """Given mismatched system chip id, when loading, then ConfigLoader raises ValueError."""
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"

    _write_yaml(
        config_dir / "chip.yaml",
        {chip_id: {"name": "Test Chip", "n_qubits": 4, "clock_master": "10.0.0.1"}},
    )
    _write_yaml(
        config_dir / "box.yaml",
        {
            "unit-a": {
                "name": "Unit A",
                "type": "quel1-a",
                "address": "10.0.0.2",
                "adapter": "dummy",
            }
        },
    )
    _write_yaml(config_dir / "wiring.yaml", {chip_id: []})
    _write_yaml(
        config_dir / "system.yaml",
        {"schema_version": 1, "chip_id": "OTHER", "backend": BACKEND_KIND_QUEL1},
    )
    _write_yaml(params_dir / "props.yaml", {})
    _write_yaml(params_dir / "params.yaml", {})

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )

    with pytest.raises(ValueError, match="chip_id mismatch"):
        loader.load()


def test_load_backend_override_takes_precedence_over_system_backend(
    tmp_path: Path,
) -> None:
    """Given explicit backend override, when loading, then ConfigLoader selects override backend kind."""
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"

    _write_yaml(
        config_dir / "chip.yaml",
        {chip_id: {"name": "Test Chip", "n_qubits": 4, "clock_master": "10.0.0.1"}},
    )
    _write_yaml(
        config_dir / "box.yaml",
        {
            "unit-a": {
                "name": "Unit A",
                "type": "quel1-a",
                "address": "10.0.0.2",
                "adapter": "dummy",
            }
        },
    )
    _write_yaml(config_dir / "wiring.yaml", {chip_id: []})
    _write_yaml(
        config_dir / "system.yaml",
        {"schema_version": 1, "chip_id": chip_id, "backend": BACKEND_KIND_QUEL3},
    )
    _write_yaml(params_dir / "props.yaml", {})
    _write_yaml(params_dir / "params.yaml", {})

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )
    loader.load(backend_kind=BACKEND_KIND_QUEL1)

    assert loader.backend_kind == BACKEND_KIND_QUEL1
    assert loader.wiring_file == "wiring.yaml"


def test_load_uses_system_yaml_backend_and_ignores_chip_yaml_backend(
    tmp_path: Path,
) -> None:
    """Given backend in system.yaml and chip.yaml, when loading, then system.yaml value is selected."""
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir(parents=True)
    _write_yaml(
        config_dir / "chip.yaml",
        {chip_id: {"name": "Test Chip", "n_qubits": 4, "backend": "unknown"}},
    )
    _write_yaml(
        config_dir / "box.yaml",
        {
            "unit-a": {
                "name": "Unit A",
                "type": "quel1-a",
                "address": "10.0.0.2",
                "adapter": "dummy",
            }
        },
    )
    _write_yaml(config_dir / "wiring.yaml", {chip_id: []})
    _write_yaml(
        config_dir / "system.yaml",
        {"schema_version": 1, "chip_id": chip_id, "backend": "quel3"},
    )
    _write_yaml(params_dir / "props.yaml", {})
    _write_yaml(params_dir / "params.yaml", {})

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )
    loader.load()

    assert loader.backend_kind == "quel3"


def test_load_ignores_chip_yaml_backend_when_system_yaml_is_missing(
    tmp_path: Path,
) -> None:
    """Given backend only in chip.yaml, when loading, then quel1 default is selected."""
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir(parents=True)
    _write_yaml(
        config_dir / "chip.yaml",
        {chip_id: {"name": "Test Chip", "n_qubits": 4, "backend": "quel3"}},
    )
    _write_yaml(
        config_dir / "box.yaml",
        {
            "unit-a": {
                "name": "Unit A",
                "type": "quel1-a",
                "address": "10.0.0.2",
                "adapter": "dummy",
            }
        },
    )
    _write_yaml(config_dir / "wiring.yaml", {chip_id: []})
    _write_yaml(params_dir / "props.yaml", {})
    _write_yaml(params_dir / "params.yaml", {})

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )
    loader.load()

    assert loader.backend_kind == "quel1"


def test_load_defaults_to_quel1_when_not_configured(
    tmp_path: Path,
) -> None:
    """Given no backend config, when loading, then quel1 is selected."""
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir(parents=True)
    _write_yaml(
        config_dir / "chip.yaml", {chip_id: {"name": "Test Chip", "n_qubits": 4}}
    )
    _write_yaml(
        config_dir / "box.yaml",
        {
            "unit-a": {
                "name": "Unit A",
                "type": "quel1-a",
                "address": "10.0.0.2",
                "adapter": "dummy",
            }
        },
    )
    _write_yaml(config_dir / "wiring.yaml", {chip_id: []})
    _write_yaml(params_dir / "props.yaml", {})
    _write_yaml(params_dir / "params.yaml", {})

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )
    loader.load()

    assert loader.backend_kind == "quel1"


def test_load_raises_for_unknown_backend_value_in_system_yaml(
    tmp_path: Path,
) -> None:
    """Given unknown backend in system.yaml, when loading, then ValueError is raised."""
    chip_id = "TESTCHIP"
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir(parents=True)
    _write_yaml(
        config_dir / "chip.yaml", {chip_id: {"name": "Test Chip", "n_qubits": 4}}
    )
    _write_yaml(
        config_dir / "box.yaml",
        {
            "unit-a": {
                "name": "Unit A",
                "type": "quel1-a",
                "address": "10.0.0.2",
                "adapter": "dummy",
            }
        },
    )
    _write_yaml(config_dir / "wiring.yaml", {chip_id: []})
    _write_yaml(
        config_dir / "system.yaml",
        {"schema_version": 1, "chip_id": chip_id, "backend": "unknown"},
    )
    _write_yaml(params_dir / "props.yaml", {})
    _write_yaml(params_dir / "params.yaml", {})

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )

    with pytest.raises(ValueError, match="Unsupported backend"):
        loader.load()


def test_load_uses_quel3_system_loader_when_backend_is_quel3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quel3 backend config, when loading, then ConfigLoader delegates assembly to Quel3SystemLoader."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)
    _write_yaml(
        config_dir / "system.yaml",
        {
            "schema_version": 1,
            "chip_id": chip_id,
            "backend": "quel3",
        },
    )
    called: list[str] = []

    class _FailQuel1SystemLoader:
        def __init__(self) -> None:
            raise AssertionError(
                "Quel1SystemLoader must not be used for quel3 backend."
            )

    class _FakeQuel3SystemLoader:
        def resolve_clock_master_address(self, **_: object) -> str | None:
            called.append("clock")
            return None

        def load_control_system(self, **_: object) -> object:
            called.append("control")
            return None

        def load_wiring_info(self, **_: object) -> object:
            called.append("wiring")
            return None

    monkeypatch.setattr(
        "qubex.system.config_loader.Quel1SystemLoader",
        _FailQuel1SystemLoader,
    )
    monkeypatch.setattr(
        "qubex.system.config_loader.Quel3SystemLoader",
        _FakeQuel3SystemLoader,
    )

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )
    loader.load()

    assert called == ["clock", "control", "wiring"]


def test_load_uses_quel1_system_loader_when_backend_is_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no backend config, when loading, then ConfigLoader delegates assembly to Quel1SystemLoader."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)
    called: list[str] = []

    class _FakeQuel1SystemLoader:
        def resolve_clock_master_address(self, **_: object) -> str | None:
            called.append("clock")
            return None

        def load_control_system(self, **_: object) -> object:
            called.append("control")
            return None

        def load_wiring_info(self, **_: object) -> object:
            called.append("wiring")
            return None

    class _FailQuel3SystemLoader:
        def __init__(self) -> None:
            raise AssertionError("Quel3SystemLoader must not be used by default.")

    monkeypatch.setattr(
        "qubex.system.config_loader.Quel1SystemLoader",
        _FakeQuel1SystemLoader,
    )
    monkeypatch.setattr(
        "qubex.system.config_loader.Quel3SystemLoader",
        _FailQuel3SystemLoader,
    )

    loader = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
        autoload=False,
    )
    loader.load()

    assert called == ["clock", "control", "wiring"]
