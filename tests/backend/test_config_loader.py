from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest
import yaml

from qubex.backend.config_loader import ConfigLoader
from qubex.backend.experiment_system import DEFAULT_PUMP_FREQUENCY


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
    """ConfigLoader should build ExperimentSystem and convert units correctly."""
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
    """ConfigLoader should load control params and pass through JPA params."""
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


def test_get_experiment_system_deprecation_warning(tmp_path: Path):
    """ConfigLoader should warn when deprecated arguments are used."""
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
    """ConfigLoader should merge per-file params over legacy params."""
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
    """ConfigLoader should log a warning when legacy params are overridden."""
    config_dir, params_dir, chip_id = _make_minimal_files(tmp_path)

    # Per-file readout_amplitude overrides legacy value for Q0 (legacy is 0.02)
    _write_yaml(
        params_dir / "readout_amplitude.yaml",
        {
            "meta": {},
            "data": {"Q0": 0.03},
        },
    )

    caplog.set_level(logging.WARNING, logger="qubex.backend.config_loader")

    _ = ConfigLoader(
        chip_id=chip_id,
        config_dir=config_dir,
        params_dir=params_dir,
    )

    # Assert a warning was logged indicating override
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(("overrides legacy" in m and "readout_amplitude" in m) for m in messages)
