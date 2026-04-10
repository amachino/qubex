"""Tests for superconducting-gap estimation helper behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from qubex.contrib.experiment.superconducting_gap import (
    get_resistance_charge,
    get_superconducting_gap,
)


def test_superconducting_gap_fills_unavailable_qubits_with_none() -> None:
    """Given unavailable qubit slots, when estimating gap, then result data stores None."""
    exp = SimpleNamespace(
        chip_id="4Qv1",
        ctx=SimpleNamespace(
            qubit_labels=["Q00", "Q01", "Q03"],
            qubits={
                "Q00": SimpleNamespace(frequency=6.0, anharmonicity=-0.3),
                "Q01": SimpleNamespace(frequency=5.5, anharmonicity=-0.25),
                "Q03": SimpleNamespace(frequency=5.2, anharmonicity=-0.2),
            },
        ),
    )

    result = get_superconducting_gap(
        exp,  # type: ignore[arg-type]
        resistance_charge={"Q00": 4700.0, "Q01": 4710.0, "Q03": 4720.0},
    )

    data = result.data["data"]
    assert isinstance(data, dict)
    assert set(data) == {"Q00", "Q01", "Q02", "Q03"}
    assert data["Q02"] is None
    assert data["Q00"] is not None
    assert data["Q01"] is not None
    assert data["Q03"] is not None


def test_superconducting_gap_raises_when_resistance_is_missing() -> None:
    """Given missing resistance, when estimating gap, then helper raises a descriptive ValueError."""
    exp = SimpleNamespace(
        chip_id="2Qv1",
        ctx=SimpleNamespace(
            qubit_labels=["Q00", "Q01"],
            qubits={
                "Q00": SimpleNamespace(frequency=6.0, anharmonicity=-0.3),
                "Q01": SimpleNamespace(frequency=5.5, anharmonicity=-0.25),
            },
        ),
    )

    with pytest.raises(ValueError, match="missing target `Q01`"):
        get_superconducting_gap(
            exp,  # type: ignore[arg-type]
            resistance_charge={"Q00": 4700.0},
        )


def test_superconducting_gap_raises_when_default_resistance_file_is_missing() -> None:
    """Given no resistance source, when default params file is absent, then helper raises FileNotFoundError."""
    missing_params_path = (
        "/home/nilton/work/work_experiments_2026_04/.tmp/nonexistent-qubex-params"
    )
    exp = SimpleNamespace(
        chip_id="2Qv1",
        config_loader=SimpleNamespace(params_path=missing_params_path),
        ctx=SimpleNamespace(
            qubit_labels=["Q00", "Q01"],
            qubits={
                "Q00": SimpleNamespace(frequency=6.0, anharmonicity=-0.3),
                "Q01": SimpleNamespace(frequency=5.5, anharmonicity=-0.25),
            },
        ),
    )

    with pytest.raises(FileNotFoundError, match="resistance_charge"):
        get_superconducting_gap(
            exp,  # type: ignore[arg-type]
            resistance_charge=None,
        )


def test_superconducting_gap_loads_cached_file_from_params(tmp_path: Path) -> None:
    """Given cached superconducting gap yaml, when helper runs, then it loads cache without resistance input."""
    params_dir = tmp_path / "test-gap-cache-load"
    params_dir.mkdir(parents=True, exist_ok=True)
    exp = SimpleNamespace(
        chip_id="4Qv1",
        config_loader=SimpleNamespace(params_path=str(params_dir)),
        ctx=SimpleNamespace(
            qubit_labels=["Q00", "Q01", "Q03"],
            qubits={
                "Q00": SimpleNamespace(frequency=6.0, anharmonicity=-0.3),
                "Q01": SimpleNamespace(frequency=5.5, anharmonicity=-0.25),
                "Q03": SimpleNamespace(frequency=5.2, anharmonicity=-0.2),
            },
        ),
    )

    cache_path = params_dir / "superconducting_gap.yaml"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"description": "cached", "unit": "ueV"},
        "data": {"Q00": 1.0, "Q01": 2.0, "Q02": None, "Q03": 3.0},
    }
    cache_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = get_superconducting_gap(
        exp,  # type: ignore[arg-type]
        resistance_charge=None,
    )

    assert result.data["meta"]["description"] == "cached"
    assert result.data["data"]["Q01"] == 2.0


def test_superconducting_gap_saves_computed_file_to_params(tmp_path: Path) -> None:
    """Given no cached superconducting gap, when helper computes, then it saves superconducting_gap.yaml to params."""
    params_dir = tmp_path / "gap-save-params"
    params_dir.mkdir(parents=True, exist_ok=True)
    exp = SimpleNamespace(
        chip_id="4Qv1",
        config_loader=SimpleNamespace(params_path=str(params_dir)),
        ctx=SimpleNamespace(
            qubit_labels=["Q00", "Q01", "Q03"],
            qubits={
                "Q00": SimpleNamespace(frequency=6.0, anharmonicity=-0.3),
                "Q01": SimpleNamespace(frequency=5.5, anharmonicity=-0.25),
                "Q03": SimpleNamespace(frequency=5.2, anharmonicity=-0.2),
            },
        ),
    )

    get_superconducting_gap(
        exp,  # type: ignore[arg-type]
        resistance_charge={"Q00": 4700.0, "Q01": 4710.0, "Q03": 4720.0},
    )

    saved_path = params_dir / "superconducting_gap.yaml"
    assert saved_path.exists()


def test_resistance_charge_loads_from_default_params_file(tmp_path: Path) -> None:
    """Given default params resistance file, when helper runs, then it loads and pads unavailable slots."""
    params_dir = tmp_path / "resistance-default-params"
    params_dir.mkdir(parents=True, exist_ok=True)
    resistance_path = params_dir / "resistance_charge.yaml"
    payload = {
        "meta": {"description": "Resistance charge after annealing", "unit": "ohms"},
        "data": {"Q00": 4700.0, "Q01": 4710.0, "Q03": 4720.0},
    }
    resistance_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    exp = SimpleNamespace(
        chip_id="4Qv1",
        config_loader=SimpleNamespace(params_path=str(params_dir)),
        ctx=SimpleNamespace(
            qubit_labels=["Q00", "Q01", "Q03"],
            qubits={
                "Q00": SimpleNamespace(frequency=6.0, anharmonicity=-0.3),
                "Q01": SimpleNamespace(frequency=5.5, anharmonicity=-0.25),
                "Q03": SimpleNamespace(frequency=5.2, anharmonicity=-0.2),
            },
        ),
    )

    result = get_resistance_charge(
        exp,  # type: ignore[arg-type]
        resistance_charge=None,
    )
    data = result.data["data"]
    assert isinstance(data, dict)
    assert set(data) == {"Q00", "Q01", "Q02", "Q03"}
    assert data["Q02"] is None
    assert data["Q00"] == 4700.0


def test_resistance_charge_raises_when_default_file_is_missing() -> None:
    """Given no source and missing default file, when helper runs, then FileNotFoundError is raised."""
    missing_params_path = (
        "/home/nilton/work/work_experiments_2026_04/.tmp/nonexistent-resistance-params"
    )
    exp = SimpleNamespace(
        chip_id="2Qv1",
        config_loader=SimpleNamespace(params_path=missing_params_path),
        ctx=SimpleNamespace(
            qubit_labels=["Q00", "Q01"],
            qubits={
                "Q00": SimpleNamespace(frequency=6.0, anharmonicity=-0.3),
                "Q01": SimpleNamespace(frequency=5.5, anharmonicity=-0.25),
            },
        ),
    )

    with pytest.raises(FileNotFoundError, match="resistance_charge"):
        get_resistance_charge(
            exp,  # type: ignore[arg-type]
            resistance_charge=None,
        )


def test_resistance_charge_accepts_numeric_index_keys() -> None:
    """Given numeric index keys, when loading resistance values, then helper maps them to qubit labels."""
    exp = SimpleNamespace(
        chip_id="4Qv1",
        ctx=SimpleNamespace(
            qubit_labels=["Q00", "Q01", "Q03"],
            qubits={
                "Q00": SimpleNamespace(frequency=6.0, anharmonicity=-0.3),
                "Q01": SimpleNamespace(frequency=5.5, anharmonicity=-0.25),
                "Q03": SimpleNamespace(frequency=5.2, anharmonicity=-0.2),
            },
        ),
    )

    result = get_resistance_charge(
        exp,  # type: ignore[arg-type]
        resistance_charge={0: 4700.0, "1": 4710.0, 3: 4720.0},
    )

    data = result.data["data"]
    assert isinstance(data, dict)
    assert data["Q00"] == 4700.0
    assert data["Q01"] == 4710.0
    assert data["Q02"] is None
    assert data["Q03"] == 4720.0
