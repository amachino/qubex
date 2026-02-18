"""Tests for configuration wiring helpers."""

from __future__ import annotations

import pytest

from qubex.configuration.wiring import (
    normalize_wiring_v2_rows,
    split_box_port_specifier,
)


def test_normalize_wiring_v2_rows_returns_legacy_wiring_shape() -> None:
    """Given wiring v2 mapping, when normalizing, then legacy mux rows are produced."""
    wiring_dict = {
        "schema_version": 2,
        "chip_id": "TEST",
        "control": {
            0: "unit-a:p2tx",
            1: "unit-a:p4tx",
            2: "unit-a:p9tx",
            3: "unit-a:p11tx",
        },
        "readout": {
            0: {
                "out": "unit-a:p0p1trx",
                "in": "unit-a:p0p1trx",
                "pump": "unit-a:p3tx",
            }
        },
    }

    rows = normalize_wiring_v2_rows(
        wiring_dict=wiring_dict,
        wiring_file="wiring.v2.yaml",
        qubit_indices={0, 1, 2, 3},
        mux_to_qubit_indices={0: (0, 1, 2, 3)},
    )

    assert rows == [
        {
            "mux": 0,
            "ctrl": ["unit-a-2", "unit-a-4", "unit-a-9", "unit-a-11"],
            "read_out": "unit-a-1",
            "read_in": "unit-a-0",
            "pump": "unit-a-3",
        }
    ]


def test_split_box_port_specifier_accepts_hyphenated_box_id() -> None:
    """Given hyphenated box id, when splitting port specifier, then id and port are resolved."""
    box_id, port = split_box_port_specifier("unit-a-11")

    assert box_id == "unit-a"
    assert port == 11


def test_normalize_wiring_v2_rows_rejects_unsupported_port_specifier() -> None:
    """Given unsupported v2 port label, when normalizing, then ValueError is raised."""
    wiring_dict = {
        "schema_version": 2,
        "chip_id": "TEST",
        "control": {
            0: "unit-a:unsupported",
            1: "unit-a:p4tx",
            2: "unit-a:p9tx",
            3: "unit-a:p11tx",
        },
        "readout": {
            0: {
                "out": "unit-a:p0p1trx",
                "in": "unit-a:p0p1trx",
                "pump": "unit-a:p3tx",
            }
        },
    }

    with pytest.raises(ValueError, match="Unsupported wiring v2 port specifier"):
        normalize_wiring_v2_rows(
            wiring_dict=wiring_dict,
            wiring_file="wiring.v2.yaml",
            qubit_indices={0, 1, 2, 3},
            mux_to_qubit_indices={0: (0, 1, 2, 3)},
        )
