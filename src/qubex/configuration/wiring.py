"""Wiring configuration normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any


def normalize_wiring_v2_rows(
    *,
    wiring_dict: Mapping[str, Any],
    wiring_file: str,
    qubit_indices: Collection[int],
    mux_to_qubit_indices: Mapping[int, Sequence[int]],
) -> list[dict[str, Any]]:
    """
    Normalize wiring v2 schema into legacy mux-entry rows.

    Parameters
    ----------
    wiring_dict : Mapping[str, Any]
        Parsed wiring v2 payload.
    wiring_file : str
        Wiring file name for diagnostics.
    qubit_indices : Collection[int]
        Known physical qubit indices.
    mux_to_qubit_indices : Mapping[int, Sequence[int]]
        Mapping from mux index to qubit-index ordering.

    Returns
    -------
    list[dict[str, Any]]
        Legacy-shaped wiring rows (`mux`, `ctrl`, `read_out`, `read_in`, `pump`).
    """
    control_map = wiring_dict.get("control")
    readout_map = wiring_dict.get("readout")
    if not isinstance(control_map, dict):
        raise TypeError(f"`{wiring_file}` schema_version=2 requires `control` mapping.")
    if not isinstance(readout_map, dict):
        raise TypeError(f"`{wiring_file}` schema_version=2 requires `readout` mapping.")

    control_by_qubit: dict[int, str] = {}
    for raw_qubit_id, raw_port in control_map.items():
        qubit_id = _coerce_non_negative_int(
            raw_qubit_id,
            field_name="control qubit id",
        )
        control_by_qubit[qubit_id] = normalize_wiring_v2_port_specifier(
            raw_port,
            direction="out",
        )

    readout_by_mux: dict[int, dict[str, str | None]] = {}
    for raw_mux_id, raw_wiring in readout_map.items():
        mux_id = _coerce_non_negative_int(raw_mux_id, field_name="readout mux id")
        if not isinstance(raw_wiring, dict):
            raise TypeError(
                f"`{wiring_file}` readout entry for mux `{mux_id}` must be a mapping."
            )
        read_out_raw = raw_wiring.get("out")
        read_in_raw = raw_wiring.get("in")
        if read_out_raw is None or read_in_raw is None:
            raise ValueError(
                f"`{wiring_file}` readout entry for mux `{mux_id}` must include `out` and `in`."
            )
        pump_raw = raw_wiring.get("pump")
        readout_by_mux[mux_id] = {
            "read_out": normalize_wiring_v2_port_specifier(
                read_out_raw,
                direction="out",
            ),
            "read_in": normalize_wiring_v2_port_specifier(
                read_in_raw,
                direction="in",
            ),
            "pump": (
                normalize_wiring_v2_port_specifier(
                    pump_raw,
                    direction="out",
                )
                if pump_raw is not None
                else None
            ),
        }

    unknown_qubits = sorted(set(control_by_qubit).difference(set(qubit_indices)))
    if unknown_qubits:
        raise ValueError(
            f"`{wiring_file}` includes unknown control qubit ids: {unknown_qubits}"
        )

    unknown_muxes = sorted(set(readout_by_mux).difference(set(mux_to_qubit_indices)))
    if unknown_muxes:
        raise ValueError(
            f"`{wiring_file}` includes unknown readout mux ids: {unknown_muxes}"
        )

    normalized: list[dict[str, Any]] = []
    for mux_id in sorted(mux_to_qubit_indices):
        readout_entry = readout_by_mux.get(mux_id)
        if readout_entry is None:
            raise ValueError(
                f"`{wiring_file}` is missing readout entry for mux `{mux_id}`."
            )
        ctrl_ports: list[str] = []
        for qubit_id in mux_to_qubit_indices[mux_id]:
            ctrl_port = control_by_qubit.get(qubit_id)
            if ctrl_port is None:
                raise ValueError(
                    f"`{wiring_file}` is missing control port for qubit `{qubit_id}`."
                )
            ctrl_ports.append(ctrl_port)
        normalized.append(
            {
                "mux": mux_id,
                "ctrl": ctrl_ports,
                "read_out": readout_entry["read_out"],
                "read_in": readout_entry["read_in"],
                "pump": readout_entry["pump"],
            }
        )
    return normalized


def normalize_wiring_v2_port_specifier(
    specifier: Any,
    *,
    direction: str,
) -> str:
    """
    Normalize one wiring v2 port specifier into `<box>-<port>` format.

    Parameters
    ----------
    specifier : Any
        Wiring v2 port specifier (`<box>:p2tx`, `<box>:p0p1trx`, or legacy `<box>-2`).
    direction : str
        One of `in` or `out`.

    Returns
    -------
    str
        Normalized `<box>-<port>` specifier.
    """
    if not isinstance(specifier, str):
        raise TypeError("wiring v2 port specifier must be a string.")
    text = specifier.strip()
    if ":" in text:
        box_id, port_label = text.split(":", maxsplit=1)
        if box_id == "" or port_label == "":
            raise ValueError(f"Invalid wiring v2 port specifier: `{specifier}`")
        port_number = resolve_wiring_v2_port_number(
            port_label=port_label,
            direction=direction,
        )
        return f"{box_id}-{port_number}"
    return normalize_legacy_port_specifier(text)


def resolve_wiring_v2_port_number(*, port_label: str, direction: str) -> int:
    """
    Resolve wiring v2 port label (`p2tx`, `p0p1trx`, ...) into a port number.

    Parameters
    ----------
    port_label : str
        Wiring v2 port label.
    direction : str
        One of `in` or `out`.

    Returns
    -------
    int
        Legacy port number.
    """
    match = re.fullmatch(
        r"p(?P<first>\d+)(?:p(?P<second>\d+))?(?P<suffix>tx|rx|trx)",
        port_label,
    )
    if match is None:
        raise ValueError(f"Unsupported wiring v2 port specifier: `{port_label}`")
    first = int(match.group("first"))
    second_raw = match.group("second")
    second = int(second_raw) if second_raw is not None else None
    suffix = match.group("suffix")
    if suffix == "tx":
        if direction != "out":
            raise ValueError(f"Port `{port_label}` is not valid for input mapping.")
        return first
    if suffix == "rx":
        if direction != "in":
            raise ValueError(f"Port `{port_label}` is not valid for output mapping.")
        return first
    if second is None:
        return first
    return second if direction == "out" else first


def split_box_port_specifier(specifier: str) -> tuple[str, int]:
    """
    Split `<box>-<port>` specifier with support for hyphenated box IDs.

    Parameters
    ----------
    specifier : str
        Legacy `<box>-<port>` specifier.

    Returns
    -------
    tuple[str, int]
        Parsed box id and port number.
    """
    box_id, separator, port_text = specifier.rpartition("-")
    if separator == "" or box_id == "" or port_text == "":
        raise ValueError(f"Invalid port specifier: `{specifier}`")
    try:
        return box_id, int(port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid port number in specifier: `{specifier}`") from exc


def normalize_legacy_port_specifier(specifier: str) -> str:
    """
    Validate and normalize legacy `<box>-<port>` wiring specifier.

    Parameters
    ----------
    specifier : str
        Legacy wiring specifier.

    Returns
    -------
    str
        Normalized legacy wiring specifier.
    """
    box_id, port_number = split_box_port_specifier(specifier)
    return f"{box_id}-{port_number}"


def _coerce_non_negative_int(value: Any, *, field_name: str) -> int:
    """
    Convert a wiring key to a non-negative integer index.

    Parameters
    ----------
    value : Any
        Raw value.
    field_name : str
        Human-readable field name for error messages.

    Returns
    -------
    int
        Parsed integer index.
    """
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return parsed
