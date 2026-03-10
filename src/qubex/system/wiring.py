"""Wiring configuration helpers."""

from __future__ import annotations


def split_box_port_specifier(specifier: str) -> tuple[str, int]:
    """
    Split one wiring port specifier into box id and port number.

    Parameters
    ----------
    specifier : str
        Preferred `<box>:<port>` specifier. Legacy `<box>-<port>` remains
        supported for compatibility.

    Returns
    -------
    tuple[str, int]
        Parsed box id and port number.
    """
    separator = ":" if ":" in specifier else "-"
    box_id, found_separator, port_text = specifier.rpartition(separator)
    if found_separator == "" or box_id == "" or port_text == "":
        raise ValueError(f"Invalid port specifier: `{specifier}`")
    try:
        return box_id, int(port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid port number in specifier: `{specifier}`") from exc
