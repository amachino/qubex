"""Wiring configuration helpers."""

from __future__ import annotations


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
