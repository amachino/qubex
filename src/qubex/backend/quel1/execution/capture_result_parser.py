"""Capture-result parser for explicit capture-param mappings."""

from __future__ import annotations

from typing import Any


def parse_capture_results_with_cprms(
    *,
    sequencer: Any,
    status: dict[tuple[str, Any], Any],
    results: dict[tuple[str, Any, int], Any],
    cap_resource_map: dict[str, Any],
    cprms: dict[tuple[str, Any, int], Any],
) -> tuple[dict[str, Any], dict[str, Any], dict]:
    """
    Parse raw capture results using explicit capture-param mappings.

    Parameters
    ----------
    sequencer : Any
        Sequencer providing ``parse_capture_result``.
    status : dict[tuple[str, Any], Any]
        Raw capture status keyed by ``(box_name, port)``.
    results : dict[tuple[str, Any, int], Any]
        Raw capture data keyed by ``(box_name, port, runit)``.
    cap_resource_map : dict[str, Any]
        Target capture-resource mapping.
    cprms : dict[tuple[str, Any, int], Any]
        Capture parameters keyed by ``(box_name, port, runit)``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict]
        Parsed status, parsed target data, and empty config payload.
    """
    bpc2target = {}
    for target, mapping in cap_resource_map.items():
        box = mapping["box"].box_name
        port = mapping["port"].port
        channel = mapping["channel_number"]
        bpc2target[(box, port, channel)] = target

    parsed_status: dict[str, Any] = {}
    parsed_results: dict[str, Any] = {}
    for (box, port, runit), target in bpc2target.items():
        try:
            status_item, result_item = sequencer.parse_capture_result(
                status[(box, port)],
                results[(box, port, runit)],
                cprms[(box, port, runit)],
            )
        except KeyError as exc:
            raise KeyError(
                "capture result not found: "
                f"{target}:{(box, port, runit)} in {results.keys()}, "
                f"raw_status:{status}, raw_results:{results}"
            ) from exc
        parsed_status[target] = status_item
        parsed_results[target] = result_item
    return parsed_status, parsed_results, {}
