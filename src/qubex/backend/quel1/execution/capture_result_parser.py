"""Capture-result parser for explicit capture-param mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, cast

PortType: TypeAlias = Any
TargetLabel: TypeAlias = str
CaptureResultKey: TypeAlias = tuple[str, PortType, int]
CaptureStatusKey: TypeAlias = tuple[str, PortType]


class _BoxRefLike(Protocol):
    """Protocol for objects exposing a box name used in capture-resource maps."""

    box_name: str


class _PortRefLike(Protocol):
    """Protocol for objects exposing a port identifier in resource maps."""

    port: PortType


class _SequencerLike(Protocol):
    """Protocol for sequencer parse API used by this parser."""

    def parse_capture_result(
        self,
        status: Any,
        data: Any,
        cprm: Any,
    ) -> tuple[Any, Any]:
        """Parse one capture result."""
        ...


CapResourceEntry: TypeAlias = Mapping[str, Any]
CapResourceMap: TypeAlias = Mapping[TargetLabel, CapResourceEntry]
CaptureStatusMap: TypeAlias = Mapping[CaptureStatusKey, Any]
CaptureResultsMap: TypeAlias = Mapping[CaptureResultKey, Any]
CaptureParamMap: TypeAlias = Mapping[CaptureResultKey, Any]


def _build_target_lookup(
    *,
    cap_resource_map: CapResourceMap,
) -> dict[CaptureResultKey, TargetLabel]:
    """
    Build mapping from backend capture tuple keys to user-facing target labels.

    Parameters
    ----------
    cap_resource_map : CapResourceMap
        Resource map keyed by target label.

    Returns
    -------
    dict[CaptureResultKey, TargetLabel]
        Mapping keyed by ``(box_name, port, runit)``.
    """
    bpc_to_target: dict[CaptureResultKey, TargetLabel] = {}
    for target, mapping in cap_resource_map.items():
        box = mapping["box"]
        port = mapping["port"]
        channel = mapping["channel_number"]
        box_name = cast(_BoxRefLike, box).box_name
        port_id = cast(_PortRefLike, port).port
        bpc_to_target[(box_name, port_id, channel)] = target
    return bpc_to_target


def parse_capture_results_with_cprms(
    *,
    sequencer: _SequencerLike,
    status: CaptureStatusMap,
    results: CaptureResultsMap,
    cap_resource_map: CapResourceMap,
    cprms: CaptureParamMap,
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

    Examples
    --------
    >>> parsed_status, parsed_results, parsed_config = parse_capture_results_with_cprms(
    ...     sequencer=sequencer,
    ...     status=raw_status,
    ...     results=raw_results,
    ...     cap_resource_map=cap_resource_map,
    ...     cprms=capture_params,
    ... )
    >>> isinstance(parsed_status, dict) and isinstance(parsed_results, dict)
    True
    """
    bpc_to_target = _build_target_lookup(cap_resource_map=cap_resource_map)

    parsed_status: dict[str, Any] = {}
    parsed_results: dict[str, Any] = {}
    for (box, port, runit), target in bpc_to_target.items():
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
