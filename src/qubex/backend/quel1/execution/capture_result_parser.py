"""Capture-result parser for explicit capture-param mappings."""

from __future__ import annotations

from typing import cast

from qubex.backend.quel1.quel1_qubecalib_protocols import (
    BoxSettingProtocol,
    CaptureParamMap,
    CaptureResourceMap,
    CaptureResultKey,
    ConfigMap,
    DataMap,
    PortSettingProtocol,
    RawCaptureResultsMap,
    RawCaptureStatusMap,
    SequencerProtocol,
    StatusMap,
)


def _build_target_lookup(
    *,
    cap_resource_map: CaptureResourceMap,
) -> dict[CaptureResultKey, str]:
    """
    Build mapping from backend capture tuple keys to user-facing target labels.

    Parameters
    ----------
    cap_resource_map : CaptureResourceMap
        Resource map keyed by target label.

    Returns
    -------
    dict[CaptureResultKey, str]
        Mapping keyed by `(box_name, port, runit)`.
    """
    bpc_to_target: dict[CaptureResultKey, str] = {}
    for target, mapping in cap_resource_map.items():
        box = cast(BoxSettingProtocol, mapping["box"])
        port = cast(PortSettingProtocol, mapping["port"])
        channel = cast(int, mapping["channel_number"])
        bpc_to_target[(box.box_name, port.port, channel)] = target
    return bpc_to_target


def parse_capture_results_with_cprms(
    *,
    sequencer: SequencerProtocol,
    status: RawCaptureStatusMap,
    results: RawCaptureResultsMap,
    cap_resource_map: CaptureResourceMap,
    cprms: CaptureParamMap,
) -> tuple[StatusMap, DataMap, ConfigMap]:
    """
    Parse raw capture results using explicit capture-param mappings.

    Parameters
    ----------
    sequencer : SequencerProtocol
        Sequencer providing `parse_capture_result`.
    status : RawCaptureStatusMap
        Raw capture status keyed by `(box_name, port)`.
    results : RawCaptureResultsMap
        Raw capture data keyed by `(box_name, port, runit)`.
    cap_resource_map : CaptureResourceMap
        Target capture-resource mapping.
    cprms : CaptureParamMap
        Capture parameters keyed by `(box_name, port, runit)`.

    Returns
    -------
    tuple[StatusMap, DataMap, ConfigMap]
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

    parsed_status: StatusMap = {}
    parsed_results: DataMap = {}
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
