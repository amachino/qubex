"""Raw execution-result payload for QuEL-1 backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Quel1BackendRawResult:
    """Raw status, data, and config returned from qube-calib execution."""

    status: dict
    data: dict
    config: dict


def make_backend_raw_result(
    *,
    status: dict,
    data: dict,
    config: dict,
) -> Quel1BackendRawResult:
    """Build canonical QuEL-1 raw result container."""
    return Quel1BackendRawResult(
        status=status,
        data=data,
        config=config,
    )
