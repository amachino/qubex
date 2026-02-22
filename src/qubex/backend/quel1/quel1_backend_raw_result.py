"""Raw execution-result payload for QuEL-1 backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Quel1BackendRawResult:
    """Raw status, data, and config returned from qube-calib execution."""

    status: dict
    data: dict
    config: dict
