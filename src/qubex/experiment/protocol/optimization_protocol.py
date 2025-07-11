from __future__ import annotations

from typing import Collection, Protocol

from ...measurement.measurement import DEFAULT_INTERVAL
from ...pulse import Waveform
from ...typing import TargetMap
from ..experiment_constants import CALIBRATION_SHOTS


class OptimizationProtocol(Protocol):
    def optimize_x90(
        self,
        qubit: str,
        *,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform: ...

    def optimize_drag_x90(
        self,
        qubit: str,
        *,
        duration: float = 16,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform: ...

    def optimize_pulse(
        self,
        qubit: str,
        *,
        pulse: Waveform,
        x90: Waveform,
        target_state: tuple[float, float, float],
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform: ...

    def optimize_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        opt_params: Collection[str] | None = None,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
        duration: float | None = None,
        ramptime: float | None = None,
        x180: TargetMap[Waveform] | Waveform | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ): ...
