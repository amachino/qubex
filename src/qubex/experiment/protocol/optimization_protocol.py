from __future__ import annotations

from typing import Collection, Protocol

from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import Waveform
from ...typing import TargetMap


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
        objective_type: str = "st",  # "st" or "rb"
        optimize_method: str = "cma",  # "cma" or "nm"
        update_cr_param: bool = True,
        opt_params: Collection[str] | None = None,
        seed: int | None = None,
        ftarget: float | None = None,
        timeout: int | None = None,
        maxiter: int | None = None,
        n_cliffords: int | None = None,
        n_trials: int | None = None,
        duration: float | None = None,
        ramptime: float | None = None,
        x180: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ): ...
