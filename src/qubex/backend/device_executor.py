"""Device execution abstraction and QUEL implementation."""

from __future__ import annotations

from typing import Any, Protocol

from .device_controller import DeviceController, RawResult


class DeviceExecutor(Protocol):
    """Protocol for executing sampled sequences on a device backend."""

    def execute(
        self,
        *,
        gen_sampled_sequence: dict[str, Any],
        cap_sampled_sequence: dict[str, Any],
        resource_map: dict[str, list[dict]],
        interval: float,
        repeats: int,
        integral_mode: str,
        dsp_demodulation: bool,
        enable_sum: bool,
        enable_classification: bool,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> RawResult:
        """Execute sampled sequences and return raw backend results."""
        ...


class QuelDeviceExecutor:
    """QUEL-specific executor using `SequencerMod` and `DeviceController`."""

    def __init__(self, *, device_controller: DeviceController) -> None:
        self._device_controller = device_controller

    def execute(
        self,
        *,
        gen_sampled_sequence: dict[str, Any],
        cap_sampled_sequence: dict[str, Any],
        resource_map: dict[str, list[dict]],
        interval: float,
        repeats: int,
        integral_mode: str,
        dsp_demodulation: bool,
        enable_sum: bool,
        enable_classification: bool,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> RawResult:
        """
        Execute sequences on QUEL hardware and return raw capture results.

        Parameters
        ----------
        gen_sampled_sequence : dict[str, Any]
            Per-target generator sampled sequences.
        cap_sampled_sequence : dict[str, Any]
            Per-target capture sampled sequences.
        resource_map : dict[str, list[dict]]
            Mapping from target to hardware resources.
        interval : float
            Repetition interval in ns.
        repeats : int
            Number of repeated shots.
        integral_mode : str
            Backend integral mode name.
        dsp_demodulation : bool
            Whether DSP demodulation is enabled.
        enable_sum : bool
            Whether DSP summation is enabled.
        enable_classification : bool
            Whether DSP classification is enabled.
        line_param0 : tuple[float, float, float] | None, optional
            Classifier parameter line 0.
        line_param1 : tuple[float, float, float] | None, optional
            Classifier parameter line 1.

        Returns
        -------
        RawResult
            Raw status, data, and configuration from the backend.
        """
        from qubex.backend.sequencer_mod import SequencerMod

        sequencer = SequencerMod(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map=resource_map,  # type: ignore[arg-type]
            interval=interval,
            sysdb=self._device_controller.qubecalib.sysdb,
            driver=self._device_controller.quel1system,
        )
        return self._device_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=repeats,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            enable_sum=enable_sum,
            enable_classification=enable_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
