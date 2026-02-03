"""Execution runner for prepared measurement schedules."""

from __future__ import annotations

from dataclasses import dataclass

from qubex.backend.quel_device_executor import QuelDeviceExecutor

from .measurement_defaults import DEFAULT_INTERVAL, DEFAULT_SHOTS
from .models import MeasureMode, MultipleMeasureResult
from .models.measurement_schedule import MeasurementSchedule


@dataclass(frozen=True)
class MeasurementRunner:
    """Run prepared measurement schedules through a device executor."""

    device_executor: QuelDeviceExecutor
    default_shots: int = DEFAULT_SHOTS
    default_interval: float = DEFAULT_INTERVAL

    def run(
        self,
        *,
        measurement_schedule: MeasurementSchedule,
        measure_mode: MeasureMode,
        shots: int | None = None,
        interval: float | None = None,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool = False,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        plot: bool = False,
    ) -> MultipleMeasureResult:
        """
        Execute a prepared measurement schedule.

        Parameters
        ----------
        measurement_schedule : MeasurementSchedule
            Prepared pulse and capture schedule pair.
        measure_mode : MeasureMode
            Measurement mode forwarded to the backend executor.
        shots : int | None, optional
            Number of shots. If omitted, `default_shots` is used.
        interval : float | None, optional
            Interval in ns. If omitted, `default_interval` is used.
        enable_dsp_demodulation : bool, optional
            Whether DSP demodulation is enabled.
        enable_dsp_sum : bool, optional
            Whether DSP summation is enabled.
        enable_dsp_classification : bool, optional
            Whether DSP classification is enabled.
        line_param0, line_param1 : tuple[float, float, float] | None, optional
            Optional DSP line parameters.
        plot : bool, optional
            Whether to plot the prepared pulse schedule.

        Returns
        -------
        MultipleMeasureResult
            Measurement result from the backend executor.
        """
        pulse_schedule = measurement_schedule.pulse_schedule
        capture_schedule = measurement_schedule.capture_schedule
        if shots is None:
            shots = self.default_shots
        if interval is None:
            interval = self.default_interval

        if not pulse_schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")

        if plot:
            pulse_schedule.plot()

        return self.device_executor.execute(
            schedule=pulse_schedule,
            capture_schedule=capture_schedule,
            measure_mode=measure_mode,
            shots=shots,
            interval=interval,
            enable_dsp_demodulation=enable_dsp_demodulation,
            enable_dsp_sum=enable_dsp_sum,
            enable_dsp_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
