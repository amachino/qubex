"""Models for measurement scheduling."""

from __future__ import annotations

from typing import Literal

from qxpulse import PulseSchedule
from qxvisualizer.figure import DEFAULT_TEMPLATE

from qubex.core import Model

from .capture_schedule import CaptureSchedule


class MeasurementSchedule(Model):
    """Pair of pulse and capture schedules for measurement."""

    pulse_schedule: PulseSchedule
    capture_schedule: CaptureSchedule

    def plot(
        self,
        *,
        show_physical_pulse: bool = False,
        title: str = "Measurement Schedule",
        width: int = 900,
        n_samples: int | None = None,
        divide_by_two_pi: bool = False,
        line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
        template: str = DEFAULT_TEMPLATE,
    ) -> None:
        """Plot this measurement schedule via the schedule visualizer."""
        from qubex.visualization.schedule_visualizer import plot_measurement_schedule

        plot_measurement_schedule(
            self,
            show_physical_pulse=show_physical_pulse,
            title=title,
            width=width,
            n_samples=n_samples,
            divide_by_two_pi=divide_by_two_pi,
            line_shape=line_shape,
            template=template,
        )
