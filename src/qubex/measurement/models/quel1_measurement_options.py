"""QuEL-1 specific execution options."""

from __future__ import annotations

from qubex.core import Model


class Quel1MeasurementOptions(Model):
    """Optional QuEL-1 DSP options for QuEL-1 execution."""

    demodulation: bool | None = None
    classification_line_param0: tuple[float, float, float] | None = None
    classification_line_param1: tuple[float, float, float] | None = None
