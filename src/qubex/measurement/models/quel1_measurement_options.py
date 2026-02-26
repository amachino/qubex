"""QuEL-1 specific execution options."""

from __future__ import annotations

from qubex.core import Model


class Quel1MeasurementOptions(Model):
    """Optional QuEL-1 DSP line parameters for state classification."""

    line_param0: tuple[float, float, float] | None = None
    line_param1: tuple[float, float, float] | None = None
