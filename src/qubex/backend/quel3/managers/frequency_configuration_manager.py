"""Plan QuEL-3 sweeps using deployable instrument configurations."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from qubex.backend.frequency_sweep import SweepSegment, plan_frequency_sweep
from qubex.backend.quel3.models import InstrumentConfiguration
from qubex.backend.quel3.quel3_backend_constants import (
    CONTROL_FREQUENCY_SWEEP_WIDTH_HZ,
    DEFAULT_MIN_FREQUENCY_MARGIN_GHZ,
    READOUT_FREQUENCY_SWEEP_WIDTH_HZ,
)
from qubex.backend.quel3.sweep_configuration import Quel3SweepConfiguration

if TYPE_CHECKING:
    from qubex.backend.quel3.quel3_backend_controller import Quel3BackendController


class Quel3FrequencyConfigurationManager:
    """Select instrument ranges and restore entry definitions on exit."""

    def __init__(self, controller: Quel3BackendController):
        self.controller = controller

    def plan(
        self,
        frequencies: ArrayLike,
        *,
        target: str,
        subrange_width: float | None = None,
        overlap: bool = False,
    ) -> tuple[SweepSegment[Quel3SweepConfiguration], ...]:
        """
        Plan fixed-width intervals without deploying instruments.

        Select the instrument span from its port, independently of its role
        and deployed range. Reserve the default frequency margin around every
        carrier, including overlap references. Apply `subrange_width` as an
        additional limit on the carriers.
        """
        spec = next(
            s
            for s in self.controller.get_instrument_configuration().instruments
            if s.alias == target
        )
        local_port_id = spec.port_id.partition(":")[2]
        if local_port_id.startswith("trx_"):
            width_hz = READOUT_FREQUENCY_SWEEP_WIDTH_HZ
        elif local_port_id.startswith("tx_"):
            width_hz = CONTROL_FREQUENCY_SWEEP_WIDTH_HZ
        else:
            raise ValueError(
                f"Unsupported port for frequency sweeps: {spec.port_id!r}."
            )
        min_margin_hz = DEFAULT_MIN_FREQUENCY_MARGIN_GHZ * 1e9

        def configure(points: tuple[float, ...]):
            lower, upper = min(points), max(points)
            sweep_width_ghz = (width_hz - 2 * min_margin_hz) / 1e9
            if upper - lower > sweep_width_ghz + math.ulp(lower) + math.ulp(upper):
                return None
            center = (lower * 1e9 + upper * 1e9) / 2
            half_hz = width_hz / 2
            # Round required endpoints outward to preserve inclusive reuse in GHz.
            updated = spec.model_copy(
                update={
                    "frequency_range_min_hz": min(
                        center - half_hz,
                        math.nextafter(lower * 1e9 - min_margin_hz, -math.inf),
                    ),
                    "frequency_range_max_hz": max(
                        center + half_hz,
                        math.nextafter(upper * 1e9 + min_margin_hz, math.inf),
                    ),
                }
            )
            if (
                lower < (updated.frequency_range_min_hz + min_margin_hz) / 1e9
                or upper > (updated.frequency_range_max_hz - min_margin_hz) / 1e9
            ):
                return None
            return (
                Quel3SweepConfiguration(
                    InstrumentConfiguration(instruments=(updated,))
                ),
                updated != spec,
            )

        return plan_frequency_sweep(
            frequencies,
            configure=configure,
            subrange_width=subrange_width,
            overlap=overlap,
        )

    @contextmanager
    def modified(self, configuration: Quel3SweepConfiguration) -> Iterator[None]:
        """Restore entry definitions after successful or failed deployment."""
        originals = {
            s.alias: s
            for s in self.controller.get_instrument_configuration().instruments
        }
        touched = []
        try:
            for spec in configuration.settings.instruments:
                original = originals[spec.alias]
                if spec != original:
                    touched.append(original)
                    self.controller.deploy_instrument(instrument=spec, append=True)
            yield
        finally:
            errors = []
            for original in reversed(touched):
                try:
                    self.controller.deploy_instrument(instrument=original, append=True)
                except Exception as exc:  # noqa: PERF203
                    errors.append(exc)
            if errors:
                raise RuntimeError(
                    "Could not restore instrument configuration."
                ) from errors[0]
