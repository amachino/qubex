"""Plan QuEL-3 sweeps using deployable instrument configurations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from qubex.backend.frequency_sweep import SweepSegment, plan_frequency_sweep
from qubex.backend.quel3.models import InstrumentConfiguration
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
        reuse_current_tuning: bool = False,
        center_frequency: float | None = None,
    ) -> tuple[SweepSegment[float, Quel3SweepConfiguration], ...]:
        """Plan intervals without deploying instruments."""
        spec = next(
            s
            for s in self.controller.get_instrument_configuration().instruments
            if s.alias == target
        )
        lower, upper = (
            spec.frequency_range_min_hz / 1e9,
            spec.frequency_range_max_hz / 1e9,
        )

        def configure(points: tuple[float, ...], reference: float | None):
            required = points if reference is None else (reference, *points)
            if all(lower <= f <= upper for f in required):
                return (
                    lower,
                    upper,
                    Quel3SweepConfiguration(
                        InstrumentConfiguration(instruments=(spec,))
                    ),
                    False,
                )
            center = (
                (min(points) + max(points)) / 2
                if center_frequency is None
                else center_frequency
            )
            half_hz = (spec.frequency_range_max_hz - spec.frequency_range_min_hz) / 2
            half = half_hz / 1e9
            if center_frequency is None and not all(
                center - half <= f <= center + half for f in required
            ):
                center = (min(required) + max(required)) / 2
            updated = spec.model_copy(
                update={
                    "frequency_range_min_hz": center * 1e9 - half_hz,
                    "frequency_range_max_hz": center * 1e9 + half_hz,
                }
            )
            return (
                updated.frequency_range_min_hz / 1e9,
                updated.frequency_range_max_hz / 1e9,
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
            include_endpoints=True,
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
