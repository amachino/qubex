"""Plan QuEL-1 sweeps and temporarily apply native frequency settings."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from numpy.typing import ArrayLike

from qubex.backend.frequency_sweep import SweepSegment, plan_frequency_sweep
from qubex.backend.quel1.mixing import MixingUtil
from qubex.backend.quel1.quel1_backend_constants import AWG_SAFE_MAX_HZ
from qubex.backend.quel1.sweep_configuration import (
    Quel1BoxFrequencySettings,
    Quel1PortFrequencySettings,
    Quel1SweepConfiguration,
)

if TYPE_CHECKING:
    from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


class Quel1FrequencyConfigurationManager:
    """Own frequency constraints and reversible native configuration updates."""

    def __init__(self, controller: Quel1BackendController):
        self.controller = controller

    def plan(
        self,
        frequencies: ArrayLike,
        *,
        configuration: dict[str, Quel1BoxFrequencySettings],
        sideband: Literal["U", "L"] | None,
        cnco_center: int | None,
        subrange_width: float | None = None,
        overlap: bool = False,
    ) -> tuple[SweepSegment[Quel1SweepConfiguration], ...]:
        """
        Center each candidate and keep carriers within the safe AWG range.

        Include overlap references when finding the midpoint. Quantize LO/CNCO
        through MixingUtil, set FNCO to zero, and allow at most
        AWG_SAFE_MAX_HZ Hz detuning from the resulting mixed carrier.
        """

        def configure(points: tuple[float, ...]):
            lo, cnco, actual = MixingUtil.calc_lo_cnco(
                ((min(points) + max(points)) / 2) * 1e9,
                cnco_center,
                sideband,
            )
            if not all(
                actual - AWG_SAFE_MAX_HZ <= f * 1e9 <= actual + AWG_SAFE_MAX_HZ
                for f in points
            ):
                return None
            settings = deepcopy(configuration)
            for box in settings.values():
                for port in box["ports"].values():
                    port["lo_freq"], port["cnco_freq"] = lo, cnco
                    for channel in port.get(
                        "channels", port.get("runits", {})
                    ).values():
                        channel["fnco_freq"] = 0
            return (
                Quel1SweepConfiguration(settings),
                settings != configuration,
            )

        return plan_frequency_sweep(
            frequencies,
            configure=configure,
            subrange_width=subrange_width,
            overlap=overlap,
        )

    def _apply(
        self,
        box: str,
        number: int | tuple[int, int],
        settings: Quel1PortFrequencySettings,
    ) -> None:
        self.controller.config_port(
            box,
            port=number,
            lo_freq_hz=settings["lo_freq"],
            cnco_freq_hz=settings["cnco_freq"],
        )
        for channel, values in settings.get("channels", {}).items():
            self.controller.config_channel(
                box, port=number, channel=channel, fnco_freq_hz=values["fnco_freq"]
            )
        for runit, values in settings.get("runits", {}).items():
            self.controller.config_runit(
                box, port=number, runit=runit, fnco_freq_hz=values["fnco_freq"]
            )
        cache = self.controller.get_box_config_cache()
        port = cache[box]["ports"][number]
        port["lo_freq"], port["cnco_freq"] = settings["lo_freq"], settings["cnco_freq"]
        for key in ("channels", "runits"):
            for channel, values in settings.get(key, {}).items():
                port[key][channel]["fnco_freq"] = values["fnco_freq"]
        self.controller.update_box_config_cache({box: cache[box]})

    @contextmanager
    def modified(self, configuration: Quel1SweepConfiguration) -> Iterator[None]:
        """Restore touched entry settings, including after partial application."""
        cache = self.controller.get_box_config_cache()
        touched = []
        try:
            for box, settings in configuration.settings.items():
                for number, port in settings["ports"].items():
                    original = cache[box]["ports"][number]
                    saved: Quel1PortFrequencySettings = {
                        "lo_freq": original["lo_freq"],
                        "cnco_freq": original["cnco_freq"],
                    }
                    for key in ("channels", "runits"):
                        if key in port:
                            saved[key] = {
                                c: {"fnco_freq": original[key][c]["fnco_freq"]}
                                for c in port.get(key, {})
                            }
                    if saved == port:
                        continue
                    touched.append((box, number, saved))
                    self._apply(box, number, port)
            self.controller.initialize_awg_and_capunits(list(configuration.settings))
            yield
        finally:
            errors = []
            for box, number, saved in reversed(touched):
                try:
                    self._apply(box, number, saved)
                except Exception as exc:  # noqa: PERF203
                    errors.append(exc)
            if errors:
                cache = self.controller.get_box_config_cache()
                for box, _, _ in touched:
                    cache.pop(box, None)
                self.controller.replace_box_config_cache(cache)
                raise RuntimeError(
                    "Could not restore frequency configuration."
                ) from errors[0]
