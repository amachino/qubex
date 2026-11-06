"""Plan QuEL-1 sweeps and temporarily apply native frequency settings."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from numpy.typing import ArrayLike

from qubex.backend.frequency_sweep import SweepSegment, plan_frequency_sweep
from qubex.backend.quel1.mixing import MixingUtil
from qubex.backend.quel1.sweep_configuration import (
    Quel1FrequencySettings,
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
        configuration: Quel1FrequencySettings,
        sideband: Literal["U", "L"] | None,
        cnco_center: int | None,
        reuse_current_tuning: bool = False,
        subrange_width: float | None = None,
        overlap: bool = False,
        center_frequency: float | None = None,
    ) -> tuple[SweepSegment[float, Quel1SweepConfiguration], ...]:
        """Partition carriers using quantized generator and capture settings."""

        def configure(points: tuple[float, ...], reference: float | None):
            required = points if reference is None else (reference, *points)
            centers = []
            for box in configuration.values():
                for port in box["ports"].values():
                    for channel in port.get(
                        "channels", port.get("runits", {})
                    ).values():
                        if (
                            port["cnco_freq"] is not None
                            and channel["fnco_freq"] is not None
                        ):
                            centers.append(  # noqa: PERF401
                                (
                                    (port["lo_freq"] or 0)
                                    + (-1 if sideband == "L" else 1)
                                    * (port["cnco_freq"] + channel["fnco_freq"])
                                )
                                * 1e-9
                            )
            if (
                reuse_current_tuning
                and centers
                and all(
                    abs(
                        (points[0] if center_frequency is None else center_frequency)
                        - c
                    )
                    <= 0.2
                    for c in centers
                )
                and all(abs(f - c) < 0.25 for f in required for c in centers)
            ):
                return (
                    max(centers) - 0.25,
                    min(centers) + 0.25,
                    Quel1SweepConfiguration(deepcopy(configuration)),
                    False,
                )
            lo, cnco, actual = MixingUtil.calc_lo_cnco(
                (
                    (min(points) + max(points)) / 2
                    if center_frequency is None
                    else center_frequency
                )
                * 1e9,
                cnco_center,
                sideband,
            )
            settings = deepcopy(configuration)
            for box in settings.values():
                for port in box["ports"].values():
                    port["lo_freq"], port["cnco_freq"] = lo, cnco
                    for channel in port.get(
                        "channels", port.get("runits", {})
                    ).values():
                        channel["fnco_freq"] = 0
            return (
                actual * 1e-9 - 0.25,
                actual * 1e-9 + 0.25,
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
