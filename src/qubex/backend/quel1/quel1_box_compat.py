"""Compatibility adapter for quel_ic_config Quel1Box APIs across 0.8/0.10."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

ApiGeneration = Literal["quelware08", "quelware10"]


def _detect_api_generation() -> ApiGeneration:
    """Detect quel_ic_config API generation from installed package version."""
    try:
        v = version("quel_ic_config")
    except PackageNotFoundError:
        return "quelware10"
    parts = v.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        major, minor = int(parts[0]), int(parts[1])
        if major == 0 and minor < 10:
            return "quelware08"
    return "quelware10"


class _CompletedTask:
    """Minimal completed-task object compatible with `.result()` calls."""

    def result(self, timeout: float | None = None) -> None:
        """Return immediately for already-completed operation."""
        _ = timeout
        return None

    def cancel(self) -> bool:
        """No-op cancel for compatibility."""
        return False

    def done(self) -> bool:
        """Return True because the task is already completed."""
        return True


@dataclass(frozen=True)
class Quel1BoxCompatAdapter:
    """Adapt legacy and modern Quel1Box interfaces to a common API."""

    _box: Any
    _api_generation: ApiGeneration

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped box."""
        return getattr(self._box, name)

    def initialize_all_awgunits(self) -> None:
        """Initialize all AWG units across quel_ic_config versions."""
        if hasattr(self._box, "initialize_all_awgunits"):
            self._box.initialize_all_awgunits()
            return
        self._box.initialize_all_awgs()

    def initialize_all_capunits(self) -> None:
        """Initialize all capture units across quel_ic_config versions."""
        self._box.initialize_all_capunits()

    def get_current_timecounter(self) -> int:
        """Return current time counter across quel_ic_config versions."""
        if hasattr(self._box, "get_current_timecounter"):
            return int(self._box.get_current_timecounter())
        if hasattr(self._box, "read_current_clock"):
            return int(self._box.read_current_clock())
        current, _ = self._box.read_current_and_latched_clock()
        return int(current)

    def get_latest_sysref_timecounter(self) -> int:
        """Return latest SYSREF time counter across quel_ic_config versions."""
        if hasattr(self._box, "get_latest_sysref_timecounter"):
            return int(self._box.get_latest_sysref_timecounter())
        _, last_sysref = self._box.read_current_and_latched_clock()
        return int(last_sysref)

    def start_wavegen(
        self,
        channels: set[tuple[Any, int]],
        timecounter: int | None = None,
    ) -> Any:
        """Start or reserve waveform generation across API generations."""
        if hasattr(self._box, "start_wavegen"):
            return self._box.start_wavegen(channels, timecounter=timecounter)

        # legacy path
        if timecounter is None:
            self._box.start_emission(channels)
        else:
            self._box.reserve_emission(channels, timecounter)
        return _CompletedTask()


def adapt_quel1_box(box: Any) -> Quel1BoxCompatAdapter:
    """Wrap a Quel1Box-like object with version-aware compatibility adapter."""
    return Quel1BoxCompatAdapter(box, _detect_api_generation())
