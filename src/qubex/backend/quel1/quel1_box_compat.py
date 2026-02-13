"""Compatibility adapter for quel_ic_config Quel1Box APIs across 0.8/0.10."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from itertools import groupby
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
class _LegacyCaptureReader:
    """Minimal legacy reader exposing a `rawwave()` method."""

    _iq: Any

    def rawwave(self) -> Any:
        """Return raw captured IQ waveform."""
        return self._iq


@dataclass(frozen=True)
class _LegacyCaptureTask:
    """Future-like wrapper converting legacy capture result shape."""

    _futures_by_port: dict[Any, Any]

    def result(self, timeout: float | None = None) -> dict[tuple[Any, int], Any]:
        """Wait for all legacy capture futures and normalize reader mapping."""
        readers: dict[tuple[Any, int], Any] = {}
        for port, future in self._futures_by_port.items():
            _status, captured = future.result(timeout=timeout)
            for runit, iq in captured.items():
                readers[(port, runit)] = _LegacyCaptureReader(iq)
        return readers

    def cancel(self) -> bool:
        """Best-effort cancellation of wrapped futures."""
        cancelled = False
        for future in self._futures_by_port.values():
            if hasattr(future, "cancel"):
                cancelled = bool(future.cancel()) or cancelled
        return cancelled

    def done(self) -> bool:
        """Return True only when all wrapped futures are completed."""
        return all(
            bool(getattr(future, "done", lambda: False)())
            for future in self._futures_by_port.values()
        )


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

    def start_capture_now(self, runits: set[tuple[Any, int]]) -> Any:
        """Start immediate capture across API generations."""
        if hasattr(self._box, "start_capture_now"):
            return self._box.start_capture_now(runits)

        # legacy path: dispatch one capture_start per port.
        futures_by_port: dict[Any, Any] = {}
        sorted_runits = sorted(runits, key=lambda item: repr(item[0]))
        for port, group in groupby(sorted_runits, key=lambda item: item[0]):
            futures_by_port[port] = self._box.capture_start(
                port=port,
                runits=[r for _, r in group],
                triggering_channel=None,
            )
        return _LegacyCaptureTask(futures_by_port)

    def start_capture_by_awg_trigger(
        self,
        runits: set[tuple[Any, int]],
        channels: set[tuple[Any, int]],
        timecounter: int | None = None,
    ) -> tuple[Any, Any]:
        """Start capture with AWG trigger across API generations."""
        if hasattr(self._box, "start_capture_by_awg_trigger"):
            return self._box.start_capture_by_awg_trigger(
                runits=runits,
                channels=channels,
                timecounter=timecounter,
            )

        # legacy path: emulate by capture_start(triggering_channel=...) + wavegen.
        if not channels:
            raise ValueError("channels must not be empty for triggered capture")
        trigger_channel = sorted(channels, key=repr)[0]
        futures_by_port: dict[Any, Any] = {}
        sorted_runits = sorted(runits, key=lambda item: repr(item[0]))
        for port, group in groupby(sorted_runits, key=lambda item: item[0]):
            futures_by_port[port] = self._box.capture_start(
                port=port,
                runits=[r for _, r in group],
                triggering_channel=trigger_channel,
            )
        cap_task = _LegacyCaptureTask(futures_by_port)
        gen_task = self.start_wavegen(channels, timecounter=timecounter)
        return cap_task, gen_task


def adapt_quel1_box(box: Any) -> Quel1BoxCompatAdapter:
    """Wrap a Quel1Box-like object with version-aware compatibility adapter."""
    return Quel1BoxCompatAdapter(box, _detect_api_generation())
