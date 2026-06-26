"""Session-local measurement stability state and corrections."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from inspect import Parameter, signature
from typing import Any, NamedTuple, cast

import numpy as np
from qxpulse import FlatTop, PulseSchedule

from qubex.measurement.measurement_context import MeasurementContext
from qubex.measurement.measurement_defaults import (
    DEFAULT_OUTPUT_GAIN_CORRECTION_DEADBAND,
    DEFAULT_OUTPUT_GAIN_CORRECTION_DEADBAND_SIGMA,
    DEFAULT_OUTPUT_GAIN_CORRECTION_MAX_RELATIVE_STEP,
    DEFAULT_OUTPUT_GAIN_CORRECTION_SMOOTHING,
    DEFAULT_OUTPUT_PHASE_CORRECTION_DEADBAND,
    DEFAULT_OUTPUT_PHASE_CORRECTION_DEADBAND_SIGMA,
    DEFAULT_OUTPUT_PHASE_CORRECTION_MAX_STEP,
    DEFAULT_OUTPUT_PHASE_CORRECTION_SMOOTHING,
    DEFAULT_OUTPUT_PHASE_MIN_RESULTANT_LENGTH,
    DEFAULT_STABILITY_CORRECTION_N_SHOTS,
    DEFAULT_STABILITY_CORRECTION_PROBE_DURATION,
)
from qubex.measurement.models.measure_result import MultipleMeasureResult
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.measurement_stability import (
    MeasurementStabilitySnapshot,
    MonitorStatistic,
    OutputSignalCorrection,
    OutputSignalKind,
    OutputSignalReferenceScope,
)
from qubex.measurement.services.measurement_monitor_service import (
    MeasurementMonitorService,
)
from qubex.system import PortType, Target
from qubex.visualization.style import FONT_FAMILY

logger = logging.getLogger(__name__)

_LIVE_STABILITY_PLOT_WIDTH = 800
_LIVE_STABILITY_PLOT_HEIGHT = 520
_LIVE_WAVEFORM_PLOT_WIDTH = 800
_LIVE_WAVEFORM_PLOT_HEIGHT = 320
_LIVE_AMPLITUDE_COLOR = "#0C5DA5"
_LIVE_PHASE_COLOR = "#00B945"


class _MonitorWaveform(NamedTuple):
    reference_target: str
    monitor_target: str
    capture_index: int
    time_ns: np.ndarray
    amplitude: np.ndarray


def _capture_array(capture: Any) -> np.ndarray:
    """Return a NumPy array from canonical or legacy capture objects."""
    if hasattr(capture, "data"):
        return np.asarray(capture.data)
    if hasattr(capture, "raw"):
        return np.asarray(capture.raw)
    return np.asarray(capture)


def _trim_capture(array: np.ndarray, trim_samples: int) -> np.ndarray:
    """Trim capture edges along the sample axis."""
    if trim_samples < 0:
        raise ValueError("trim_samples must be non-negative.")
    if trim_samples == 0:
        return array
    if array.shape[-1] <= 2 * trim_samples:
        raise ValueError("trim_samples removes all monitor samples.")
    return array[..., trim_samples:-trim_samples]


def _phase_statistics(samples: np.ndarray) -> tuple[float, float, float]:
    """Return circular phase statistics and mean resultant length."""
    amplitudes = np.abs(samples)
    valid = amplitudes > 0.0
    if not np.any(valid):
        return np.nan, np.nan, 0.0

    unit_samples = samples[valid] / amplitudes[valid]
    mean_vector = np.mean(unit_samples)
    resultant_length = float(np.abs(mean_vector))
    if resultant_length <= np.finfo(float).eps:
        return np.nan, np.nan, resultant_length

    mean_phase = float(np.angle(mean_vector))
    offsets = np.angle(unit_samples * np.exp(-1j * mean_phase))
    return mean_phase, float(np.std(offsets)), resultant_length


def _signal_stability_series(
    snapshots: Collection[MeasurementStabilitySnapshot],
) -> dict[str, tuple[list[float], list[float], list[float], list[float]]]:
    """Return elapsed, relative amplitude, percent move, and relative phase."""
    raw_series: dict[str, list[tuple[float, float, float]]] = {}
    for snapshot in snapshots:
        elapsed_s = 0.0 if snapshot.elapsed_s is None else float(snapshot.elapsed_s)
        for statistic in snapshot.signals.values():
            label = (
                f"{statistic.reference_target} -> {statistic.monitor_target}"
                f" [{statistic.capture_index}]"
            )
            raw_series.setdefault(label, []).append(
                (
                    elapsed_s,
                    float(statistic.amplitude_mean),
                    float(statistic.phase_mean_rad),
                )
            )

    series: dict[str, tuple[list[float], list[float], list[float], list[float]]] = {}
    for label, points in raw_series.items():
        baseline = points[0][1]
        if not np.isfinite(baseline) or baseline == 0.0:
            continue
        x = [point[0] for point in points]
        y = [point[1] / baseline for point in points]
        percent = [100.0 * (value - 1.0) for value in y]
        phases = np.asarray([point[2] for point in points], dtype=np.float64)
        relative_phase = np.full(phases.shape, np.nan, dtype=np.float64)
        finite_phase = np.isfinite(phases)
        if np.any(finite_phase):
            unwrapped_phase = np.unwrap(phases[finite_phase])
            relative_phase[finite_phase] = unwrapped_phase - unwrapped_phase[0]
        series[label] = (x, y, percent, relative_phase.tolist())
    return series


def _add_signal_stability_trace(
    fig: Any,
    *,
    label: str,
    x: list[float],
    y: list[float],
    percent: list[float],
    phase: list[float],
) -> None:
    fig.add_scatter(
        x=x,
        y=y,
        customdata=percent,
        mode="lines+markers",
        name=label,
        line={"color": _LIVE_AMPLITUDE_COLOR},
        hovertemplate=(
            "elapsed=%{x:.3f}s<br>"
            "relative=%{y:.8f}<br>"
            "move=%{customdata:+.4f}%<extra>%{fullData.name}</extra>"
        ),
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=x,
        y=phase,
        mode="lines+markers",
        name=f"{label} phase",
        line={"color": _LIVE_PHASE_COLOR},
        hovertemplate=(
            "elapsed=%{x:.3f}s<br>phase=%{y:+.6f}rad<extra>%{fullData.name}</extra>"
        ),
        row=2,
        col=1,
    )


def _apply_signal_stability_layout(fig: Any) -> None:
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(
        title="Output signal stability",
        template="qubex",
        width=_LIVE_STABILITY_PLOT_WIDTH,
        height=_LIVE_STABILITY_PLOT_HEIGHT,
        font={"family": FONT_FAMILY},
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="relative amplitude (initial=1)", row=1, col=1)
    fig.update_yaxes(title_text="phase shift (rad, initial=0)", row=2, col=1)
    fig.update_xaxes(title_text="elapsed time (s)", row=2, col=1)


def _make_signal_stability_figure(
    snapshots: Collection[MeasurementStabilitySnapshot],
) -> Any:
    """Return a relative monitor amplitude and phase figure."""
    from plotly.subplots import make_subplots

    import qubex.visualization  # noqa: F401

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.58, 0.42],
    )
    for label, (x, y, percent, phase) in _signal_stability_series(snapshots).items():
        _add_signal_stability_trace(
            fig,
            x=x,
            y=y,
            percent=percent,
            phase=phase,
            label=label,
        )
    _apply_signal_stability_layout(fig)
    return fig


def _make_signal_stability_widget(
    snapshots: Collection[MeasurementStabilitySnapshot],
) -> Any:
    """Return a live-updated relative monitor amplitude and phase widget."""
    import plotly.graph_objects as go

    return go.FigureWidget(_make_signal_stability_figure(snapshots))


def _update_signal_stability_widget(
    widget: Any,
    snapshots: Collection[MeasurementStabilitySnapshot],
) -> None:
    """Update the live stability widget in place."""
    series = _signal_stability_series(snapshots)
    trace_names = [
        trace_name for label in series for trace_name in (label, f"{label} phase")
    ]
    with widget.batch_update() if hasattr(widget, "batch_update") else nullcontext():
        if [trace.name for trace in widget.data] != trace_names:
            widget.data = ()
            for label, (x, y, percent, phase) in series.items():
                _add_signal_stability_trace(
                    widget,
                    x=x,
                    y=y,
                    percent=percent,
                    phase=phase,
                    label=label,
                )
            return
        for index, (x, y, percent, phase) in enumerate(series.values()):
            amplitude_trace = widget.data[2 * index]
            phase_trace = widget.data[2 * index + 1]
            amplitude_trace.x = x
            amplitude_trace.y = y
            amplitude_trace.customdata = percent
            phase_trace.x = x
            phase_trace.y = phase


def _waveform_trace_label(waveform: _MonitorWaveform) -> str:
    return (
        f"{waveform.reference_target} -> {waveform.monitor_target}"
        f" [{waveform.capture_index}]"
    )


def _add_monitor_waveform_trace(fig: Any, waveform: _MonitorWaveform) -> None:
    fig.add_scatter(
        x=waveform.time_ns,
        y=waveform.amplitude,
        mode="lines",
        name=_waveform_trace_label(waveform),
        hovertemplate=(
            "time=%{x:.3f}ns<br>|IQ|=%{y:.6g}<extra>%{fullData.name}</extra>"
        ),
    )


def _apply_monitor_waveform_layout(fig: Any) -> None:
    fig.update_layout(
        title="Latest raw monitor waveform",
        template="qubex",
        width=_LIVE_WAVEFORM_PLOT_WIDTH,
        height=_LIVE_WAVEFORM_PLOT_HEIGHT,
        font={"family": FONT_FAMILY},
        xaxis_title="time (ns)",
        yaxis_title="|IQ|",
    )


def _make_monitor_waveform_widget(waveforms: Collection[_MonitorWaveform]) -> Any:
    """Return a live-updated raw waveform widget."""
    import plotly.graph_objects as go

    import qubex.visualization as viz

    widget = go.FigureWidget(viz.make_figure())
    _apply_monitor_waveform_layout(widget)
    _update_monitor_waveform_widget(widget, waveforms)
    return widget


def _update_monitor_waveform_widget(
    widget: Any,
    waveforms: Collection[_MonitorWaveform],
) -> None:
    """Replace the raw waveform widget with the latest monitor capture."""
    waveforms = list(waveforms)
    with widget.batch_update() if hasattr(widget, "batch_update") else nullcontext():
        if [trace.name for trace in widget.data] != [
            _waveform_trace_label(waveform) for waveform in waveforms
        ]:
            widget.data = ()
            for waveform in waveforms:
                _add_monitor_waveform_trace(widget, waveform)
            return
        for trace, waveform in zip(widget.data, waveforms, strict=False):
            trace.x = waveform.time_ns
            trace.y = waveform.amplitude


def _display_widget(widget: Any) -> bool:
    """Display a widget in a notebook output cell if IPython is available."""
    try:
        from IPython.display import display
    except ImportError:
        return False
    display(widget)
    return True


def _show_signal_stability_figure(
    snapshots: Collection[MeasurementStabilitySnapshot],
) -> None:
    """Show the stability plot when live notebook display is unavailable."""
    fig = _make_signal_stability_figure(snapshots)
    fig.show()


def _limited_output_gain(
    *,
    previous_gain: float,
    raw_gain: float,
    max_gain_relative_step: float,
    effective_gain_correction_deadband: float,
) -> float:
    """Return a bounded gain update, ignoring changes inside the deadband."""
    relative_step = raw_gain / previous_gain
    if abs(relative_step - 1.0) <= effective_gain_correction_deadband:
        return previous_gain
    return previous_gain * float(
        np.clip(
            relative_step,
            1.0 - max_gain_relative_step,
            1.0 + max_gain_relative_step,
        )
    )


def _wrapped_phase_delta_rad(current: float, previous: float) -> float:
    """Return the wrapped phase difference between two phase measurements."""
    return float(np.angle(np.exp(1j * (current - previous))))


def _limited_output_phase_offset(
    *,
    previous_offset: float,
    raw_offset: float,
    max_step: float,
    correction_deadband: float,
) -> float:
    """Return a bounded phase-offset update in radians."""
    step = raw_offset - previous_offset
    if abs(step) <= correction_deadband:
        return previous_offset
    return previous_offset + float(np.clip(step, -max_step, max_step))


def _relative_sem(mean: float, sem: float) -> float:
    """Return relative standard error, or NaN when it is not well-defined."""
    if mean == 0.0 or not np.isfinite(mean) or not np.isfinite(sem):
        return np.nan
    return abs(sem / mean)


def _effective_gain_correction_deadband(
    *,
    base_deadband: float,
    sigma: float,
    previous: OutputSignalCorrection,
    measured: MonitorStatistic,
    auto: bool,
) -> float:
    """Return the gain deadband after optional measurement-noise expansion."""
    if not auto:
        return base_deadband

    reference_relative_sem = _relative_sem(
        previous.reference_amplitude,
        previous.reference_amplitude_sem,
    )
    measured_relative_sem = _relative_sem(
        measured.amplitude_mean,
        measured.amplitude_sem,
    )
    uncertainty = float(
        np.hypot(
            0.0 if not np.isfinite(reference_relative_sem) else reference_relative_sem,
            0.0 if not np.isfinite(measured_relative_sem) else measured_relative_sem,
        )
    )
    return max(base_deadband, sigma * uncertainty)


def _call_loopback_capture(
    capture: Callable[..., MeasurementResult],
    schedule: PulseSchedule,
    *,
    n_shots: int | None,
    block_outputs: bool,
    shot_averaging: bool,
    capture_targets: list[str],
    configure_monitor_nco: bool | None = None,
) -> MeasurementResult:
    """Call loopback capture, passing monitor-NCO control when supported."""
    kwargs: dict[str, Any] = {
        "n_shots": n_shots,
        "block_outputs": block_outputs,
        "shot_averaging": shot_averaging,
        "capture_targets": capture_targets,
    }
    if configure_monitor_nco is not None and _accepts_keyword(
        capture,
        "configure_monitor_nco",
    ):
        kwargs["configure_monitor_nco"] = configure_monitor_nco
    return capture(schedule, **kwargs)


def _accepts_keyword(callable_: Callable[..., object], name: str) -> bool:
    """Return whether a callable accepts a keyword argument."""
    try:
        parameters = signature(callable_).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _effective_phase_correction_deadband(
    *,
    base_deadband: float,
    sigma: float,
    previous: OutputSignalCorrection,
    measured: MonitorStatistic,
    auto: bool,
) -> float:
    """Return the phase deadband after optional measurement-noise expansion."""
    if not auto:
        return base_deadband

    reference_sem = previous.reference_phase_sem_rad
    measured_sem = measured.phase_sem_rad
    uncertainty = float(
        np.hypot(
            0.0 if not np.isfinite(reference_sem) else reference_sem,
            0.0 if not np.isfinite(measured_sem) else measured_sem,
        )
    )
    return max(base_deadband, sigma * uncertainty)


class MeasurementStabilityService:
    """
    Manage session-local measurement stability baselines and corrections.

    Parameters
    ----------
    context : MeasurementContext
        Shared measurement context used to resolve targets and apply
        session-local output corrections.
    monitor_service : MeasurementMonitorService | None, optional
        Monitor service used when public methods are called without an
        explicit ``capture`` callable.
    """

    def __init__(
        self,
        *,
        context: MeasurementContext,
        monitor_service: MeasurementMonitorService | None = None,
    ) -> None:
        self._context = context
        self._monitor_service = monitor_service
        self._output_corrections: dict[str, OutputSignalCorrection] = {}
        self._corrections_suspended = 0

    @property
    def context(self) -> MeasurementContext:
        """Return the measurement context."""
        return self._context

    @property
    def has_output_signal_baseline(self) -> bool:
        """Return whether output-signal baseline data is available."""
        return len(self._output_corrections) > 0

    @property
    def corrections_enabled(self) -> bool:
        """Return whether corrections should be applied to outgoing schedules."""
        return self._corrections_suspended == 0

    def _resolve_loopback_capture(
        self,
        capture: Callable[..., MeasurementResult] | None,
    ) -> Callable[..., MeasurementResult]:
        """Return the explicit capture callable or the injected monitor service."""
        if capture is not None:
            return capture
        if self._monitor_service is None:
            raise ValueError(
                "capture must be provided when no monitor_service is configured."
            )
        return self._monitor_service.capture_loopback

    @contextmanager
    def suspend_corrections(self) -> Iterator[None]:
        """Temporarily disable correction application for diagnostic captures."""
        self._corrections_suspended += 1
        try:
            yield
        finally:
            self._corrections_suspended -= 1

    def snapshot(
        self,
        *,
        output_corrections: dict[str, OutputSignalCorrection] | None = None,
        signals: dict[str, MonitorStatistic] | None = None,
        sample_index: int | None = None,
        elapsed_s: float | None = None,
        timestamp: str | None = None,
    ) -> MeasurementStabilitySnapshot:
        """Return a snapshot of current stability state."""
        return MeasurementStabilitySnapshot(
            output_corrections=dict(
                self._output_corrections
                if output_corrections is None
                else output_corrections
            ),
            signals={} if signals is None else dict(signals),
            sample_index=sample_index,
            elapsed_s=elapsed_s,
            timestamp=timestamp,
        )

    def establish_output_signal_baseline(
        self,
        *,
        capture: Callable[..., MeasurementResult] | None = None,
        targets: Collection[str] | str | None = None,
        include_control: bool = True,
        include_readout: bool = True,
        n_shots: int | None = DEFAULT_STABILITY_CORRECTION_N_SHOTS,
        probe_amplitude: float = 0.1,
        probe_duration: float = DEFAULT_STABILITY_CORRECTION_PROBE_DURATION,
        block_outputs: bool = True,
        reference_scope: OutputSignalReferenceScope = "box",
        trim_samples: int = 0,
        estimate_gain_noise: bool = True,
        estimate_phase_noise: bool = True,
    ) -> MeasurementStabilitySnapshot:
        """
        Capture baseline monitor signals and reset session-local corrections.

        Parameters
        ----------
        capture : Callable[..., MeasurementResult] | None, optional
            Capture callable used for monitor-path loopback acquisition. When
            omitted, the injected monitor service is used.
        targets : Collection[str] | str | None, optional
            Output target labels to probe. When omitted, targets are selected
            from the enabled control/readout groups.
        include_control : bool, optional
            Whether control-output targets are eligible when ``targets`` is
            omitted.
        include_readout : bool, optional
            Whether readout-output targets are eligible when ``targets`` is
            omitted.
        n_shots : int | None, optional
            Number of shots used for each monitor capture.
        probe_amplitude : float, optional
            Amplitude of the flat-top probe waveform.
        probe_duration : float, optional
            Duration of the probe plateau in ns.
        block_outputs : bool, optional
            Whether active output ports are RF-blocked during monitor capture.
        reference_scope : {"box", "target"}, optional
            Scope used to share one baseline across selected targets.
        trim_samples : int, optional
            Number of samples trimmed from both waveform edges before
            statistics are computed.
        estimate_gain_noise : bool, optional
            Whether to retain shot-level gain noise in the baseline capture.
        estimate_phase_noise : bool, optional
            Whether to retain shot-level phase noise in the baseline capture.

        Returns
        -------
        MeasurementStabilitySnapshot
            Snapshot containing the newly captured baseline statistics.

        Notes
        -----
        When ``reference_scope="box"``, one representative target is measured
        per physical box and stored as the reference for all selected targets
        on that box.
        """
        capture = self._resolve_loopback_capture(capture)
        if trim_samples < 0:
            raise ValueError("trim_samples must be non-negative.")
        resolved_reference_scope = self._validate_reference_scope(reference_scope)
        selected_targets = self._resolve_output_targets(
            targets=targets,
            include_control=include_control,
            include_readout=include_readout,
        )
        statistics = self.measure_monitor_statistics(
            capture=capture,
            targets=targets,
            include_control=include_control,
            include_readout=include_readout,
            n_shots=n_shots,
            probe_amplitude=probe_amplitude,
            probe_duration=probe_duration,
            block_outputs=block_outputs,
            reference_scope=resolved_reference_scope,
            trim_samples=trim_samples,
            apply_corrections=False,
            shot_averaging=not (estimate_gain_noise or estimate_phase_noise),
            configure_monitor_nco=True,
        )
        self._set_output_signal_baseline_from_statistics(
            statistics=statistics,
            selected_targets=selected_targets,
            reference_scope=resolved_reference_scope,
        )
        return self.snapshot(signals=self._monitor_statistics_by_key(statistics))

    def update_output_signal_corrections(
        self,
        *,
        capture: Callable[..., MeasurementResult] | None = None,
        targets: Collection[str] | str | None = None,
        include_control: bool = True,
        include_readout: bool = True,
        n_shots: int | None = DEFAULT_STABILITY_CORRECTION_N_SHOTS,
        probe_amplitude: float = 0.1,
        probe_duration: float = DEFAULT_STABILITY_CORRECTION_PROBE_DURATION,
        block_outputs: bool = True,
        max_gain_relative_step: float = DEFAULT_OUTPUT_GAIN_CORRECTION_MAX_RELATIVE_STEP,
        gain_smoothing: float = DEFAULT_OUTPUT_GAIN_CORRECTION_SMOOTHING,
        gain_correction_deadband: float = DEFAULT_OUTPUT_GAIN_CORRECTION_DEADBAND,
        auto_gain_correction_deadband: bool = True,
        gain_correction_deadband_sigma: float = DEFAULT_OUTPUT_GAIN_CORRECTION_DEADBAND_SIGMA,
        max_phase_step: float = DEFAULT_OUTPUT_PHASE_CORRECTION_MAX_STEP,
        phase_smoothing: float = DEFAULT_OUTPUT_PHASE_CORRECTION_SMOOTHING,
        phase_correction_deadband: float = DEFAULT_OUTPUT_PHASE_CORRECTION_DEADBAND,
        auto_phase_correction_deadband: bool = True,
        phase_correction_deadband_sigma: float = DEFAULT_OUTPUT_PHASE_CORRECTION_DEADBAND_SIGMA,
        phase_min_resultant_length: float = DEFAULT_OUTPUT_PHASE_MIN_RESULTANT_LENGTH,
        reference_scope: OutputSignalReferenceScope | None = None,
        trim_samples: int = 0,
    ) -> MeasurementStabilitySnapshot:
        """
        Update session-local output gain and phase corrections.

        Parameters
        ----------
        capture : Callable[..., MeasurementResult] | None, optional
            Capture callable used for monitor-path loopback acquisition. When
            omitted, the injected monitor service is used.
        targets : Collection[str] | str | None, optional
            Output target labels to probe.
        include_control : bool, optional
            Whether control-output targets are eligible when ``targets`` is
            omitted.
        include_readout : bool, optional
            Whether readout-output targets are eligible when ``targets`` is
            omitted.
        n_shots : int | None, optional
            Number of shots used for each monitor capture.
        probe_amplitude : float, optional
            Amplitude of the flat-top probe waveform.
        probe_duration : float, optional
            Duration of the probe plateau in ns.
        block_outputs : bool, optional
            Whether active output ports are RF-blocked during monitor capture.
        max_gain_relative_step : float, optional
            Maximum gain-correction change applied in one update.
        gain_smoothing : float, optional
            First-order smoothing factor for gain corrections.
        gain_correction_deadband : float, optional
            Minimum relative gain residual required before changing gain.
        auto_gain_correction_deadband : bool, optional
            Whether gain deadband expands with measured SEM.
        gain_correction_deadband_sigma : float, optional
            SEM multiplier used for automatic gain deadband expansion.
        max_phase_step : float, optional
            Maximum phase-correction change in radians applied in one update.
        phase_smoothing : float, optional
            First-order smoothing factor for phase corrections.
        phase_correction_deadband : float, optional
            Minimum phase residual in radians required before changing phase.
        auto_phase_correction_deadband : bool, optional
            Whether phase deadband expands with measured SEM.
        phase_correction_deadband_sigma : float, optional
            SEM multiplier used for automatic phase deadband expansion.
        phase_min_resultant_length : float, optional
            Minimum circular mean quality required for phase updates.
        reference_scope : OutputSignalReferenceScope | None, optional
            Baseline sharing scope. When omitted after baseline exists, the
            stored baseline scope is reused.
        trim_samples : int, optional
            Number of samples trimmed from both waveform edges before
            statistics are computed.

        Returns
        -------
        MeasurementStabilitySnapshot
            Snapshot after baseline capture or correction update.

        Notes
        -----
        If no baseline exists, this method captures one and returns its
        snapshot without applying an additional correction update.
        """
        capture = self._resolve_loopback_capture(capture)
        if max_gain_relative_step < 0:
            raise ValueError("max_gain_relative_step must be non-negative.")
        if not 0.0 <= gain_smoothing <= 1.0:
            raise ValueError("gain_smoothing must be between 0 and 1.")
        if gain_correction_deadband < 0:
            raise ValueError("gain_correction_deadband must be non-negative.")
        if gain_correction_deadband_sigma < 0:
            raise ValueError("gain_correction_deadband_sigma must be non-negative.")
        if max_phase_step < 0:
            raise ValueError("max_phase_step must be non-negative.")
        if not 0.0 <= phase_smoothing <= 1.0:
            raise ValueError("phase_smoothing must be between 0 and 1.")
        if phase_correction_deadband < 0:
            raise ValueError("phase_correction_deadband must be non-negative.")
        if phase_correction_deadband_sigma < 0:
            raise ValueError("phase_correction_deadband_sigma must be non-negative.")
        if not 0.0 <= phase_min_resultant_length <= 1.0:
            raise ValueError("phase_min_resultant_length must be between 0 and 1.")
        if trim_samples < 0:
            raise ValueError("trim_samples must be non-negative.")
        if not self.has_output_signal_baseline:
            return self.establish_output_signal_baseline(
                capture=capture,
                targets=targets,
                include_control=include_control,
                include_readout=include_readout,
                n_shots=n_shots,
                probe_amplitude=probe_amplitude,
                probe_duration=probe_duration,
                block_outputs=block_outputs,
                reference_scope="box" if reference_scope is None else reference_scope,
                trim_samples=trim_samples,
                estimate_gain_noise=auto_gain_correction_deadband,
                estimate_phase_noise=auto_phase_correction_deadband,
            )

        resolved_reference_scope = self._resolve_update_reference_scope(reference_scope)
        selected_targets = self._resolve_output_targets(
            targets=targets,
            include_control=include_control,
            include_readout=include_readout,
        )
        updated = dict(self._output_corrections)
        with self.suspend_corrections():
            for reference_target, group_targets in self._group_update_targets(
                selected_targets,
                reference_scope=resolved_reference_scope,
            ):
                records_to_update = [
                    (target, self._output_corrections[target.label])
                    for target in group_targets
                    if target.label in self._output_corrections
                ]
                if not records_to_update:
                    continue
                measured = self._measure_monitor_statistic(
                    capture=capture,
                    target=reference_target,
                    n_shots=n_shots,
                    probe_amplitude=probe_amplitude,
                    probe_duration=probe_duration,
                    block_outputs=block_outputs,
                    trim_samples=trim_samples,
                    shot_averaging=not (
                        auto_gain_correction_deadband or auto_phase_correction_deadband
                    ),
                    configure_monitor_nco=False,
                )
                if measured.amplitude_mean <= 0:
                    raise ValueError("measured monitor amplitudes must be positive.")
                monitor_target = self._resolve_monitor_target(reference_target)
                for target, previous in records_to_update:
                    updated[target.label] = self._updated_output_correction(
                        target_label=target.label,
                        previous=previous,
                        monitor_target=monitor_target,
                        measured=measured,
                        reference_target=reference_target.label,
                        reference_scope=resolved_reference_scope,
                        max_gain_relative_step=max_gain_relative_step,
                        gain_smoothing=gain_smoothing,
                        gain_correction_deadband=gain_correction_deadband,
                        auto_gain_correction_deadband=auto_gain_correction_deadband,
                        gain_correction_deadband_sigma=gain_correction_deadband_sigma,
                        max_phase_step=max_phase_step,
                        phase_smoothing=phase_smoothing,
                        phase_correction_deadband=phase_correction_deadband,
                        auto_phase_correction_deadband=(auto_phase_correction_deadband),
                        phase_correction_deadband_sigma=(
                            phase_correction_deadband_sigma
                        ),
                        phase_min_resultant_length=phase_min_resultant_length,
                    )
        self._output_corrections = updated
        return self.snapshot()

    def _set_output_signal_baseline_from_statistics(
        self,
        *,
        statistics: Collection[MonitorStatistic],
        selected_targets: Collection[Target],
        reference_scope: OutputSignalReferenceScope,
    ) -> None:
        """Reset output-signal baseline records from measured monitor statistics."""
        targets_by_label = {target.label: target for target in selected_targets}
        records: dict[str, OutputSignalCorrection] = {}
        for measured in statistics:
            if measured.amplitude_mean <= 0:
                raise ValueError("baseline monitor amplitudes must be positive.")
            if not np.isfinite(measured.phase_mean_rad):
                raise ValueError("baseline monitor phases must be finite.")
            for target_label in measured.covered_targets:
                target = targets_by_label[target_label]
                records[target.label] = OutputSignalCorrection(
                    target=target.label,
                    kind=self._resolve_output_kind(target),
                    port_id=str(target.channel.port.id),
                    monitor_target=measured.monitor_target,
                    reference_amplitude=measured.amplitude_mean,
                    reference_amplitude_sem=measured.amplitude_sem,
                    measured_amplitude=measured.amplitude_mean,
                    measured_amplitude_sem=measured.amplitude_sem,
                    reference_phase_rad=measured.phase_mean_rad,
                    reference_phase_sem_rad=measured.phase_sem_rad,
                    measured_phase_rad=measured.phase_mean_rad,
                    measured_phase_sem_rad=measured.phase_sem_rad,
                    gain=1.0,
                    previous_gain=1.0,
                    raw_gain=1.0,
                    limited_gain=1.0,
                    effective_gain_correction_deadband=0.0,
                    phase_offset_rad=0.0,
                    previous_phase_offset_rad=0.0,
                    raw_phase_offset_rad=0.0,
                    limited_phase_offset_rad=0.0,
                    effective_phase_correction_deadband=0.0,
                    reference_target=measured.reference_target,
                    reference_scope=reference_scope,
                )
        self._output_corrections = records

    def _update_output_signal_corrections_from_statistics(
        self,
        *,
        statistics: Collection[MonitorStatistic],
        applied_corrections: dict[str, OutputSignalCorrection] | None = None,
        targets: Collection[str] | str | None = None,
        include_control: bool = True,
        include_readout: bool = True,
        max_gain_relative_step: float = DEFAULT_OUTPUT_GAIN_CORRECTION_MAX_RELATIVE_STEP,
        gain_smoothing: float = DEFAULT_OUTPUT_GAIN_CORRECTION_SMOOTHING,
        gain_correction_deadband: float = DEFAULT_OUTPUT_GAIN_CORRECTION_DEADBAND,
        auto_gain_correction_deadband: bool = True,
        gain_correction_deadband_sigma: float = DEFAULT_OUTPUT_GAIN_CORRECTION_DEADBAND_SIGMA,
        max_phase_step: float = DEFAULT_OUTPUT_PHASE_CORRECTION_MAX_STEP,
        phase_smoothing: float = DEFAULT_OUTPUT_PHASE_CORRECTION_SMOOTHING,
        phase_correction_deadband: float = DEFAULT_OUTPUT_PHASE_CORRECTION_DEADBAND,
        auto_phase_correction_deadband: bool = True,
        phase_correction_deadband_sigma: float = DEFAULT_OUTPUT_PHASE_CORRECTION_DEADBAND_SIGMA,
        phase_min_resultant_length: float = DEFAULT_OUTPUT_PHASE_MIN_RESULTANT_LENGTH,
        reference_scope: OutputSignalReferenceScope | None = None,
    ) -> MeasurementStabilitySnapshot:
        """Update output signal corrections from measured monitor statistics."""
        if max_gain_relative_step < 0:
            raise ValueError("max_gain_relative_step must be non-negative.")
        if not 0.0 <= gain_smoothing <= 1.0:
            raise ValueError("gain_smoothing must be between 0 and 1.")
        if gain_correction_deadband < 0:
            raise ValueError("gain_correction_deadband must be non-negative.")
        if gain_correction_deadband_sigma < 0:
            raise ValueError("gain_correction_deadband_sigma must be non-negative.")
        if max_phase_step < 0:
            raise ValueError("max_phase_step must be non-negative.")
        if not 0.0 <= phase_smoothing <= 1.0:
            raise ValueError("phase_smoothing must be between 0 and 1.")
        if phase_correction_deadband < 0:
            raise ValueError("phase_correction_deadband must be non-negative.")
        if phase_correction_deadband_sigma < 0:
            raise ValueError("phase_correction_deadband_sigma must be non-negative.")
        if not 0.0 <= phase_min_resultant_length <= 1.0:
            raise ValueError("phase_min_resultant_length must be between 0 and 1.")
        if not self.has_output_signal_baseline:
            raise ValueError(
                "No output-signal baseline is available. "
                "Call establish_output_signal_baseline() first."
            )

        measured_by_reference: dict[str, MonitorStatistic] = {}
        for statistic in statistics:
            measured_by_reference.setdefault(statistic.reference_target, statistic)

        resolved_reference_scope = self._resolve_update_reference_scope(reference_scope)
        selected_targets = self._resolve_output_targets(
            targets=targets,
            include_control=include_control,
            include_readout=include_readout,
        )
        updated = dict(self._output_corrections)

        for reference_target, group_targets in self._group_update_targets(
            selected_targets,
            reference_scope=resolved_reference_scope,
        ):
            records_to_update = [
                (target, self._output_corrections[target.label])
                for target in group_targets
                if target.label in self._output_corrections
            ]
            if not records_to_update:
                continue
            measured = measured_by_reference.get(reference_target.label)
            if measured is None:
                raise ValueError(
                    f"No monitor statistic found for {reference_target.label}."
                )
            if measured.amplitude_mean <= 0:
                raise ValueError("measured monitor amplitudes must be positive.")
            monitor_target = self._resolve_monitor_target(reference_target)
            for target, previous in records_to_update:
                applied = (
                    None
                    if applied_corrections is None
                    else applied_corrections.get(target.label)
                )
                updated[target.label] = self._updated_output_correction(
                    target_label=target.label,
                    previous=previous,
                    applied_gain=1.0 if applied is None else applied.gain,
                    applied_phase_offset_rad=(
                        0.0 if applied is None else applied.phase_offset_rad
                    ),
                    monitor_target=monitor_target,
                    measured=measured,
                    reference_target=reference_target.label,
                    reference_scope=resolved_reference_scope,
                    max_gain_relative_step=max_gain_relative_step,
                    gain_smoothing=gain_smoothing,
                    gain_correction_deadband=gain_correction_deadband,
                    auto_gain_correction_deadband=auto_gain_correction_deadband,
                    gain_correction_deadband_sigma=gain_correction_deadband_sigma,
                    max_phase_step=max_phase_step,
                    phase_smoothing=phase_smoothing,
                    phase_correction_deadband=phase_correction_deadband,
                    auto_phase_correction_deadband=auto_phase_correction_deadband,
                    phase_correction_deadband_sigma=phase_correction_deadband_sigma,
                    phase_min_resultant_length=phase_min_resultant_length,
                )

        self._output_corrections = updated
        return self.snapshot()

    def measure_monitor_statistics(
        self,
        *,
        capture: Callable[..., MeasurementResult] | None = None,
        targets: Collection[str] | str | None = None,
        include_control: bool = True,
        include_readout: bool = True,
        n_shots: int | None = DEFAULT_STABILITY_CORRECTION_N_SHOTS,
        probe_amplitude: float = 0.1,
        probe_duration: float = DEFAULT_STABILITY_CORRECTION_PROBE_DURATION,
        block_outputs: bool = True,
        reference_scope: OutputSignalReferenceScope = "box",
        trim_samples: int = 0,
        apply_corrections: bool = False,
        shot_averaging: bool = True,
        configure_monitor_nco: bool | None = None,
        _waveforms_out: list[_MonitorWaveform] | None = None,
    ) -> list[MonitorStatistic]:
        """
        Probe selected outputs and return monitor amplitude/phase statistics.

        Parameters
        ----------
        capture : Callable[..., MeasurementResult] | None, optional
            Capture callable used for monitor-path loopback acquisition. When
            omitted, the injected monitor service is used.
        targets : Collection[str] | str | None, optional
            Output target labels to probe.
        include_control : bool, optional
            Whether control-output targets are eligible when ``targets`` is
            omitted.
        include_readout : bool, optional
            Whether readout-output targets are eligible when ``targets`` is
            omitted.
        n_shots : int | None, optional
            Number of shots used for each monitor capture.
        probe_amplitude : float, optional
            Amplitude of the flat-top probe waveform.
        probe_duration : float, optional
            Duration of the probe plateau in ns.
        block_outputs : bool, optional
            Whether active output ports are RF-blocked during monitor capture.
        reference_scope : {"box", "target"}, optional
            Scope used to choose representative probe targets.
        trim_samples : int, optional
            Number of samples trimmed from both waveform edges before
            statistics are computed.
        apply_corrections : bool, optional
            Whether session-local output corrections are applied to the probe
            schedule.
        shot_averaging : bool, optional
            Whether monitor captures are averaged across shots.
        configure_monitor_nco : bool | None, optional
            Forwarded to monitor capture to control receiver NCO configuration.
        _waveforms_out : list[_MonitorWaveform] | None, optional
            Internal sink used by live plotting to receive the latest waveform.

        Returns
        -------
        list[MonitorStatistic]
            Amplitude and phase statistics for each monitor capture.

        Notes
        -----
        The default box-level scope measures one representative target per
        physical box, matching the baseline workflow.
        """
        capture = self._resolve_loopback_capture(capture)
        resolved_reference_scope = self._validate_reference_scope(reference_scope)
        selected_targets = self._resolve_output_targets(
            targets=targets,
            include_control=include_control,
            include_readout=include_readout,
        )
        if targets is not None:
            self._warn_if_explicit_monitor_targets_replace_baseline(
                selected_targets,
                reference_scope=resolved_reference_scope,
            )
        statistics: list[MonitorStatistic] = []
        correction_context = (
            nullcontext() if apply_corrections else self.suspend_corrections()
        )
        with correction_context:
            for reference_target, group_targets in self._group_reference_targets(
                selected_targets,
                reference_scope=resolved_reference_scope,
            ):
                covered_targets = tuple(target.label for target in group_targets)
                monitor_target = self._resolve_monitor_target(reference_target)
                schedule = self._build_probe_schedule(
                    target=reference_target,
                    probe_amplitude=probe_amplitude,
                    probe_duration=probe_duration,
                )
                result = _call_loopback_capture(
                    capture,
                    schedule,
                    n_shots=n_shots,
                    block_outputs=block_outputs,
                    shot_averaging=shot_averaging,
                    capture_targets=[monitor_target],
                    configure_monitor_nco=configure_monitor_nco,
                )
                capture_statistics = self._compute_probe_monitor_statistics(
                    result,
                    reference_target=reference_target.label,
                    monitor_target=monitor_target,
                    trim_samples=trim_samples,
                )
                if _waveforms_out is not None:
                    _waveforms_out.extend(
                        self._extract_probe_monitor_waveforms(
                            result,
                            reference_target=reference_target.label,
                            monitor_target=monitor_target,
                            trim_samples=trim_samples,
                        )
                    )
                if len(capture_statistics) == 0:
                    raise ValueError(f"No monitor capture found for {monitor_target}.")
                statistics.extend(
                    MonitorStatistic(
                        reference_target=reference_target.label,
                        covered_targets=covered_targets,
                        monitor_target=monitor_target,
                        capture_index=statistic.capture_index,
                        amplitude_mean=statistic.amplitude_mean,
                        amplitude_std=statistic.amplitude_std,
                        amplitude_sem=statistic.amplitude_sem,
                        amplitude_relative_sem=statistic.amplitude_relative_sem,
                        phase_mean_rad=statistic.phase_mean_rad,
                        phase_std_rad=statistic.phase_std_rad,
                        phase_sem_rad=statistic.phase_sem_rad,
                        phase_resultant_length=statistic.phase_resultant_length,
                        n_shots=statistic.n_shots,
                        n_samples=statistic.n_samples,
                    )
                    for statistic in capture_statistics
                )
        return statistics

    def check_signal_stability(
        self,
        *,
        capture: Callable[..., MeasurementResult] | None = None,
        duration: float,
        sample_interval: float | None = 10.0,
        targets: Collection[str] | str | None = None,
        include_control: bool = True,
        include_readout: bool = True,
        n_shots: int | None = DEFAULT_STABILITY_CORRECTION_N_SHOTS,
        probe_amplitude: float = 0.1,
        probe_duration: float = DEFAULT_STABILITY_CORRECTION_PROBE_DURATION,
        block_outputs: bool = True,
        reference_scope: OutputSignalReferenceScope = "box",
        trim_samples: int = 0,
        max_gain_relative_step: float = DEFAULT_OUTPUT_GAIN_CORRECTION_MAX_RELATIVE_STEP,
        gain_smoothing: float = DEFAULT_OUTPUT_GAIN_CORRECTION_SMOOTHING,
        gain_correction_deadband: float = DEFAULT_OUTPUT_GAIN_CORRECTION_DEADBAND,
        auto_gain_correction_deadband: bool = True,
        gain_correction_deadband_sigma: float = DEFAULT_OUTPUT_GAIN_CORRECTION_DEADBAND_SIGMA,
        max_phase_step: float = DEFAULT_OUTPUT_PHASE_CORRECTION_MAX_STEP,
        phase_smoothing: float = DEFAULT_OUTPUT_PHASE_CORRECTION_SMOOTHING,
        phase_correction_deadband: float = DEFAULT_OUTPUT_PHASE_CORRECTION_DEADBAND,
        auto_phase_correction_deadband: bool = True,
        phase_correction_deadband_sigma: float = DEFAULT_OUTPUT_PHASE_CORRECTION_DEADBAND_SIGMA,
        phase_min_resultant_length: float = DEFAULT_OUTPUT_PHASE_MIN_RESULTANT_LENGTH,
        update_corrections: bool = True,
        plot: bool = False,
    ) -> list[MeasurementStabilitySnapshot]:
        """
        Repeatedly sample monitor statistics and optionally update corrections.

        Parameters
        ----------
        capture : Callable[..., MeasurementResult] | None, optional
            Capture callable used for monitor-path loopback acquisition. When
            omitted, the injected monitor service is used.
        duration : float
            Total monitoring duration in seconds.
        sample_interval : float | None, optional
            Delay between completed samples in seconds. If ``None``, the next
            sample starts immediately.
        targets : Collection[str] | str | None, optional
            Output target labels to probe.
        include_control : bool, optional
            Whether control-output targets are eligible when ``targets`` is
            omitted.
        include_readout : bool, optional
            Whether readout-output targets are eligible when ``targets`` is
            omitted.
        n_shots : int | None, optional
            Number of shots used for each monitor capture.
        probe_amplitude : float, optional
            Amplitude of the flat-top probe waveform.
        probe_duration : float, optional
            Duration of the probe plateau in ns.
        block_outputs : bool, optional
            Whether active output ports are RF-blocked during monitor capture.
        reference_scope : {"box", "target"}, optional
            Scope used to choose representative probe targets.
        trim_samples : int, optional
            Number of samples trimmed from both waveform edges before
            statistics are computed.
        max_gain_relative_step : float, optional
            Maximum gain-correction change applied in one update.
        gain_smoothing : float, optional
            First-order smoothing factor for gain corrections.
        gain_correction_deadband : float, optional
            Minimum relative gain residual required before changing gain.
        auto_gain_correction_deadband : bool, optional
            Whether gain deadband expands with measured SEM.
        gain_correction_deadband_sigma : float, optional
            SEM multiplier used for automatic gain deadband expansion.
        max_phase_step : float, optional
            Maximum phase-correction change in radians applied in one update.
        phase_smoothing : float, optional
            First-order smoothing factor for phase corrections.
        phase_correction_deadband : float, optional
            Minimum phase residual in radians required before changing phase.
        auto_phase_correction_deadband : bool, optional
            Whether phase deadband expands with measured SEM.
        phase_correction_deadband_sigma : float, optional
            SEM multiplier used for automatic phase deadband expansion.
        phase_min_resultant_length : float, optional
            Minimum circular mean quality required for phase updates.
        update_corrections : bool, optional
            Whether to update session-local gain/phase corrections after each
            sample. If false, samples are passive measurements.
        plot : bool, optional
            When true in a notebook, display two live ``FigureWidget`` objects.
            The first shows relative amplitude normalized to the baseline and
            phase shifted so the first sample is zero. The second shows the
            latest raw monitor ``|IQ|`` waveform. Both widgets are updated in
            place for each sample.

        Returns
        -------
        list[MeasurementStabilitySnapshot]
            Baseline plus sampled stability snapshots.
        """
        capture = self._resolve_loopback_capture(capture)
        if duration < 0.0:
            raise ValueError("duration must be non-negative.")
        if sample_interval is not None and sample_interval <= 0.0:
            raise ValueError("sample_interval must be positive when specified.")
        if max_gain_relative_step < 0:
            raise ValueError("max_gain_relative_step must be non-negative.")
        if not 0.0 <= gain_smoothing <= 1.0:
            raise ValueError("gain_smoothing must be between 0 and 1.")
        if gain_correction_deadband < 0:
            raise ValueError("gain_correction_deadband must be non-negative.")
        if gain_correction_deadband_sigma < 0:
            raise ValueError("gain_correction_deadband_sigma must be non-negative.")
        if max_phase_step < 0:
            raise ValueError("max_phase_step must be non-negative.")
        if not 0.0 <= phase_smoothing <= 1.0:
            raise ValueError("phase_smoothing must be between 0 and 1.")
        if phase_correction_deadband < 0:
            raise ValueError("phase_correction_deadband must be non-negative.")
        if phase_correction_deadband_sigma < 0:
            raise ValueError("phase_correction_deadband_sigma must be non-negative.")
        if not 0.0 <= phase_min_resultant_length <= 1.0:
            raise ValueError("phase_min_resultant_length must be between 0 and 1.")

        resolved_reference_scope = self._validate_reference_scope(reference_scope)
        start_time = time.perf_counter()
        end_time = start_time + duration
        baseline_timestamp = datetime.now(timezone.utc).isoformat()
        selected_targets = self._resolve_output_targets(
            targets=targets,
            include_control=include_control,
            include_readout=include_readout,
        )
        baseline_waveforms: list[_MonitorWaveform] | None = [] if plot else None
        baseline_statistics = self.measure_monitor_statistics(
            capture=capture,
            targets=targets,
            include_control=include_control,
            include_readout=include_readout,
            n_shots=n_shots,
            probe_amplitude=probe_amplitude,
            probe_duration=probe_duration,
            block_outputs=block_outputs,
            reference_scope=resolved_reference_scope,
            trim_samples=trim_samples,
            apply_corrections=False,
            shot_averaging=not (
                auto_gain_correction_deadband or auto_phase_correction_deadband
            ),
            configure_monitor_nco=True,
            _waveforms_out=baseline_waveforms,
        )
        self._set_output_signal_baseline_from_statistics(
            statistics=baseline_statistics,
            selected_targets=selected_targets,
            reference_scope=resolved_reference_scope,
        )
        snapshots: list[MeasurementStabilitySnapshot] = [
            self.snapshot(
                signals=self._monitor_statistics_by_key(baseline_statistics),
                sample_index=0,
                elapsed_s=0.0,
                timestamp=baseline_timestamp,
            )
        ]
        sample_index = 1
        stability_widget = None
        waveform_widget = None
        widgets_displayed = False
        if plot:
            stability_widget = _make_signal_stability_widget(snapshots)
            waveform_widget = _make_monitor_waveform_widget(baseline_waveforms or [])
            widgets_displayed = _display_widget(stability_widget)
            widgets_displayed = _display_widget(waveform_widget) and widgets_displayed

        if duration == 0.0:
            if plot and not widgets_displayed:
                _show_signal_stability_figure(snapshots)
            return snapshots

        while True:
            if sample_interval is not None:
                next_sample_time = start_time + sample_index * sample_interval
                if next_sample_time > end_time:
                    break
                sleep_s = next_sample_time - time.perf_counter()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            elif time.perf_counter() >= end_time:
                break

            sample_start = time.perf_counter()
            timestamp = datetime.now(timezone.utc).isoformat()
            applied_corrections = dict(self._output_corrections)
            waveforms: list[_MonitorWaveform] | None = [] if plot else None
            statistics = self.measure_monitor_statistics(
                capture=capture,
                targets=targets,
                include_control=include_control,
                include_readout=include_readout,
                n_shots=n_shots,
                probe_amplitude=probe_amplitude,
                probe_duration=probe_duration,
                block_outputs=block_outputs,
                reference_scope=resolved_reference_scope,
                trim_samples=trim_samples,
                apply_corrections=True,
                shot_averaging=not (
                    auto_gain_correction_deadband or auto_phase_correction_deadband
                ),
                configure_monitor_nco=False,
                _waveforms_out=waveforms,
            )
            elapsed_s = sample_start - start_time
            snapshots.append(
                self.snapshot(
                    output_corrections=applied_corrections,
                    signals=self._monitor_statistics_by_key(statistics),
                    sample_index=sample_index,
                    elapsed_s=elapsed_s,
                    timestamp=timestamp,
                )
            )
            if plot:
                if stability_widget is not None:
                    _update_signal_stability_widget(stability_widget, snapshots)
                if waveform_widget is not None:
                    _update_monitor_waveform_widget(waveform_widget, waveforms or [])
            if update_corrections:
                self._update_output_signal_corrections_from_statistics(
                    statistics=statistics,
                    applied_corrections=applied_corrections,
                    targets=targets,
                    include_control=include_control,
                    include_readout=include_readout,
                    max_gain_relative_step=max_gain_relative_step,
                    gain_smoothing=gain_smoothing,
                    gain_correction_deadband=gain_correction_deadband,
                    auto_gain_correction_deadband=auto_gain_correction_deadband,
                    gain_correction_deadband_sigma=gain_correction_deadband_sigma,
                    max_phase_step=max_phase_step,
                    phase_smoothing=phase_smoothing,
                    phase_correction_deadband=phase_correction_deadband,
                    auto_phase_correction_deadband=auto_phase_correction_deadband,
                    phase_correction_deadband_sigma=phase_correction_deadband_sigma,
                    phase_min_resultant_length=phase_min_resultant_length,
                    reference_scope=resolved_reference_scope,
                )
            sample_index += 1

        if plot and not widgets_displayed:
            _show_signal_stability_figure(snapshots)
        return snapshots

    @staticmethod
    def _monitor_statistics_by_key(
        statistics: Collection[MonitorStatistic],
    ) -> dict[str, MonitorStatistic]:
        """Return monitor statistics keyed by stable signal series."""
        return {
            (
                f"{statistic.reference_target}|{statistic.monitor_target}|"
                f"{statistic.capture_index}"
            ): statistic
            for statistic in statistics
        }

    @staticmethod
    def _updated_output_correction(
        *,
        target_label: str,
        previous: OutputSignalCorrection,
        applied_gain: float = 1.0,
        applied_phase_offset_rad: float = 0.0,
        monitor_target: str,
        measured: MonitorStatistic,
        reference_target: str,
        reference_scope: OutputSignalReferenceScope,
        max_gain_relative_step: float,
        gain_smoothing: float,
        gain_correction_deadband: float,
        auto_gain_correction_deadband: bool,
        gain_correction_deadband_sigma: float,
        max_phase_step: float,
        phase_smoothing: float,
        phase_correction_deadband: float,
        auto_phase_correction_deadband: bool,
        phase_correction_deadband_sigma: float,
        phase_min_resultant_length: float,
    ) -> OutputSignalCorrection:
        """Return an updated output signal correction record."""
        measured_raw_gain = (
            applied_gain * previous.reference_amplitude / measured.amplitude_mean
        )
        raw_gain = measured_raw_gain
        effective_gain_deadband = _effective_gain_correction_deadband(
            base_deadband=gain_correction_deadband,
            sigma=gain_correction_deadband_sigma,
            previous=previous,
            measured=measured,
            auto=auto_gain_correction_deadband,
        )
        limited_gain = _limited_output_gain(
            previous_gain=previous.gain,
            raw_gain=raw_gain,
            max_gain_relative_step=max_gain_relative_step,
            effective_gain_correction_deadband=effective_gain_deadband,
        )
        gain = (1.0 - gain_smoothing) * previous.gain + gain_smoothing * limited_gain

        measured_phase = measured.phase_mean_rad
        previous_phase_offset = previous.phase_offset_rad
        effective_phase_deadband = _effective_phase_correction_deadband(
            base_deadband=phase_correction_deadband,
            sigma=phase_correction_deadband_sigma,
            previous=previous,
            measured=measured,
            auto=auto_phase_correction_deadband,
        )
        if (
            np.isfinite(measured_phase)
            and measured.phase_resultant_length >= phase_min_resultant_length
        ):
            phase_residual = _wrapped_phase_delta_rad(
                measured_phase,
                previous.reference_phase_rad,
            )
            measured_raw_phase_offset = applied_phase_offset_rad + phase_residual
            raw_phase_offset = measured_raw_phase_offset
            has_pending_phase_update = (
                abs(raw_phase_offset - previous_phase_offset) > effective_phase_deadband
            )
            if abs(phase_residual) <= effective_phase_deadband and (
                not has_pending_phase_update
            ):
                raw_phase_offset = previous.raw_phase_offset_rad
                limited_phase_offset = previous_phase_offset
                phase_offset = previous_phase_offset
            else:
                limited_phase_offset = _limited_output_phase_offset(
                    previous_offset=previous_phase_offset,
                    raw_offset=raw_phase_offset,
                    max_step=max_phase_step,
                    correction_deadband=effective_phase_deadband,
                )
                phase_offset = (
                    1.0 - phase_smoothing
                ) * previous_phase_offset + phase_smoothing * limited_phase_offset
        else:
            measured_phase = previous.measured_phase_rad
            raw_phase_offset = previous.raw_phase_offset_rad
            limited_phase_offset = previous_phase_offset
            phase_offset = previous_phase_offset

        return OutputSignalCorrection(
            target=target_label,
            kind=previous.kind,
            port_id=previous.port_id,
            monitor_target=monitor_target,
            reference_amplitude=previous.reference_amplitude,
            reference_amplitude_sem=previous.reference_amplitude_sem,
            measured_amplitude=measured.amplitude_mean,
            measured_amplitude_sem=measured.amplitude_sem,
            reference_phase_rad=previous.reference_phase_rad,
            reference_phase_sem_rad=previous.reference_phase_sem_rad,
            measured_phase_rad=float(measured_phase),
            measured_phase_sem_rad=measured.phase_sem_rad,
            gain=float(gain),
            previous_gain=previous.gain,
            raw_gain=float(raw_gain),
            limited_gain=float(limited_gain),
            effective_gain_correction_deadband=float(effective_gain_deadband),
            phase_offset_rad=float(phase_offset),
            previous_phase_offset_rad=previous_phase_offset,
            raw_phase_offset_rad=float(raw_phase_offset),
            limited_phase_offset_rad=float(limited_phase_offset),
            effective_phase_correction_deadband=float(effective_phase_deadband),
            reference_target=reference_target,
            reference_scope=reference_scope,
        )

    def _compute_probe_monitor_statistics(
        self,
        result: MeasurementResult | MultipleMeasureResult,
        *,
        reference_target: str,
        monitor_target: str,
        trim_samples: int = 0,
    ) -> list[MonitorStatistic]:
        """Compute statistics for a monitor probe result."""
        target_candidates = list(dict.fromkeys((monitor_target, reference_target)))
        return self.compute_monitor_statistics(
            result,
            targets=target_candidates,
            trim_samples=trim_samples,
        )

    def _extract_probe_monitor_waveforms(
        self,
        result: MeasurementResult | MultipleMeasureResult,
        *,
        reference_target: str,
        monitor_target: str,
        trim_samples: int,
    ) -> list[_MonitorWaveform]:
        """Extract one ``|IQ|`` time trace per monitor capture for live display."""
        target_candidates = list(dict.fromkeys((monitor_target, reference_target)))
        waveforms: list[_MonitorWaveform] = []
        for target in target_candidates:
            captures = result.data.get(target)
            if captures is None:
                continue
            for capture_index, capture in enumerate(captures):
                array = np.asarray(
                    _trim_capture(_capture_array(capture), trim_samples),
                    dtype=np.complex128,
                )
                if array.size == 0:
                    continue
                if array.ndim >= 2:
                    samples = array.reshape(-1, array.shape[-1])
                    amplitude = np.mean(np.abs(samples), axis=0)
                else:
                    amplitude = np.abs(array.reshape(-1))
                time_ns = (
                    np.arange(amplitude.size, dtype=np.float64) + trim_samples
                ) * float(capture.sampling_period)
                waveforms.append(
                    _MonitorWaveform(
                        reference_target=reference_target,
                        monitor_target=target,
                        capture_index=capture_index,
                        time_ns=time_ns,
                        amplitude=np.asarray(amplitude, dtype=np.float64),
                    )
                )
        return waveforms

    def compute_monitor_statistics(
        self,
        result: MeasurementResult | MultipleMeasureResult,
        *,
        targets: Collection[str] | None = None,
        trim_samples: int = 0,
    ) -> list[MonitorStatistic]:
        """
        Compute amplitude and phase statistics for monitor captures.

        Parameters
        ----------
        result
            Measurement result returned by monitor capture.
        targets
            Monitor target names to include. If omitted, all result targets are
            included.
        trim_samples
            Number of edge samples to remove from each capture before computing
            statistics.

        Returns
        -------
        list[MonitorStatistic]
            Statistics for each target and capture index.
        """
        target_filter = None if targets is None else set(targets)
        statistics: list[MonitorStatistic] = []
        for target, captures in result.data.items():
            if target_filter is not None and target not in target_filter:
                continue
            for capture_index, capture in enumerate(captures):
                array = _trim_capture(_capture_array(capture), trim_samples)
                complex_array = np.asarray(array, dtype=np.complex128)
                flat = complex_array.reshape(-1)
                if flat.size == 0:
                    raise ValueError(f"Monitor capture for {target} is empty.")
                if complex_array.ndim >= 2:
                    shot_arrays = complex_array.reshape(complex_array.shape[0], -1)
                    shot_amplitudes = np.mean(
                        np.abs(shot_arrays),
                        axis=1,
                    )
                    shot_phases = np.array(
                        [_phase_statistics(shot)[0] for shot in shot_arrays],
                        dtype=np.float64,
                    )
                else:
                    shot_amplitudes = np.abs(flat)
                    amplitudes = np.abs(flat)
                    shot_phases = np.angle(flat[amplitudes > 0.0])
                amplitude_mean = float(np.mean(shot_amplitudes))
                amplitude_std = float(np.std(shot_amplitudes))
                amplitude_sem = float(amplitude_std / np.sqrt(shot_amplitudes.size))
                phase_mean, phase_std, phase_resultant_length = _phase_statistics(flat)
                finite_shot_phases = shot_phases[np.isfinite(shot_phases)]
                if np.isfinite(phase_mean) and finite_shot_phases.size > 0:
                    phase_offsets = np.angle(
                        np.exp(1j * (finite_shot_phases - phase_mean))
                    )
                    phase_sem = float(
                        np.std(phase_offsets) / np.sqrt(finite_shot_phases.size)
                    )
                else:
                    phase_sem = np.nan
                statistics.append(
                    MonitorStatistic(
                        reference_target=target,
                        covered_targets=(target,),
                        monitor_target=target,
                        capture_index=capture_index,
                        amplitude_mean=amplitude_mean,
                        amplitude_std=amplitude_std,
                        amplitude_sem=amplitude_sem,
                        amplitude_relative_sem=_relative_sem(
                            amplitude_mean,
                            amplitude_sem,
                        ),
                        phase_mean_rad=phase_mean,
                        phase_std_rad=phase_std,
                        phase_sem_rad=phase_sem,
                        phase_resultant_length=phase_resultant_length,
                        n_shots=int(shot_amplitudes.size),
                        n_samples=int(flat.size),
                    )
                )
        return statistics

    def apply_schedule_corrections(
        self,
        schedule: MeasurementSchedule,
    ) -> MeasurementSchedule:
        """Return a schedule copy with session-local output corrections applied."""
        if not self.corrections_enabled or not self.has_output_signal_baseline:
            return schedule

        pulse_schedule = schedule.pulse_schedule
        waveforms = pulse_schedule.get_sampled_sequences(
            copy=True,
        )
        changed = False
        for label, waveform in list(waveforms.items()):
            correction = self._output_corrections.get(label)
            if correction is None:
                continue
            factor = correction.gain * np.exp(1j * correction.phase_offset_rad)
            if factor == 1.0:
                continue
            waveforms[label] = np.asarray(waveform) * factor
            changed = True
        if not changed:
            return schedule

        corrected = PulseSchedule.from_waveforms(waveforms)
        frequencies = {
            label: frequency
            for label, frequency in pulse_schedule.get_frequencies().items()
            if frequency is not None
        }
        if frequencies:
            corrected.set_frequencies(frequencies)
        return MeasurementSchedule(
            pulse_schedule=corrected,
            capture_schedule=schedule.capture_schedule,
        )

    def get_output_gain(self, target: str) -> float:
        """Return current output gain for a target label."""
        correction = self._output_corrections.get(target)
        return 1.0 if correction is None else correction.gain

    def get_output_phase_offset(self, target: str) -> float:
        """Return current output phase offset in radians for a target label."""
        correction = self._output_corrections.get(target)
        return 0.0 if correction is None else correction.phase_offset_rad

    def _resolve_output_targets(
        self,
        *,
        targets: Collection[str] | str | None,
        include_control: bool,
        include_readout: bool,
    ) -> list[Target]:
        target_map = {
            target.label: target for target in self.context.experiment_system.targets
        }
        if targets is None:
            labels = self._resolve_default_output_target_labels(target_map)
        elif isinstance(targets, str):
            labels = [targets]
        else:
            labels = list(dict.fromkeys(str(target) for target in targets))

        selected: list[Target] = []
        for label in labels:
            target = target_map.get(label)
            if target is None:
                raise ValueError(f"Unknown output target: {label}.")
            try:
                kind = self._resolve_output_kind(target)
            except ValueError:
                if targets is None:
                    continue
                raise
            if kind == "control" and not include_control:
                continue
            if kind == "readout" and not include_readout:
                continue
            selected.append(target)
        return selected

    def _resolve_default_output_target_labels(
        self,
        target_map: dict[str, Target],
    ) -> list[str]:
        """Return default output target labels within the active qubit set."""
        active_qubits = self._active_qubit_labels()
        if active_qubits is None:
            return list(target_map)
        if not active_qubits:
            return []
        return [
            target.label
            for target in target_map.values()
            if self._is_active_output_target(target, active_qubits)
        ]

    def _active_qubit_labels(self) -> list[str] | None:
        """Return active qubit labels when the context exposes them."""
        labels = getattr(self.context, "qubit_labels", None)
        if labels is None:
            return None
        return [str(label) for label in labels]

    def _is_active_output_target(
        self,
        target: Target,
        active_qubits: list[str],
    ) -> bool:
        """Return whether a default output target belongs to active qubits."""
        is_related_to_qubits = getattr(target, "is_related_to_qubits", None)
        if callable(is_related_to_qubits):
            if not bool(is_related_to_qubits(active_qubits)):
                return False
        elif str(getattr(target, "qubit", "")) not in active_qubits:
            return False

        if not bool(getattr(target, "is_cr", False)):
            return True
        try:
            _control_qubit, target_qubit = (
                self.context.experiment_system.resolve_cr_pair(target.label)
            )
        except ValueError:
            return True
        return target_qubit == "CR" or target_qubit in active_qubits

    def _group_reference_targets(
        self,
        targets: list[Target],
        *,
        reference_scope: OutputSignalReferenceScope,
    ) -> list[tuple[Target, list[Target]]]:
        """Group output targets by the requested reference scope."""
        if reference_scope == "target":
            return [(target, [target]) for target in targets]

        groups: dict[str, list[Target]] = {}
        for target in targets:
            groups.setdefault(str(target.channel.port.box_id), []).append(target)
        return [(group_targets[0], group_targets) for group_targets in groups.values()]

    def _warn_if_explicit_monitor_targets_replace_baseline(
        self,
        targets: list[Target],
        *,
        reference_scope: OutputSignalReferenceScope,
    ) -> None:
        """Warn when explicit monitor targets differ from stored box references."""
        if not self.has_output_signal_baseline:
            return
        baseline_by_box = {
            correction.port_id.split(".", maxsplit=1)[0]: correction.reference_target
            for correction in self._output_corrections.values()
            if correction.reference_scope == "box"
            and correction.reference_target is not None
        }
        if not baseline_by_box:
            return
        for reference_target, _group_targets in self._group_reference_targets(
            targets,
            reference_scope=reference_scope,
        ):
            box_id = str(reference_target.channel.port.box_id)
            baseline_reference = baseline_by_box.get(box_id)
            if (
                baseline_reference is None
                or reference_target.label == baseline_reference
            ):
                continue
            logger.warning(
                "Explicit monitor target %s differs from the stored baseline "
                "reference target %s on box %s. This may require changing the "
                "monitor NCO phase origin; prefer re-establishing the stability "
                "baseline before using this target for periodic correction.",
                reference_target.label,
                baseline_reference,
                box_id,
            )

    def _group_update_targets(
        self,
        targets: list[Target],
        *,
        reference_scope: OutputSignalReferenceScope,
    ) -> list[tuple[Target, list[Target]]]:
        """Group targets for correction updates using stored reference targets."""
        target_map = {
            target.label: target for target in self.context.experiment_system.targets
        }
        if reference_scope == "target":
            return [(target, [target]) for target in targets]

        groups: dict[str, list[Target]] = {}
        for target in targets:
            previous = self._output_corrections.get(target.label)
            if previous is None:
                continue
            reference_target = previous.reference_target or target.label
            groups.setdefault(reference_target, []).append(target)

        grouped_targets: list[tuple[Target, list[Target]]] = []
        for reference_target, group_targets in groups.items():
            representative = target_map.get(reference_target)
            if representative is None:
                representative = group_targets[0]
            grouped_targets.append((representative, group_targets))
        return grouped_targets

    def _resolve_update_reference_scope(
        self,
        reference_scope: OutputSignalReferenceScope | None,
    ) -> OutputSignalReferenceScope:
        """Resolve update reference scope from the baseline unless overridden."""
        if reference_scope is not None:
            return self._validate_reference_scope(reference_scope)
        scopes = {
            correction.reference_scope
            for correction in self._output_corrections.values()
        }
        if len(scopes) == 1:
            return self._validate_reference_scope(scopes.pop())
        return "target"

    @staticmethod
    def _validate_reference_scope(
        reference_scope: str,
    ) -> OutputSignalReferenceScope:
        """Return a validated output-signal reference scope."""
        if reference_scope not in ("box", "target"):
            raise ValueError("reference_scope must be 'box' or 'target'.")
        return cast(OutputSignalReferenceScope, reference_scope)

    def _measure_monitor_statistic(
        self,
        *,
        capture: Callable[..., MeasurementResult],
        target: Target,
        n_shots: int | None,
        probe_amplitude: float,
        probe_duration: float,
        block_outputs: bool,
        trim_samples: int = 0,
        shot_averaging: bool = True,
        configure_monitor_nco: bool | None = None,
    ) -> MonitorStatistic:
        monitor_target = self._resolve_monitor_target(target)
        schedule = self._build_probe_schedule(
            target=target,
            probe_amplitude=probe_amplitude,
            probe_duration=probe_duration,
        )
        result = _call_loopback_capture(
            capture,
            schedule,
            n_shots=n_shots,
            block_outputs=block_outputs,
            shot_averaging=shot_averaging,
            capture_targets=[monitor_target],
            configure_monitor_nco=configure_monitor_nco,
        )
        stats = self._compute_probe_monitor_statistics(
            result,
            reference_target=target.label,
            monitor_target=monitor_target,
            trim_samples=trim_samples,
        )
        if len(stats) == 0:
            raise ValueError(f"No monitor capture found for {monitor_target}.")
        return stats[0]

    @staticmethod
    def _build_probe_schedule(
        *,
        target: Target,
        probe_amplitude: float,
        probe_duration: float,
    ) -> PulseSchedule:
        with PulseSchedule([target.label]) as schedule:
            schedule.add(
                target.label,
                FlatTop(duration=probe_duration, amplitude=probe_amplitude, tau=0.0),
            )
        return schedule

    def _resolve_monitor_target(self, target: Target) -> str:
        port = target.channel.port
        for box in self.context.experiment_system.control_system.boxes:
            if box.id != port.box_id:
                continue
            for candidate in box.ports:
                if candidate.type == PortType.MNTR_IN:
                    return str(candidate.id)
        raise ValueError(f"No monitor input port found for box {port.box_id}.")

    @staticmethod
    def _resolve_output_kind(target: Target) -> OutputSignalKind:
        port_type = target.channel.port.type
        if port_type == PortType.READ_OUT:
            return "readout"
        if port_type == PortType.CTRL:
            return "control"
        raise ValueError(
            f"Unsupported output port type for {target.label}: {port_type}."
        )
