"""Tests for measurement stability corrections."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qxpulse import FlatTop, PulseSchedule

from qubex.measurement.measurement_defaults import (
    DEFAULT_STABILITY_CORRECTION_N_SHOTS,
    DEFAULT_STABILITY_CORRECTION_PROBE_DURATION,
)
from qubex.measurement.models.capture_data import CaptureData
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.measurement_stability import (
    MeasurementStabilitySnapshot,
)
from qubex.measurement.services.measurement_stability_service import (
    MeasurementStabilityService,
)
from qubex.system import PortType


def _make_config(
    *,
    n_shots: int = 2,
    shot_averaging: bool = True,
) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=n_shots,
        shot_interval=100.0,
        shot_averaging=shot_averaging,
        time_integration=False,
        state_classification=False,
    )


def _make_monitor_result(
    target: str,
    amplitude: float,
    phase: float = 0.0,
) -> MeasurementResult:
    return _make_monitor_result_data(
        target,
        np.array(
            [
                amplitude * np.exp(1j * phase),
                amplitude * np.exp(1j * phase),
            ]
        ),
    )


def _make_monitor_result_data(
    target: str,
    data: np.ndarray,
    *,
    shot_averaging: bool = True,
) -> MeasurementResult:
    config = _make_config(
        n_shots=np.asarray(data).shape[0] if not shot_averaging else 2,
        shot_averaging=shot_averaging,
    )
    return MeasurementResult(
        data={
            target: [
                CaptureData.from_primary_data(
                    target=target,
                    data=data,
                    config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )


def _make_context(
    *,
    active_qubits: list[str] | None = None,
    include_extra_targets: bool = False,
) -> Any:
    if active_qubits is None:
        active_qubits = ["Q00"]
    control_port = SimpleNamespace(
        id="B0.CTRL0.OUT",
        box_id="B0",
        type=PortType.CTRL,
    )
    readout_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        type=PortType.READ_OUT,
    )
    monitor_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        type=PortType.MNTR_IN,
    )
    other_control_port = SimpleNamespace(
        id="B1.CTRL0.OUT",
        box_id="B1",
        type=PortType.CTRL,
    )
    other_readout_port = SimpleNamespace(
        id="B1.READ0.OUT",
        box_id="B1",
        type=PortType.READ_OUT,
    )
    other_monitor_port = SimpleNamespace(
        id="B1.MNTR0.IN",
        box_id="B1",
        type=PortType.MNTR_IN,
    )

    def make_target(
        *,
        label: str,
        qubit: str,
        port: object,
        is_cr: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            label=label,
            qubit=qubit,
            is_cr=is_cr,
            channel=SimpleNamespace(port=port),
            is_related_to_qubits=lambda qubits, qubit=qubit: qubit in qubits,
        )

    control_target = make_target(label="Q00", qubit="Q00", port=control_port)
    readout_target = make_target(label="RQ00", qubit="Q00", port=readout_port)
    box = SimpleNamespace(id="B0", ports=[control_port, readout_port, monitor_port])
    other_box = SimpleNamespace(
        id="B1",
        ports=[other_control_port, other_readout_port, other_monitor_port],
    )
    targets = [control_target, readout_target]
    if include_extra_targets:
        targets.extend(
            [
                make_target(
                    label="Q00-CR",
                    qubit="Q00",
                    port=control_port,
                    is_cr=True,
                ),
                make_target(
                    label="Q00-Q02",
                    qubit="Q00",
                    port=control_port,
                    is_cr=True,
                ),
                make_target(label="Q02", qubit="Q02", port=other_control_port),
                make_target(label="RQ02", qubit="Q02", port=other_readout_port),
            ]
        )

    def resolve_cr_pair(label: str) -> tuple[str, str]:
        control, target = label.split("-", maxsplit=1)
        return control, target

    experiment_system = SimpleNamespace(
        targets=targets,
        control_system=SimpleNamespace(boxes=[box, other_box]),
        resolve_cr_pair=resolve_cr_pair,
    )
    return SimpleNamespace(
        experiment_system=experiment_system,
        qubit_labels=active_qubits,
    )


def test_measurement_stability_updates_and_applies_output_gains() -> None:
    """Given monitor drift, when corrections update, then schedules are scaled."""
    service = MeasurementStabilityService(context=_make_context())
    amplitudes = {"Q00": 1.0, "RQ00": 1.0}
    captured_labels: list[str] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (n_shots, block_outputs, capture_targets)
        captured_labels.append(schedule.labels[0])
        return _make_monitor_result("B0.MNTR0.IN", amplitudes[schedule.labels[0]])

    baseline = service.establish_output_signal_baseline(capture=capture)
    assert baseline.output_corrections["Q00"].gain == pytest.approx(1.0)
    assert baseline.output_corrections["RQ00"].gain == pytest.approx(1.0)
    assert baseline.output_corrections["RQ00"].reference_target == "Q00"
    assert baseline.output_corrections["RQ00"].reference_scope == "box"

    amplitudes = {"Q00": 2.0, "RQ00": 0.5}
    snapshot = service.update_output_signal_corrections(
        capture=capture,
        max_gain_relative_step=1.0,
        gain_smoothing=1.0,
    )
    assert snapshot.output_corrections["Q00"].gain == pytest.approx(0.5)
    assert snapshot.output_corrections["RQ00"].gain == pytest.approx(0.5)
    assert captured_labels == ["Q00", "Q00"]

    with PulseSchedule(["Q00", "RQ00"]) as schedule:
        schedule.add("Q00", FlatTop(duration=16, amplitude=0.2, tau=0.0))
        schedule.add("RQ00", FlatTop(duration=16, amplitude=0.3, tau=0.0))
    measurement_schedule = MeasurementSchedule(
        pulse_schedule=schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )

    corrected = service.apply_schedule_corrections(measurement_schedule)
    waveforms = corrected.pulse_schedule.get_sampled_sequences()

    assert waveforms["Q00"][0] == pytest.approx(0.1 + 0.0j)
    assert waveforms["RQ00"][0] == pytest.approx(0.15 + 0.0j)


def test_measurement_stability_update_reuses_baseline_monitor_nco() -> None:
    """Given a primed baseline, when updating, then monitor NCO is not reset."""
    service = MeasurementStabilityService(context=_make_context())
    configure_monitor_nco_values: list[bool | None] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
        configure_monitor_nco: bool | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, shot_averaging, capture_targets)
        configure_monitor_nco_values.append(configure_monitor_nco)
        return _make_monitor_result("B0.MNTR0.IN", 1.0, 0.0)

    service.establish_output_signal_baseline(capture=capture)
    service.update_output_signal_corrections(capture=capture)

    assert configure_monitor_nco_values == [True, False]


def test_measurement_stability_uses_stability_probe_defaults() -> None:
    """Given omitted probe settings, when updating, then stability defaults are used."""
    service = MeasurementStabilityService(context=_make_context())
    calls: list[tuple[int | None, float]] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
        configure_monitor_nco: bool | None = None,
    ) -> MeasurementResult:
        _ = (block_outputs, shot_averaging, capture_targets, configure_monitor_nco)
        sequence = schedule.get_sequence(schedule.labels[0])
        calls.append((n_shots, sequence.duration))
        return _make_monitor_result("B0.MNTR0.IN", 1.0, 0.0)

    service.update_output_signal_corrections(capture=capture)

    assert calls == [
        (
            DEFAULT_STABILITY_CORRECTION_N_SHOTS,
            DEFAULT_STABILITY_CORRECTION_PROBE_DURATION,
        )
    ]


def test_measurement_stability_warns_when_explicit_monitor_target_differs_from_baseline(
    caplog,
) -> None:
    """Given a box baseline, when another target is monitored, then it warns."""
    service = MeasurementStabilityService(context=_make_context())

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
        configure_monitor_nco: bool | None = None,
    ) -> MeasurementResult:
        _ = (
            schedule,
            n_shots,
            block_outputs,
            shot_averaging,
            capture_targets,
            configure_monitor_nco,
        )
        return _make_monitor_result("B0.MNTR0.IN", 1.0, 0.0)

    service.establish_output_signal_baseline(capture=capture)
    caplog.clear()

    with caplog.at_level(
        "WARNING",
        logger="qubex.measurement.services.measurement_stability_service",
    ):
        service.measure_monitor_statistics(
            capture=capture,
            targets=["RQ00"],
            reference_scope="target",
        )

    assert "Explicit monitor target RQ00 differs" in caplog.text
    assert "reference target Q00" in caplog.text


def test_measurement_stability_updates_and_applies_output_phase_offsets() -> None:
    """Given monitor phase drift, when corrections update, then schedules are rotated."""
    service = MeasurementStabilityService(context=_make_context())
    phase = 0.0

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, capture_targets)
        return _make_monitor_result("B0.MNTR0.IN", 1.0, phase)

    service.establish_output_signal_baseline(capture=capture)
    phase = 0.3
    snapshot = service.update_output_signal_corrections(
        capture=capture,
        max_gain_relative_step=1.0,
        gain_smoothing=1.0,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )
    correction = snapshot.output_corrections["Q00"]

    assert correction.reference_phase_rad == pytest.approx(0.0)
    assert correction.measured_phase_rad == pytest.approx(0.3)
    assert correction.raw_phase_offset_rad == pytest.approx(0.3)
    assert correction.phase_offset_rad == pytest.approx(0.3)
    assert service.get_output_phase_offset("Q00") == pytest.approx(0.3)

    with PulseSchedule(["Q00"]) as schedule:
        schedule.add("Q00", FlatTop(duration=16, amplitude=0.2, tau=0.0))
    measurement_schedule = MeasurementSchedule(
        pulse_schedule=schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )

    corrected = service.apply_schedule_corrections(measurement_schedule)
    waveforms = corrected.pulse_schedule.get_sampled_sequences()

    assert waveforms["Q00"][0] == pytest.approx(0.2 * np.exp(0.3j))


def test_measurement_stability_baseline_and_update_share_trimmed_samples() -> None:
    """Given trimmed probes, when updating, then baseline and update use the same window."""
    service = MeasurementStabilityService(context=_make_context())
    data = np.array([10.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 10.0 + 0.0j])

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, capture_targets)
        return _make_monitor_result_data("B0.MNTR0.IN", data)

    baseline = service.establish_output_signal_baseline(
        capture=capture,
        trim_samples=1,
    )
    data = np.array([10.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j, 10.0 + 0.0j])
    snapshot = service.update_output_signal_corrections(
        capture=capture,
        trim_samples=1,
        max_gain_relative_step=1.0,
        gain_smoothing=1.0,
        gain_correction_deadband=0.0,
    )

    assert baseline.output_corrections["Q00"].reference_amplitude == pytest.approx(1.0)
    assert snapshot.output_corrections["Q00"].measured_amplitude == pytest.approx(2.0)
    assert snapshot.output_corrections["Q00"].gain == pytest.approx(0.5)


def test_measurement_stability_default_targets_stay_with_active_qubits() -> None:
    """Given active qubits, when default baselines are captured, then other muxes are skipped."""
    service = MeasurementStabilityService(
        context=_make_context(active_qubits=["Q00"], include_extra_targets=True)
    )
    captured_labels: list[str] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (n_shots, block_outputs, capture_targets)
        captured_labels.append(schedule.labels[0])
        return _make_monitor_result("B0.MNTR0.IN", 1.0)

    baseline = service.establish_output_signal_baseline(capture=capture)

    assert set(baseline.output_corrections) == {"Q00", "Q00-CR", "RQ00"}
    assert captured_labels == ["Q00"]
    assert baseline.output_corrections["Q00-CR"].reference_target == "Q00"


def test_measurement_stability_captures_monitor_target_only() -> None:
    """Given monitor baseline capture, when probing, then READ_IN targets are not captured."""
    service = MeasurementStabilityService(context=_make_context())
    captured_targets: list[list[str] | None] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs)
        captured_targets.append(capture_targets)
        return _make_monitor_result("B0.MNTR0.IN", 1.0)

    service.establish_output_signal_baseline(capture=capture)

    assert captured_targets == [["B0.MNTR0.IN"]]


def test_measurement_stability_accepts_source_labeled_monitor_result() -> None:
    """Given source-labeled loopback result, baseline should still use monitor probe."""
    service = MeasurementStabilityService(context=_make_context())
    captured_targets: list[list[str] | None] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (n_shots, block_outputs)
        captured_targets.append(capture_targets)
        return _make_monitor_result(schedule.labels[0], 1.0)

    baseline = service.establish_output_signal_baseline(capture=capture)

    assert captured_targets == [["B0.MNTR0.IN"]]
    assert set(baseline.output_corrections) == {"Q00", "RQ00"}
    assert baseline.output_corrections["Q00"].monitor_target == "B0.MNTR0.IN"


def test_measurement_stability_statistics_show_covered_targets() -> None:
    """Given box scope diagnostics, when summarized, then covered targets are visible."""
    service = MeasurementStabilityService(context=_make_context())
    captured_targets: list[list[str] | None] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs)
        captured_targets.append(capture_targets)
        return _make_monitor_result(schedule.labels[0], 1.0)

    statistics = service.measure_monitor_statistics(
        capture=capture,
        targets=["Q00", "RQ00"],
        reference_scope="box",
    )
    assert statistics[0].reference_target == "Q00"
    assert statistics[0].covered_targets == ("Q00", "RQ00")
    assert captured_targets == [["B0.MNTR0.IN"]]
    assert statistics[0].monitor_target == "B0.MNTR0.IN"


def test_check_signal_stability_returns_snapshot_history(monkeypatch) -> None:
    """Given stability checks, when sampled, then snapshots include both signals."""
    service = MeasurementStabilityService(context=_make_context())
    clock = {"now": 0.0}

    def perf_counter() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(
        "qubex.measurement.services.measurement_stability_service.time.perf_counter",
        perf_counter,
    )
    monkeypatch.setattr(
        "qubex.measurement.services.measurement_stability_service.time.sleep",
        sleep,
    )

    amplitudes = [1.0, 2.0]
    phases = [0.0, 0.22]
    captured: list[tuple[list[str] | None, bool]] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs)
        sample_index = len(captured)
        captured.append((capture_targets, service.corrections_enabled))
        clock["now"] += 0.1
        value = amplitudes[sample_index] * np.exp(1j * phases[sample_index])
        if service.corrections_enabled:
            correction = service.snapshot().output_corrections["Q00"]
            value *= correction.gain * np.exp(1j * correction.phase_offset_rad)
        config = _make_config()
        return MeasurementResult(
            data={
                "B0.MNTR0.IN": [
                    CaptureData.from_primary_data(
                        target="B0.MNTR0.IN",
                        data=np.array([value, value]),
                        config=config,
                        sampling_period=2.0,
                    )
                ]
            },
            measurement_config=config,
        )

    snapshots = service.check_signal_stability(
        capture=capture,
        targets=["Q00"],
        duration=0.0,
        sample_interval=0.5,
        max_gain_relative_step=1.0,
        gain_smoothing=1.0,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )

    updated = service.snapshot().output_corrections["Q00"]
    assert len(snapshots) == 1
    assert isinstance(snapshots[0], MeasurementStabilitySnapshot)
    assert captured == [
        (["B0.MNTR0.IN"], False),
    ]
    assert clock["now"] == pytest.approx(0.1)
    assert snapshots[0].sample_index == 0
    assert snapshots[0].elapsed_s == pytest.approx(0.0)
    assert snapshots[0].output_corrections["Q00"].gain == pytest.approx(1.0)
    assert snapshots[0].output_corrections["Q00"].phase_offset_rad == pytest.approx(0.0)
    assert updated.gain == pytest.approx(1.0)
    assert updated.phase_offset_rad == pytest.approx(0.0)
    signal = next(iter(snapshots[0].signals.values()))
    assert signal.amplitude_mean == pytest.approx(1.0)
    assert signal.phase_mean_rad == pytest.approx(0.0)


def test_check_signal_stability_configures_monitor_nco_once(monkeypatch) -> None:
    """Given loopback NCO control, when stability is checked, then NCO is primed once."""
    service = MeasurementStabilityService(context=_make_context())
    clock = {"now": 0.0}

    def perf_counter() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(
        "qubex.measurement.services.measurement_stability_service.time.perf_counter",
        perf_counter,
    )
    monkeypatch.setattr(
        "qubex.measurement.services.measurement_stability_service.time.sleep",
        sleep,
    )

    configure_monitor_nco_values: list[bool | None] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
        configure_monitor_nco: bool | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, shot_averaging, capture_targets)
        configure_monitor_nco_values.append(configure_monitor_nco)
        return _make_monitor_result("B0.MNTR0.IN", 1.0, 0.0)

    _ = service.check_signal_stability(
        capture=capture,
        targets=["Q00"],
        duration=1.0,
        sample_interval=0.5,
    )

    assert configure_monitor_nco_values == [True, False, False]


def test_check_signal_stability_plot_option_plots_relative_amplitude_and_phase(
    monkeypatch,
) -> None:
    """Given plot requested, when stability is checked, then amplitude and phase are shown."""
    service = MeasurementStabilityService(context=_make_context())
    clock = {"now": 0.0}

    def perf_counter() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(
        "qubex.measurement.services.measurement_stability_service.time.perf_counter",
        perf_counter,
    )
    monkeypatch.setattr(
        "qubex.measurement.services.measurement_stability_service.time.sleep",
        sleep,
    )

    displayed: list[object] = []

    def display(widget: object) -> None:
        displayed.append(widget)

    monkeypatch.setattr("IPython.display.display", display)

    amplitudes = [2.0, 3.0, 4.0]
    phases = [3.0, -3.0, 3.1]

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, shot_averaging, capture_targets)
        index = len(calls)
        amplitude = amplitudes[index]
        phase = phases[index]
        calls.append(amplitude)
        return _make_monitor_result("B0.MNTR0.IN", amplitude, phase)

    calls: list[float] = []
    snapshots = service.check_signal_stability(
        capture=capture,
        targets=["Q00"],
        duration=1.0,
        sample_interval=0.5,
        update_corrections=False,
        plot=True,
    )

    assert len(snapshots) == 3
    assert calls == amplitudes
    assert len(displayed) == 1
    stability_widget, waveform_widget = displayed[0].children
    assert len(stability_widget.data) == 2
    assert list(stability_widget.data[0].x) == pytest.approx([0.0, 0.5, 1.0])
    assert list(stability_widget.data[0].y) == pytest.approx([1.0, 1.5, 2.0])
    assert list(stability_widget.data[0].customdata) == pytest.approx(
        [0.0, 50.0, 100.0]
    )
    assert stability_widget.data[0].hovertemplate == (
        "elapsed=%{x:.2f} s<br>"
        "relative=%{y:.4f}<br>"
        "move=%{customdata:+.2f}%<extra>%{fullData.name}</extra>"
    )
    assert list(stability_widget.data[1].x) == pytest.approx([0.0, 0.5, 1.0])
    assert list(stability_widget.data[1].y) == pytest.approx(
        [0.0, 0.28318530717958645, 0.1]
    )
    assert stability_widget.data[1].line.color == "#00B945"
    assert stability_widget.data[1].hovertemplate == (
        "elapsed=%{x:.2f} s<br>phase=%{y:+.4f} rad<extra>%{fullData.name}</extra>"
    )
    assert stability_widget.layout.width == 800
    assert stability_widget.layout.font.family == "Times New Roman, Times, serif"
    assert stability_widget.layout.yaxis.title.text == (
        "relative amplitude (initial=1)"
    )
    assert stability_widget.layout.yaxis2.title.text == ("phase shift (rad, initial=0)")
    amplitude_domain = stability_widget.layout.yaxis.domain
    phase_domain = stability_widget.layout.yaxis2.domain
    assert amplitude_domain[1] - amplitude_domain[0] == pytest.approx(
        phase_domain[1] - phase_domain[0]
    )
    assert stability_widget.layout.xaxis2.title.text == "elapsed time (s)"
    assert list(waveform_widget.data[0].x) == pytest.approx([0.0, 2.0])
    assert list(waveform_widget.data[0].y) == pytest.approx([4.0, 4.0])
    assert waveform_widget.data[0].hovertemplate == (
        "time=%{x:.1f} ns<br>|IQ|=%{y:.4g}<extra>%{fullData.name}</extra>"
    )
    assert waveform_widget.layout.width == 800
    assert waveform_widget.layout.font.family == "Times New Roman, Times, serif"
    assert waveform_widget.layout.yaxis.title.text == "|IQ|"


def test_check_signal_stability_updates_phase_from_corrected_residual(
    monkeypatch,
) -> None:
    """Given corrected samples, when phase is step-limited, then updates keep converging."""
    service = MeasurementStabilityService(context=_make_context())
    clock = {"now": 0.0}

    def perf_counter() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(
        "qubex.measurement.services.measurement_stability_service.time.perf_counter",
        perf_counter,
    )
    monkeypatch.setattr(
        "qubex.measurement.services.measurement_stability_service.time.sleep",
        sleep,
    )

    true_amplitude = 2.0
    true_phase = 0.3
    observed_amplitudes: list[float] = []
    observed_phases: list[float] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, shot_averaging, capture_targets)
        amplitude = 1.0
        phase = 0.0
        if service.has_output_signal_baseline:
            amplitude = true_amplitude
            phase = true_phase
        if service.corrections_enabled:
            amplitude *= service.get_output_gain("Q00")
            phase -= service.get_output_phase_offset("Q00")
        observed_amplitudes.append(amplitude)
        observed_phases.append(phase)
        return _make_monitor_result("B0.MNTR0.IN", amplitude, phase)

    snapshots = service.check_signal_stability(
        capture=capture,
        targets=["Q00"],
        duration=1.0,
        sample_interval=0.5,
        max_gain_relative_step=1.0,
        gain_smoothing=1.0,
        max_phase_step=0.1,
        phase_smoothing=1.0,
        phase_correction_deadband=0.0,
        auto_phase_correction_deadband=False,
        auto_gain_correction_deadband=False,
    )

    phase_offsets = [
        snapshot.output_corrections["Q00"].phase_offset_rad for snapshot in snapshots
    ]
    gains = [snapshot.output_corrections["Q00"].gain for snapshot in snapshots]
    final_correction = service.snapshot().output_corrections["Q00"]

    assert observed_amplitudes == [
        pytest.approx(1.0),
        pytest.approx(2.0),
        pytest.approx(1.0),
    ]
    assert observed_phases == [
        pytest.approx(0.0),
        pytest.approx(0.3),
        pytest.approx(0.2),
    ]
    assert phase_offsets == [
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(0.1),
    ]
    assert gains == [
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(0.5),
    ]
    assert final_correction.raw_gain == pytest.approx(0.5)
    assert final_correction.gain == pytest.approx(0.5)
    assert final_correction.raw_phase_offset_rad == pytest.approx(0.3)
    assert final_correction.phase_offset_rad == pytest.approx(0.2)


def test_measurement_stability_box_scope_updates_subset_from_representative() -> None:
    """Given a box reference, when updating one target, then it probes the representative."""
    service = MeasurementStabilityService(context=_make_context())
    amplitudes = {"Q00": 1.0, "RQ00": 0.25}
    captured_labels: list[str] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (n_shots, block_outputs, capture_targets)
        captured_labels.append(schedule.labels[0])
        return _make_monitor_result("B0.MNTR0.IN", amplitudes[schedule.labels[0]])

    service.establish_output_signal_baseline(capture=capture)
    amplitudes = {"Q00": 2.0, "RQ00": 0.25}
    snapshot = service.update_output_signal_corrections(
        capture=capture,
        targets=["RQ00"],
        max_gain_relative_step=1.0,
        gain_smoothing=1.0,
    )

    assert captured_labels == ["Q00", "Q00"]
    assert snapshot.output_corrections["Q00"].gain == pytest.approx(1.0)
    assert snapshot.output_corrections["RQ00"].gain == pytest.approx(0.5)


def test_measurement_stability_bounds_gain_updates() -> None:
    """Given large monitor drift, when updating, then gain changes are bounded."""
    service = MeasurementStabilityService(context=_make_context())
    amplitude = 2.4

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, capture_targets)
        return _make_monitor_result("B0.MNTR0.IN", amplitude)

    service.establish_output_signal_baseline(capture=capture)
    amplitude = 3.0

    snapshot = service.update_output_signal_corrections(
        capture=capture,
        max_gain_relative_step=0.005,
        gain_smoothing=0.5,
    )
    correction = snapshot.output_corrections["Q00"]

    assert correction.raw_gain == pytest.approx(0.8)
    assert correction.limited_gain == pytest.approx(0.995)
    assert correction.gain == pytest.approx(0.9975)


def test_measurement_stability_deadband_skips_small_gain_updates() -> None:
    """Given small monitor drift, when inside deadband, then gain stays unchanged."""
    service = MeasurementStabilityService(context=_make_context())
    amplitude = 1.0

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, capture_targets)
        return _make_monitor_result("B0.MNTR0.IN", amplitude)

    service.establish_output_signal_baseline(capture=capture)
    amplitude = 1.001

    snapshot = service.update_output_signal_corrections(
        capture=capture,
        max_gain_relative_step=0.005,
        gain_smoothing=1.0,
        gain_correction_deadband=0.002,
    )
    correction = snapshot.output_corrections["Q00"]

    assert correction.raw_gain == pytest.approx(1.0 / 1.001)
    assert correction.limited_gain == pytest.approx(1.0)
    assert correction.gain == pytest.approx(1.0)


def test_measurement_stability_computes_shot_amplitude_uncertainty() -> None:
    """Given shot waveforms, when statistics are computed, then SEM uses shot amplitudes."""
    service = MeasurementStabilityService(context=_make_context())
    data = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 1.0 + 0.0j],
            [3.0 + 0.0j, 3.0 + 0.0j],
            [3.0 + 0.0j, 3.0 + 0.0j],
        ]
    )
    result = _make_monitor_result_data(
        "B0.MNTR0.IN",
        data,
        shot_averaging=False,
    )

    statistic = service.compute_monitor_statistics(result)[0]

    assert statistic.amplitude_mean == pytest.approx(2.0)
    assert statistic.amplitude_std == pytest.approx(1.0)
    assert statistic.amplitude_sem == pytest.approx(0.5)
    assert statistic.amplitude_relative_sem == pytest.approx(0.25)
    assert statistic.n_shots == 4
    assert statistic.n_samples == 8


def test_measurement_stability_computes_shot_phase_uncertainty() -> None:
    """Given shot phases, when statistics are computed, then phase SEM uses shot phases."""
    service = MeasurementStabilityService(context=_make_context())
    phases = np.array([-0.1, -0.1, 0.1, 0.1])
    data = np.exp(1j * phases)[:, np.newaxis] * np.ones((4, 2))
    result = _make_monitor_result_data(
        "B0.MNTR0.IN",
        data,
        shot_averaging=False,
    )

    statistic = service.compute_monitor_statistics(result)[0]

    assert statistic.phase_mean_rad == pytest.approx(0.0)
    assert statistic.phase_std_rad == pytest.approx(0.1)
    assert statistic.phase_sem_rad == pytest.approx(0.05)
    assert statistic.phase_resultant_length > 0.99
    assert statistic.n_shots == 4


def test_measurement_stability_auto_gain_deadband_uses_shot_uncertainty() -> None:
    """Given noisy shot amplitudes, when auto deadband is enabled, then gain is held."""
    service = MeasurementStabilityService(context=_make_context())
    captures = [
        np.array(
            [
                [0.99 + 0.0j, 0.99 + 0.0j],
                [1.01 + 0.0j, 1.01 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.991 + 0.0j, 0.991 + 0.0j],
                [1.011 + 0.0j, 1.011 + 0.0j],
            ]
        ),
    ]

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, capture_targets)
        return _make_monitor_result_data(
            "B0.MNTR0.IN",
            captures.pop(0),
            shot_averaging=shot_averaging,
        )

    service.establish_output_signal_baseline(capture=capture)
    snapshot = service.update_output_signal_corrections(
        capture=capture,
        max_gain_relative_step=1.0,
        gain_smoothing=1.0,
        gain_correction_deadband=0.0,
        auto_gain_correction_deadband=True,
        gain_correction_deadband_sigma=3.0,
    )
    correction = snapshot.output_corrections["Q00"]

    assert correction.raw_gain == pytest.approx(1.0 / 1.001)
    assert correction.effective_gain_correction_deadband > abs(
        correction.raw_gain - 1.0
    )
    assert correction.limited_gain == pytest.approx(1.0)
    assert correction.gain == pytest.approx(1.0)


def test_measurement_stability_auto_phase_deadband_uses_shot_uncertainty() -> None:
    """Given noisy shot phases, when auto deadband is enabled, then phase is held."""
    service = MeasurementStabilityService(context=_make_context())
    captures = [
        np.exp(1j * np.array([-0.1, 0.1]))[:, np.newaxis] * np.ones((2, 2)),
        np.exp(1j * np.array([-0.08, 0.12]))[:, np.newaxis] * np.ones((2, 2)),
    ]

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, capture_targets)
        return _make_monitor_result_data(
            "B0.MNTR0.IN",
            captures.pop(0),
            shot_averaging=shot_averaging,
        )

    service.establish_output_signal_baseline(capture=capture)
    snapshot = service.update_output_signal_corrections(
        capture=capture,
        max_phase_step=1.0,
        phase_smoothing=1.0,
        phase_correction_deadband=0.0,
        auto_phase_correction_deadband=True,
        phase_correction_deadband_sigma=3.0,
    )
    correction = snapshot.output_corrections["Q00"]

    assert correction.measured_phase_rad == pytest.approx(0.02)
    assert correction.effective_phase_correction_deadband > 0.02
    assert correction.raw_phase_offset_rad == pytest.approx(0.0)
    assert correction.limited_phase_offset_rad == pytest.approx(0.0)
    assert correction.phase_offset_rad == pytest.approx(0.0)


def test_measurement_stability_ignores_low_resultant_phase() -> None:
    """Given incoherent phase samples, when updating, then phase correction is held."""
    service = MeasurementStabilityService(context=_make_context())
    low_resultant_samples = np.exp(
        1j * (1.2 + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 0.0]))
    )
    captures = [
        np.ones((4, 1), dtype=np.complex128),
        low_resultant_samples[:, np.newaxis],
    ]

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, shot_averaging, capture_targets)
        return _make_monitor_result_data(
            "B0.MNTR0.IN",
            captures.pop(0),
            shot_averaging=shot_averaging,
        )

    service.establish_output_signal_baseline(capture=capture)
    snapshot = service.update_output_signal_corrections(
        capture=capture,
        max_phase_step=1.0,
        phase_smoothing=1.0,
        phase_correction_deadband=0.0,
        auto_gain_correction_deadband=False,
        auto_phase_correction_deadband=False,
        phase_min_resultant_length=0.5,
    )

    correction = snapshot.output_corrections["Q00"]
    assert correction.phase_offset_rad == pytest.approx(0.0)
    assert correction.raw_phase_offset_rad == pytest.approx(0.0)


def test_measurement_stability_target_scope_measures_each_target() -> None:
    """Given target reference scope, when updating, then each target is probed independently."""
    service = MeasurementStabilityService(context=_make_context())
    amplitudes = {"Q00": 1.0, "RQ00": 1.0}
    captured_labels: list[str] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
    ) -> MeasurementResult:
        _ = (n_shots, block_outputs, capture_targets)
        captured_labels.append(schedule.labels[0])
        return _make_monitor_result("B0.MNTR0.IN", amplitudes[schedule.labels[0]])

    service.establish_output_signal_baseline(
        capture=capture,
        reference_scope="target",
    )
    amplitudes = {"Q00": 2.0, "RQ00": 0.5}
    snapshot = service.update_output_signal_corrections(
        capture=capture,
        reference_scope="target",
        max_gain_relative_step=1.0,
        gain_smoothing=1.0,
    )

    assert captured_labels == ["Q00", "RQ00", "Q00", "RQ00"]
    assert snapshot.output_corrections["Q00"].gain == pytest.approx(0.5)
    assert snapshot.output_corrections["RQ00"].gain == pytest.approx(2.0)


def test_measurement_stability_update_without_baseline_establishes_baseline() -> None:
    """Given no baseline, when updating corrections, then it captures baseline."""
    service = MeasurementStabilityService(context=_make_context())
    configure_monitor_nco_values: list[bool | None] = []

    def capture(
        schedule: PulseSchedule,
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        capture_targets: list[str] | None = None,
        configure_monitor_nco: bool | None = None,
    ) -> MeasurementResult:
        _ = (schedule, n_shots, block_outputs, capture_targets)
        configure_monitor_nco_values.append(configure_monitor_nco)
        return _make_monitor_result("B0.MNTR0.IN", 1.0)

    snapshot = service.update_output_signal_corrections(capture=capture)

    assert service.has_output_signal_baseline
    assert configure_monitor_nco_values == [True]
    assert snapshot.output_corrections["Q00"].gain == pytest.approx(1.0)
