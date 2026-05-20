"""Tests for electrical-delay measurement reset behavior."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.experiment.services.characterization_service import CharacterizationService


class _FakeSystemManager:
    """System-manager stub for electrical-delay tests."""

    def __init__(self) -> None:
        self.modified_backend_settings_calls = 0

    @contextmanager
    def modified_backend_settings(self, *_args, **_kwargs):
        """Given a settings override request, yield without side effects."""
        self.modified_backend_settings_calls += 1
        yield


class _FakeContext:
    """Experiment-context stub for electrical-delay tests."""

    def __init__(self) -> None:
        self.qubit_labels = ["Q00"]
        self.params = SimpleNamespace(readout_amplitude={"Q00": 0.1})
        self.targets = {
            "R00": SimpleNamespace(sideband="U", fine_frequency=1.0),
        }
        self.current_frequency = 1.0
        self.reset_calls: list[list[str] | None] = []
        self.system_manager = _FakeSystemManager()
        read_box = SimpleNamespace(
            id="BOX0",
            traits=SimpleNamespace(
                readout_cnco_center=1_500_000_000,
                readout_ssb="U",
            ),
        )
        self.experiment_system = SimpleNamespace(
            get_mux_by_qubit=lambda _qubit: SimpleNamespace(label="MUX0"),
            get_readout_box_for_qubit=lambda _qubit: read_box,
        )

    @staticmethod
    def resolve_read_label(_target: str) -> str:
        """Resolve readout label used by characterization service."""
        return "R00"

    @staticmethod
    def resolve_qubit_label(_target: str) -> str:
        """Resolve qubit label used by characterization service."""
        return "Q00"

    def reset_awg_and_capunits(self, box_ids=None, qubits=None) -> None:
        """Record reset calls."""
        _ = qubits
        self.reset_calls.append(box_ids)

    @contextmanager
    def modified_frequencies(self, frequencies: dict[str, float]):
        """Apply a temporary frequency for measurement stubs."""
        self.current_frequency = next(iter(frequencies.values()))
        yield


class _Quel3SystemManager:
    """QuEL-3 system-manager stub that rejects QuEL-1 backend retunes."""

    backend_kind = "quel3"

    @contextmanager
    def modified_backend_settings(self, *_args, **_kwargs):
        """Given QuEL-3 spectroscopy, backend-settings retunes must not be used."""
        raise AssertionError("QuEL-3 spectroscopy must not retune backend settings")
        yield


class _FakeFigure:
    """Plotly-like no-op figure for characterization tests."""

    def set_subplots(self, **_kwargs) -> None:
        """Given subplot setup, ignore it."""
        return

    def add_scatter(self, **_kwargs) -> None:
        """Given plotted data, ignore it."""
        return

    def add_vline(self, **_kwargs) -> None:
        """Given vertical marker, ignore it."""
        return

    def add_annotation(self, **_kwargs) -> None:
        """Given annotation, ignore it."""
        return

    def update_xaxes(self, **_kwargs) -> None:
        """Given x-axis update, ignore it."""
        return

    def update_yaxes(self, **_kwargs) -> None:
        """Given y-axis update, ignore it."""
        return

    def update_layout(self, **_kwargs) -> None:
        """Given layout update, ignore it."""
        return

    def show(self) -> None:
        """Given show request, ignore it."""
        return


class _Quel3Context(_FakeContext):
    """Experiment-context stub for QuEL-3 spectroscopy tests."""

    def __init__(self) -> None:
        super().__init__()
        self.params = SimpleNamespace(
            control_amplitude={"Q00": 0.1},
            readout_amplitude={"Q00": 0.1},
        )
        self.targets = {
            "Q00": SimpleNamespace(frequency=4.9),
            "R00": SimpleNamespace(
                frequency=6.2,
                sideband=None,
                fine_frequency=6.2,
            ),
        }
        self.system_manager = _Quel3SystemManager()
        read_box = SimpleNamespace(
            id="BOX0",
            traits=SimpleNamespace(
                default_readout_frequency_range=(6.15, 6.25, 0.01),
                readout_cnco_center=None,
                readout_ssb=None,
            ),
        )
        ctrl_box = SimpleNamespace(
            id="BOX0",
            traits=SimpleNamespace(
                default_control_frequency_range=(4.8, 5.0, 0.01),
                ctrl_ssb=None,
            ),
        )
        self.experiment_system = SimpleNamespace(
            get_mux_by_qubit=lambda _qubit: SimpleNamespace(label="MUX0"),
            get_readout_box_for_qubit=lambda _qubit: read_box,
            get_control_box_for_qubit=lambda _qubit: ctrl_box,
        )


def test_measure_electrical_delay_skips_redundant_reset_when_backend_settings_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given LO/CNCO retune path, electrical-delay measurement skips redundant reset."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _FakeContext()
    service.__dict__["_experiment_context"] = ctx

    def _fake_measure(*_args, **_kwargs):
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    service.__dict__["_measurement_service"] = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.MixingUtil.calc_lo_cnco",
        lambda *_args, **_kwargs: (10_000_000_000, 1_500_000_000, 0),
    )

    tau = service.measure_electrical_delay(
        target="Q00",
        f_start=1.5,
        df=0.0001,
        n_samples=4,
        readout_amplitude=0.1,
        shots=1,
        interval=0,
        plot=False,
        confirm=False,
    )

    assert isinstance(tau, float)
    assert ctx.system_manager.modified_backend_settings_calls == 1
    assert ctx.reset_calls == []


def test_measure_electrical_delay_quel3_uses_direct_frequency_sweep() -> None:
    """Given QuEL-3 backend, electrical-delay measurement avoids LO/CNCO retunes."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _Quel3Context()
    service.__dict__["_experiment_context"] = ctx

    def _fake_measure(*_args, **_kwargs):
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    service.__dict__["_measurement_service"] = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    tau = service.measure_electrical_delay(
        target="Q00",
        f_start=6.5,
        df=0.0001,
        n_samples=4,
        readout_amplitude=0.1,
        shots=1,
        interval=0,
        plot=False,
        confirm=False,
    )

    assert isinstance(tau, float)
    assert ctx.reset_calls == []


def test_scan_resonator_frequencies_avoids_duplicate_reset_per_subrange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given subrange LO retune, scan avoids explicit reset duplication per subrange."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _FakeContext()
    service.__dict__["_experiment_context"] = ctx

    def _fake_measure(*_args, **_kwargs):
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    service.__dict__["_measurement_service"] = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.ExperimentUtil.split_frequency_range",
        lambda **_kwargs: [
            np.array([9.8, 9.9]),
            np.array([10.1, 10.2]),
        ],
    )
    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.MixingUtil.calc_lo_cnco",
        lambda *_args, **_kwargs: (10_000_000_000, 1_500_000_000, 0),
    )
    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.viz.make_figure",
        lambda **_kwargs: _FakeFigure(),
    )
    monkeypatch.setattr(
        "scipy.signal.find_peaks",
        lambda values, **_kwargs: (np.array([], dtype=int), {}),
    )

    result = service.scan_resonator_frequencies(
        target="Q00",
        frequency_range=np.array([9.8, 9.9, 10.1, 10.2]),
        electrical_delay=0.0,
        readout_amplitude=0.1,
        plot=False,
        save_image=False,
        subrange_width=0.2,
        shots=1,
        interval=0,
    )

    assert "peaks" in result.data
    assert ctx.system_manager.modified_backend_settings_calls == 2
    assert ctx.reset_calls == []


def test_scan_resonator_frequencies_quel3_uses_direct_frequency_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given QuEL-3 backend, resonator spectroscopy avoids LO/CNCO retunes."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _Quel3Context()
    service.__dict__["_experiment_context"] = ctx

    def _fake_measure(*_args, **_kwargs):
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    service.__dict__["_measurement_service"] = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.viz.make_figure",
        lambda **_kwargs: _FakeFigure(),
    )
    monkeypatch.setattr(
        "scipy.signal.find_peaks",
        lambda values, **_kwargs: (np.array([], dtype=int), {}),
    )

    result = service.scan_resonator_frequencies(
        target="Q00",
        frequency_range=np.array([6.15, 6.16, 6.17]),
        electrical_delay=0.0,
        readout_amplitude=0.1,
        plot=False,
        save_image=False,
        subrange_width=0.02,
        shots=1,
        interval=0,
    )

    assert result.data["peaks"].size == 0


def test_scan_qubit_frequencies_quel3_uses_direct_frequency_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given QuEL-3 backend, qubit spectroscopy avoids LO/CNCO retunes."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _Quel3Context()
    service.__dict__["_experiment_context"] = ctx

    def _fake_execute(*_args, **_kwargs):
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": [SimpleNamespace(kerneled=signal)]})

    service.__dict__["_measurement_service"] = SimpleNamespace(execute=_fake_execute)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.viz.make_figure",
        lambda **_kwargs: _FakeFigure(),
    )
    monkeypatch.setattr(
        "scipy.signal.find_peaks",
        lambda values, **_kwargs: (np.array([], dtype=int), {}),
    )

    result = service.scan_qubit_frequencies(
        target="Q00",
        frequency_range=np.array([4.85, 4.86, 4.87]),
        control_amplitude=0.1,
        readout_amplitude=0.1,
        readout_frequency=6.2,
        subrange_width=0.02,
        shots=1,
        interval=0,
        plot=False,
        save_image=False,
    )

    assert result.data["peaks"].size == 0
    assert result.data["frequency_guess"]["f_ge"] is None
