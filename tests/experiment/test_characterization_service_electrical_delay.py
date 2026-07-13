"""Tests for electrical-delay measurement reset behavior."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.experiment.services.characterization_service import CharacterizationService


class _FakeFigure:
    """Plotly-like figure stub for resonator scan tests."""

    def set_subplots(self, **_kwargs) -> None:
        """Accept subplot configuration."""
        return

    def add_scatter(self, **_kwargs) -> None:
        """Accept scatter traces."""
        return

    def add_vline(self, **_kwargs) -> None:
        """Accept vertical markers."""
        return

    def add_annotation(self, **_kwargs) -> None:
        """Accept annotations."""
        return

    def update_xaxes(self, **_kwargs) -> None:
        """Accept x-axis updates."""
        return

    def update_yaxes(self, **_kwargs) -> None:
        """Accept y-axis updates."""
        return

    def update_layout(self, **_kwargs) -> None:
        """Accept layout updates."""
        return


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


def test_scan_resonator_frequencies_forwards_interval_to_electrical_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given scan interval, internal electrical-delay measurement receives it."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _FakeContext()
    service.__dict__["_experiment_context"] = ctx

    electrical_delay_calls: list[dict[str, Any]] = []

    def _fake_measure_electrical_delay(
        self: CharacterizationService,
        target: str,
        **kwargs: Any,
    ) -> float:
        _ = self
        electrical_delay_calls.append({"target": target, **kwargs})
        return 0.0

    def _fake_measure(*_args, **_kwargs):
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    service.measure_electrical_delay = MethodType(
        _fake_measure_electrical_delay,
        service,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.ExperimentUtil.split_frequency_range",
        lambda **_kwargs: [
            np.array([9.8, 9.9]),
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
        frequency_range=np.array([9.8, 9.9]),
        readout_amplitude=0.1,
        plot=False,
        save_image=False,
        subrange_width=0.2,
        shots=7,
        interval=123.0,
    )

    assert "peaks" in result.data
    assert electrical_delay_calls == [
        {
            "target": "Q00",
            "f_start": 9.8,
            "shots": 7,
            "interval": 123.0,
            "plot": False,
            "confirm": False,
        }
    ]
