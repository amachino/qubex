"""Tests for electrical-delay measurement reset behavior."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.experiment.services.characterization_service import CharacterizationService
from qubex.system.system_manager import SystemManager


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

    def show(self) -> None:
        """Accept show requests."""
        return


class _FakeSystemManager:
    """System-manager stub for electrical-delay tests."""

    def __init__(self) -> None:
        self.modified_backend_settings_calls = 0

    def frequency_sweep_reference(self, target):
        """Provide a starting carrier without backend-specific metadata."""
        return 1.0

    def plan_frequency_sweep(
        self, target, frequencies, *, subrange_width=None, overlap=False, **kwargs
    ):
        """Provide windows to the characterization workflow."""
        from qubex.backend.frequency_sweep import plan_frequency_sweep

        return plan_frequency_sweep(
            frequencies,
            subrange_width=subrange_width,
            overlap=overlap,
            configure=lambda points, _reference: (
                min(points) - 1,
                max(points) + 1,
                {},
                True,
            ),
        )

    @contextmanager
    def modified_sweep_configuration(self, configuration):
        """Count one preparation per window."""
        self.modified_backend_settings_calls += 1
        yield

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
        self.measurement = SimpleNamespace()
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
        target = SimpleNamespace(label="R00", frequency=self.current_frequency)
        control = SimpleNamespace(label="Q00", frequency=self.current_frequency)

        def update_targets(values):
            for item in (target, control):
                if item.label in values:
                    item.frequency = values[item.label]

        def update_backend(values):
            self.current_frequency = next(iter(values.values()))

        manager = SimpleNamespace(
            experiment_system=SimpleNamespace(
                targets=[control, target], modify_target_frequencies=update_targets
            ),
            backend_controller=SimpleNamespace(
                modify_target_frequencies=update_backend
            ),
        )
        with SystemManager.modified_frequencies(cast(Any, manager), frequencies):
            yield


class _Quel3SystemManager(_FakeSystemManager):
    """QuEL-3 system-manager stub that rejects QuEL-1 backend retunes."""

    backend_kind = "quel3"

    @contextmanager
    def modified_backend_settings(self, *_args, **_kwargs):
        """Given QuEL-3 spectroscopy, backend-settings retunes must not be used."""
        raise AssertionError("QuEL-3 spectroscopy must not retune backend settings")
        yield


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
        assert "frequencies" not in _kwargs
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    ctx.measurement = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
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
    assert ctx.current_frequency == 1.0


def test_measure_electrical_delay_quel3_uses_direct_frequency_sweep() -> None:
    """Given QuEL-3 backend, electrical-delay measurement avoids LO/CNCO retunes."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _Quel3Context()
    service.__dict__["_experiment_context"] = ctx

    def _fake_measure(*_args, **_kwargs):
        assert "frequencies" not in _kwargs
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    ctx.measurement = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
    )

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
    assert ctx.current_frequency == 1.0


def test_scan_resonator_frequencies_avoids_duplicate_reset_per_subrange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given subrange LO retune, scan avoids explicit reset duplication per subrange."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _FakeContext()
    service.__dict__["_experiment_context"] = ctx

    def _fake_measure(*_args, **_kwargs):
        assert "frequencies" not in _kwargs
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    ctx.measurement = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
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
    assert ctx.current_frequency == 1.0


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
        assert "frequencies" not in _kwargs
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    service.measure_electrical_delay = MethodType(
        _fake_measure_electrical_delay,
        service,
    )
    ctx.measurement = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
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


def test_scan_resonator_frequencies_quel3_uses_direct_frequency_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given QuEL-3 backend, resonator spectroscopy avoids LO/CNCO retunes."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _Quel3Context()
    service.__dict__["_experiment_context"] = ctx

    def _fake_measure(*_args, **_kwargs):
        assert "frequencies" not in _kwargs
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=signal)})

    ctx.measurement = SimpleNamespace(measure=_fake_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
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
        assert "frequencies" not in _kwargs
        signal = np.exp(-1j * 2 * np.pi * ctx.current_frequency)
        return SimpleNamespace(data={"Q00": [SimpleNamespace(kerneled=signal)]})

    ctx.measurement = SimpleNamespace(execute=_fake_execute)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
    )

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


def test_resonator_scan_corrects_phase_jump_at_configuration_boundary(monkeypatch):
    """A reference acquired in the new configuration removes its phase jump."""
    ctx = _FakeContext()
    calls = []

    def measure(*args, **kwargs):
        frequency = ctx.current_frequency
        assert "frequencies" not in kwargs
        window = ctx.system_manager.modified_backend_settings_calls
        calls.append((window, frequency, kwargs["readout_amplitudes"]["Q00"]))
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(kerneled=np.exp(1j * (frequency + 0.7 * window)))
            }
        )

    ctx.measurement = SimpleNamespace(measure=measure)
    service = CharacterizationService(
        experiment_context=cast(Any, ctx),
        measurement_service=cast(Any, None),
        calibration_service=cast(Any, None),
        pulse_service=cast(
            Any,
            SimpleNamespace(
                readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
            ),
        ),
    )
    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.viz.make_figure",
        lambda **kwargs: _FakeFigure(),
    )
    monkeypatch.setattr(
        "scipy.signal.find_peaks",
        lambda values, **kwargs: (np.array([], dtype=int), {}),
    )
    result = service.scan_resonator_frequencies(
        "Q00",
        frequency_range=[5.0, 5.1, 5.4, 5.5],
        readout_amplitude=0.2,
        electrical_delay=0,
        subrange_width=0.2,
        plot=False,
        save_image=False,
    )
    np.testing.assert_allclose(
        result.data["phases_diff"], [0.1, 0.3, 0.1], atol=1e-12, rtol=1e-12
    )
    assert calls == [
        (1, 5.0, 0.2),
        (1, 5.1, 0.2),
        (2, 5.1, 0.2),
        (2, 5.4, 0.2),
        (3, 5.4, 0.2),
        (3, 5.5, 0.2),
    ]


@pytest.mark.parametrize("fail_acquisition", [False, True])
def test_electrical_delay_updates_quel1_converter_carrier(
    monkeypatch, fail_acquisition
):
    """Electrical-delay acquisitions update the driver carrier and restore it on exit."""
    from unittest.mock import Mock

    from quel_ic_config import QUEL1_BOXTYPE_ALIAS
    from qxdriver_quel1.pulse import GenSampledSequence, GenSampledSubSequence
    from qxdriver_quel1.runtime.converter import Converter
    from qxdriver_quel1.sysconf import BoxSetting, PortSetting

    ctx = _FakeContext()
    ctx.current_frequency = 5.0
    multiplex = Mock(wraps=Converter.multiplex)
    monkeypatch.setattr(Converter, "multiplex", multiplex)
    carriers = []

    def measure(*args, **kwargs):
        assert "frequencies" not in kwargs
        frequency = 5.01 + len(carriers) * 0.0001
        sequence = GenSampledSequence(
            target_name="R00",
            prev_blank=0,
            post_blank=0,
            repeats=1,
            padding=2,
            modulation_frequency=frequency - 5.0,
            sub_sequences=[
                GenSampledSubSequence(
                    real=np.ones(3) * 0.1, imag=np.zeros(3), repeats=1, post_blank=0
                )
            ],
        )
        Converter.convert_to_gen_device_specific_sequence(
            gen_sampled_sequence={"R00": sequence},
            cap_sampled_sequence={},
            resource_map={
                "R00": {
                    "box": BoxSetting(
                        box_name="B0",
                        ipaddr_wss="127.0.0.1",
                        boxtype=next(iter(QUEL1_BOXTYPE_ALIAS.values())),
                    ),
                    "port": PortSetting(
                        port_name="P0", box_name="B0", port=0, ndelay_or_nwait=(0,)
                    ),
                    "channel_number": 0,
                    "target": {"frequency": ctx.current_frequency},
                }
            },
            port_config=cast(
                Any,
                {
                    "R00": SimpleNamespace(
                        lo_freq=None,
                        cnco_freq=5e9,
                        fnco_freq=0.0,
                        sideband="U",
                        dump_config={"direction": "out"},
                        box_name="B0",
                        port=0,
                        channel=0,
                    )
                },
            ),
            repeats=1,
            interval=64.0,
        )
        modulation = multiplex.call_args.kwargs["modfreqs"]["R00"]
        carriers.append(modulation + 5.0)
        assert carriers[-1] == pytest.approx(frequency, abs=1e-12)
        if fail_acquisition:
            raise RuntimeError("acquisition failed")
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(
                    kerneled=np.exp(-1j * 2 * np.pi * modulation * 220.1)
                )
            }
        )

    ctx.measurement = SimpleNamespace(measure=measure)
    service = CharacterizationService(
        experiment_context=cast(Any, ctx),
        measurement_service=cast(Any, None),
        calibration_service=cast(Any, None),
        pulse_service=cast(
            Any,
            SimpleNamespace(
                readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
            ),
        ),
    )
    kwargs: dict[str, Any] = dict(
        f_start=5.01, df=0.0001, n_samples=4, plot=False, confirm=False
    )
    if fail_acquisition:
        with pytest.raises(RuntimeError, match="acquisition failed"):
            service.measure_electrical_delay("Q00", **kwargs)
    else:
        tau = service.measure_electrical_delay("Q00", **kwargs)
        assert tau == pytest.approx(220.1, abs=1e-7)
        assert len(carriers) >= 4
    assert ctx.current_frequency == 5.0
