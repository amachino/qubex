# ruff: noqa: SLF001

"""Tests for QuEL-1 continuous-wave controller operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


@dataclass
class _FakeSystemConfigDatabase:
    box_names: tuple[str, ...] = ("A",)

    def asdict(self) -> dict[str, dict[str, dict[str, object]]]:
        """Return minimal box settings for runtime-context validation."""
        return {"box_settings": {name: {} for name in self.box_names}}


@dataclass
class _FakeQubeCalib:
    system_config_database: _FakeSystemConfigDatabase = field(
        default_factory=_FakeSystemConfigDatabase
    )


class _FakeWavegenTask:
    def __init__(self) -> None:
        self.cancel_calls: list[dict[str, float]] = []
        self._done = False
        self._cancelled = False

    def done(self) -> bool:
        """Return whether task completed."""
        return self._done

    def cancelled(self) -> bool:
        """Return whether task was cancelled."""
        return self._cancelled

    def cancel(self, *, timeout: float, polling_period: float) -> bool:
        """Record cancellation arguments and mark the task cancelled."""
        self.cancel_calls.append({"timeout": timeout, "polling_period": polling_period})
        self._cancelled = True
        return True


class _FakeBox:
    def __init__(self) -> None:
        self.config_port_calls: list[dict[str, Any]] = []
        self.register_wavedata_calls: list[dict[str, Any]] = []
        self.config_channel_calls: list[dict[str, Any]] = []
        self.start_wavegen_calls: list[dict[str, Any]] = []
        self.tasks: list[_FakeWavegenTask] = []
        self.port_dump: dict[str, Any] = {
            "lo_freq": 9_000_000_000,
            "cnco_freq": 1_500_000_000,
            "sideband": "L",
            "vatt": 1234,
            "fullscale_current": 40500,
            "rfswitch": "block",
            "channels": {
                0: {"fnco_freq": 100_000_000},
                1: {"fnco_freq": 200_000_000},
            },
        }

    def config_port(self, **kwargs: Any) -> None:
        """Record port configuration."""
        self.config_port_calls.append(kwargs)

    def register_wavedata(self, **kwargs: Any) -> None:
        """Record registered wave data."""
        self.register_wavedata_calls.append(kwargs)

    def config_channel(self, **kwargs: Any) -> None:
        """Record channel configuration."""
        self.config_channel_calls.append(kwargs)

    def start_wavegen(
        self,
        channels: set[tuple[int, int]],
        **kwargs: Any,
    ) -> _FakeWavegenTask:
        """Record wavegen start and return a cancellable task."""
        task = _FakeWavegenTask()
        self.start_wavegen_calls.append({"channels": channels, **kwargs})
        self.tasks.append(task)
        return task

    def dump_port(self, port: int) -> dict[str, Any]:
        """Return a fake port dump."""
        _ = port
        return self.port_dump


class _FakeBoxPool:
    def __init__(self, *, box: _FakeBox) -> None:
        self._boxes = {"A": (box, object())}


def _make_connected_controller(box: _FakeBox) -> Quel1BackendController:
    controller = Quel1BackendController()
    cast(Any, controller)._runtime_context._qubecalib = _FakeQubeCalib()
    controller._connection_manager.set_connected_state(
        boxpool=cast(Any, _FakeBoxPool(box=box)),
        quel1system=cast(Any, object()),
        cap_resource_map={},
        gen_resource_map={},
    )
    return controller


def test_start_continuous_wave_registers_repeated_chunk_and_starts_wavegen(
    caplog,
) -> None:
    """Given CW settings, start configures the AWG and starts wave generation."""
    caplog.set_level(logging.INFO)
    box = _FakeBox()
    controller = _make_connected_controller(box)

    config = controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
        awg_freq_hz=7_812_500.0 * 2,
        amplitude=0.5,
        phase_rad=0.25,
        lo_freq_hz=10_500_000_000,
        cnco_freq_hz=2_250_000_000,
        fnco_freq_hz=750_000_000,
        sideband="U",
        vatt=2048,
        fullscale_current=39000,
        rfswitch="pass",
        configure_port=True,
    )

    assert config.box_name == "A"
    assert config.port == 2
    assert config.channel == 0
    assert config.awg_freq_hz == pytest.approx(15_625_000.0)
    assert config.cycles_per_chunk == 2
    assert config.amplitude == 0.5
    assert config.lo_freq_hz == 10_500_000_000
    assert config.cnco_freq_hz == 2_250_000_000
    assert config.fnco_freq_hz == 750_000_000
    assert config.actual_output_freq_hz == pytest.approx(13_515_625_000.0)
    assert config.sideband == "U"
    assert config.vatt == 2048
    assert config.fullscale_current == 39000
    assert config.rfswitch == "pass"
    assert config.duration_s > 0
    assert "Continuous wave frequencies" in caplog.text
    assert "awg_freq_hz=15625000.0" in caplog.text
    assert "actual_output_freq_ghz=13.515625" in caplog.text
    assert box.config_port_calls == [
        {
            "port": 2,
            "lo_freq": 10_500_000_000,
            "cnco_freq": 2_250_000_000,
            "sideband": "U",
            "vatt": 2048,
            "fullscale_current": 39000,
            "rfswitch": "pass",
        }
    ]
    assert len(box.register_wavedata_calls) == 1
    wavedata_call = box.register_wavedata_calls[0]
    assert wavedata_call["port"] == 2
    assert wavedata_call["channel"] == 0
    assert wavedata_call["name"] == config.waveform_name
    assert wavedata_call["allow_update"] is True
    iq = wavedata_call["iq"]
    assert isinstance(iq, np.ndarray)
    assert iq.shape == (64,)
    assert iq.dtype == np.complex64

    awg_param = box.config_channel_calls[0]["awg_param"]
    assert box.config_channel_calls[0]["port"] == 2
    assert box.config_channel_calls[0]["channel"] == 0
    assert box.config_channel_calls[0]["fnco_freq"] == 750_000_000
    assert awg_param.num_wait_word == 0
    assert awg_param.num_repeat == 0xFFFF_FFFF
    assert len(awg_param.chunks) == 1
    assert awg_param.chunks[0].name_of_wavedata == config.waveform_name
    assert awg_param.chunks[0].num_blank_word == 0
    assert awg_param.chunks[0].num_repeat == 0xFFFF_FFFF
    assert box.start_wavegen_calls == [
        {
            "channels": {(2, 0)},
            "disable_timeout": True,
            "return_after_start_emission": True,
        }
    ]


def test_start_continuous_wave_rejects_non_grid_frequency() -> None:
    """Given off-grid frequency, start raises ValueError before touching hardware."""
    box = _FakeBox()
    controller = _make_connected_controller(box)

    with pytest.raises(ValueError, match="integer multiple"):
        controller.start_continuous_wave(
            box_name="A",
            port=2,
            channel=0,
            awg_freq_hz=1_000_000.0,
        )

    assert box.register_wavedata_calls == []
    assert box.start_wavegen_calls == []


def test_start_continuous_wave_preserves_current_frequencies_by_default() -> None:
    """Given no frequency update flag, start does not reconfigure LO or NCOs."""
    box = _FakeBox()
    controller = _make_connected_controller(box)

    config = controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
    )

    assert config.awg_freq_hz == 0.0
    assert config.lo_freq_hz == 9_000_000_000
    assert config.cnco_freq_hz == 1_500_000_000
    assert config.fnco_freq_hz == 100_000_000
    assert config.actual_output_freq_hz == pytest.approx(7_400_000_000.0)
    assert config.sideband == "L"
    assert config.vatt == 1234
    assert config.fullscale_current == 40500
    assert config.rfswitch == "block"
    assert box.config_port_calls == []
    assert "fnco_freq" not in box.config_channel_calls[0]


def test_start_continuous_wave_allows_awg_freq_without_configure_port() -> None:
    """Given AWG frequency only, start updates waveform without configuring the port."""
    box = _FakeBox()
    controller = _make_connected_controller(box)

    config = controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
        awg_freq_hz=7_812_500.0 * 3,
    )

    assert config.awg_freq_hz == pytest.approx(23_437_500.0)
    assert config.cycles_per_chunk == 3
    assert config.actual_output_freq_hz == pytest.approx(7_376_562_500.0)
    assert box.config_port_calls == []
    assert "fnco_freq" not in box.config_channel_calls[0]


def test_start_continuous_wave_warns_when_fnco_plus_awg_exceeds_alias_limit(
    caplog,
) -> None:
    """Given high IF output, start logs a warning without failing."""
    caplog.set_level(logging.INFO)
    box = _FakeBox()
    controller = _make_connected_controller(box)

    config = controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
        awg_freq_hz=250_000_000.0,
        fnco_freq_hz=750_000_000,
        configure_port=True,
    )

    assert config.awg_freq_hz == pytest.approx(250_000_000.0)
    assert "exceeds 800000000.0 Hz" in caplog.text


def test_start_continuous_wave_requires_output_update_flag_for_settings() -> None:
    """Given output settings without update flag, start raises before hardware writes."""
    box = _FakeBox()
    controller = _make_connected_controller(box)

    with pytest.raises(ValueError, match="configure_port=True"):
        controller.start_continuous_wave(
            box_name="A",
            port=2,
            channel=0,
            awg_freq_hz=0.0,
            sideband="U",
        )

    assert box.config_port_calls == []
    assert box.register_wavedata_calls == []


def test_start_continuous_wave_rejects_duplicate_active_channel() -> None:
    """Given active CW on a channel, starting the same channel again raises."""
    box = _FakeBox()
    controller = _make_connected_controller(box)

    controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
        awg_freq_hz=0.0,
    )

    with pytest.raises(RuntimeError, match="already running"):
        controller.start_continuous_wave(
            box_name="A",
            port=2,
            channel=0,
            awg_freq_hz=0.0,
        )


def test_stop_continuous_wave_cancels_active_task_and_forgets_it() -> None:
    """Given active CW, stop cancels the wavegen task and clears state."""
    box = _FakeBox()
    controller = _make_connected_controller(box)
    controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
        awg_freq_hz=0.0,
    )
    task = box.tasks[0]

    assert (
        controller.stop_continuous_wave(
            box_name="A",
            port=2,
            channel=0,
            timeout=3.0,
            polling_period=0.02,
        )
        is True
    )

    assert task.cancel_calls == [{"timeout": 3.0, "polling_period": 0.02}]
    assert (
        controller.stop_continuous_wave(
            box_name="A",
            port=2,
            channel=0,
        )
        is False
    )


def test_stop_all_continuous_waves_cancels_every_active_task() -> None:
    """Given multiple active CW channels, stop all cancels each task."""
    box = _FakeBox()
    controller = _make_connected_controller(box)
    controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
        awg_freq_hz=0.0,
    )
    controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=1,
        awg_freq_hz=0.0,
    )

    controller.stop_all_continuous_waves(timeout=1.5, polling_period=0.05)

    assert [task.cancel_calls for task in box.tasks] == [
        [{"timeout": 1.5, "polling_period": 0.05}],
        [{"timeout": 1.5, "polling_period": 0.05}],
    ]


def test_disconnect_stops_active_continuous_waves() -> None:
    """Given active CW, disconnect cancels tasks before clearing backend state."""
    box = _FakeBox()
    controller = _make_connected_controller(box)
    controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
        awg_freq_hz=0.0,
    )
    task = box.tasks[0]

    controller.disconnect()

    assert task.cancel_calls == [{"timeout": 2.0, "polling_period": 0.01}]


def test_connect_rebuild_stops_active_continuous_waves() -> None:
    """Given active CW, reconnecting different boxes cancels remembered tasks."""
    box = _FakeBox()
    controller = _make_connected_controller(box)
    controller.start_continuous_wave(
        box_name="A",
        port=2,
        channel=0,
        awg_freq_hz=0.0,
    )
    task = box.tasks[0]
    connect_calls: list[dict[str, Any]] = []

    controller._connection_manager.requires_reconnect = (  # type: ignore[method-assign]
        lambda box_names: box_names == ["B"]
    )

    def _connect(*, box_names: str | list[str] | None, parallel: bool | None) -> None:
        connect_calls.append({"box_names": box_names, "parallel": parallel})

    controller._connection_manager.connect = _connect  # type: ignore[method-assign]

    controller.connect(box_names=["B"], parallel=False)

    assert task.cancel_calls == [{"timeout": 2.0, "polling_period": 0.01}]
    assert connect_calls == [{"box_names": ["B"], "parallel": False}]
