"""Tests for the Qblox backend server DC voltage driver."""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any

import pytest

from qubex.external_devices.dc_voltage import (
    DCVoltageController,
    DCVoltageProfile,
    ExternalDevicesConfig,
    create_dc_voltage_controller,
)
from qubex.external_devices.dc_voltage.drivers.qblox_server import (
    QbloxServerConnectionConfig,
    QbloxServerDevice,
)


class _FakeSocket:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = responses
        self.sent: list[bytes] = []
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, size: int) -> bytes:
        if not self.responses:
            return b""
        response = self.responses.pop(0)
        chunk = response[:size]
        remainder = response[size:]
        if remainder:
            self.responses.insert(0, remainder)
        return chunk

    def close(self) -> None:
        self.closed = True


def _socket_factory(fake_socket: _FakeSocket) -> Callable[..., _FakeSocket]:
    def create_connection(*_args: Any, **_kwargs: Any) -> _FakeSocket:
        return fake_socket

    return create_connection


def test_connection_config_parses_server_and_channel_mapping() -> None:
    """Qblox server config should parse endpoint and logical channel names."""
    config = QbloxServerConnectionConfig.from_dict(
        "QBLOX1",
        {
            "host": "dc-backend.example",
            "port": 12345,
            "timeout_s": 600,
            "device_names": {1: "Qblox-A-1", 2: "Qblox-A-2"},
        },
    )

    assert config == QbloxServerConnectionConfig(
        host="dc-backend.example",
        port=12345,
        timeout_s=600.0,
        channels={1: "Qblox-A-1", 2: "Qblox-A-2"},
    )


def test_qblox_server_reports_no_physical_output_switch() -> None:
    """Qblox server devices should report that physical switching is unavailable."""
    device = QbloxServerDevice(
        host="server",
        port=12345,
        channels={1: "Qblox1-1"},
        socket_factory=_socket_factory(_FakeSocket([])),
    )

    assert device.supports_output_switch is False


def test_connection_config_derives_channels_from_device_id() -> None:
    """Backend device names should derive from the device id and channels."""
    config = QbloxServerConnectionConfig.from_dict(
        "Qblox1",
        {
            "host": "dc-backend.example",
            "port": 12345,
        },
        device_channels=(15, 16),
    )

    assert config.channels == {15: "Qblox1-15", 16: "Qblox1-16"}


def test_connection_config_requires_channels_for_derived_names() -> None:
    """Deriving backend names requires a device `channels` list."""
    with pytest.raises(ValueError, match="channels"):
        QbloxServerConnectionConfig.from_dict(
            "Qblox1",
            {"host": "server", "port": 1},
        )


def test_connection_config_rejects_unknown_settings() -> None:
    """Unknown driver params should fail at parse time."""
    with pytest.raises(ValueError, match="Unknown Qblox server settings"):
        QbloxServerConnectionConfig.from_dict(
            "Qblox1",
            {"host": "server", "port": 1, "device_prefix": "Qblox1"},
            device_channels=(1,),
        )


@pytest.mark.parametrize(
    ("connection", "error", "message"),
    [
        ({"port": 1, "device_names": {1: "dev"}}, ValueError, "host"),
        ({"host": "server", "device_names": {1: "dev"}}, ValueError, "port"),
        (
            {"host": "server", "port": 1, "device_names": {"1": "dev"}},
            ValueError,
            "must be integers",
        ),
        (
            {"host": "server", "port": 1, "device_names": {1: "bad\x00name"}},
            ValueError,
            "NUL",
        ),
    ],
)
def test_connection_config_rejects_invalid_settings(
    connection: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Qblox server config should reject unsafe endpoint and channel settings."""
    with pytest.raises(error, match=message):
        QbloxServerConnectionConfig.from_dict("QBLOX1", connection)


def test_device_sends_existing_set_voltage_protocol() -> None:
    """Voltage setting should use command 0x62 and a little-endian double."""
    fake_socket = _FakeSocket([b"\x00"])
    device = QbloxServerDevice(
        host="server",
        port=12345,
        channels={1: "Qblox-A-1"},
        socket_factory=_socket_factory(fake_socket),
    )

    device.set_voltage(channel=1, voltage=-0.25)

    assert fake_socket.sent == [b"\x62Qblox-A-1\x00" + struct.pack("<d", -0.25)]


def test_device_reads_fragmented_voltage_response_exactly() -> None:
    """Voltage readback should assemble a fragmented 9-byte TCP response."""
    payload = struct.pack("<d", 0.375)
    fake_socket = _FakeSocket([b"\x00" + payload[:2], payload[2:5], payload[5:]])
    device = QbloxServerDevice(
        host="server",
        port=12345,
        channels={1: "Qblox-A-1"},
        socket_factory=_socket_factory(fake_socket),
    )

    voltage = device.get_voltage(channel=1)

    assert voltage == pytest.approx(0.375)
    assert fake_socket.sent == [b"\x63Qblox-A-1\x00"]


def test_device_rejects_get_voltage_error_without_waiting_for_payload() -> None:
    """A failed voltage read should reject its status-only response immediately."""
    fake_socket = _FakeSocket([b"\xff"])
    device = QbloxServerDevice(
        host="server",
        port=12345,
        channels={1: "Qblox-A-1"},
        socket_factory=_socket_factory(fake_socket),
    )

    with pytest.raises(RuntimeError, match="rejected"):
        device.get_voltage(channel=1)

    assert fake_socket.responses == []


def test_device_sends_existing_native_sweep_protocol() -> None:
    """Native ramp should use command 0x64 and the backend's five doubles."""
    fake_socket = _FakeSocket([b"\x00"])
    device = QbloxServerDevice(
        host="server",
        port=12345,
        channels={2: "Qblox-A-2"},
        socket_factory=_socket_factory(fake_socket),
    )

    device.ramp_voltage(
        channel=2,
        start_voltage=0.0,
        target_voltage=0.2,
        rate_v_per_s=0.1,
        step_size_v=0.01,
        wait_s=0.1,
    )

    assert fake_socket.sent == [
        b"\x64Qblox-A-2\x00" + struct.pack("<ddddd", 0.0, 0.2, 0.1, 0.01, 0.1)
    ]


def test_device_rejects_backend_error_status() -> None:
    """A nonzero backend status should raise an operation error."""
    device = QbloxServerDevice(
        host="server",
        port=12345,
        channels={1: "Qblox-A-1"},
        socket_factory=_socket_factory(_FakeSocket([b"\xff"])),
    )

    with pytest.raises(RuntimeError, match="rejected"):
        device.set_voltage(channel=1, voltage=0.1)


def test_controller_factory_resolves_qblox_server_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller factory should create the registered Qblox server driver."""
    fake_socket = _FakeSocket([b"\x00" + struct.pack("<d", 0.1)])
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.drivers.qblox_server.socket.create_connection",
        _socket_factory(fake_socket),
    )
    config = ExternalDevicesConfig.from_dict(
        {
            "devices": {
                "Qblox1": {
                    "driver": "qblox_server",
                    "params": {
                        "host": "server",
                        "port": 12345,
                    },
                    "channels": [15],
                },
            },
            "wiring": [{"mux": 8, "bias": "Qblox1-15"}],
        }
    ).dc_voltage

    controller = create_dc_voltage_controller(config)
    readings = controller.read_channels([15])

    assert fake_socket.closed is True
    assert fake_socket.sent == [b"\x63Qblox1-15\x00"]
    assert readings == {15: (0.1, True)}


def test_controller_delegates_complete_ramp_to_native_server() -> None:
    """A native-ramp device should receive one complete ramp request."""
    calls: list[tuple[Any, ...]] = []
    readbacks = iter([0.0, 0.2])

    class _NativeRampDevice:
        supports_native_ramp = True
        supports_output_switch = False

        def close(self) -> None:
            calls.append(("close",))

        def on(self, channel: int) -> None:
            raise AssertionError(channel)

        def off(self, channel: int) -> None:
            raise AssertionError(channel)

        def set_voltage(self, channel: int, voltage: float) -> None:
            raise AssertionError((channel, voltage))

        def get_voltage(self, channel: int) -> float:
            calls.append(("get", channel))
            return next(readbacks)

        def is_output_on(self, channel: int) -> bool:
            return True

        def ramp_voltage(
            self,
            channel: int,
            start_voltage: float,
            target_voltage: float,
            rate_v_per_s: float,
            step_size_v: float,
            wait_s: float,
        ) -> None:
            calls.append(
                (
                    "ramp",
                    channel,
                    start_voltage,
                    target_voltage,
                    rate_v_per_s,
                    step_size_v,
                    wait_s,
                )
            )

    controller = DCVoltageController(device_factory=_NativeRampDevice)
    controller.apply_voltage(
        channel=1,
        voltage=0.2,
        profile=DCVoltageProfile(
            channel=1,
            ramp_rate_v_per_s=0.1,
            ramp_step_size_v=0.01,
            ramp_wait_s=0.1,
        ),
    )

    assert calls[0] == ("get", 1)
    assert calls[1][0:2] == ("ramp", 1)
    assert calls[1][2:] == pytest.approx((0.0, 0.2, 0.1, 0.01, 0.1))
    assert calls[2:] == [("get", 1), ("close",)]
