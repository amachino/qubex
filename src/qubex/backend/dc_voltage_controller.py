from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Final

from ..third_party.ons61797 import ONS61797

PORT: Final = "/dev/ttyACM0"


@contextmanager
def dc_voltage(voltages: dict[int, float]):
    try:
        ons61797 = ONS61797(port=PORT)
        original_voltages = {}
        for channel, voltage in voltages.items():
            original_voltages[channel] = ons61797.get_voltage(channel)
            ons61797.set_voltage(channel, voltage)
            ons61797.on(channel)
        yield ons61797
    finally:
        for channel, voltage in original_voltages.items():
            ons61797.set_voltage(channel, voltage)
            ons61797.off(channel)
            ons61797.close()


def with_connection(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            if self.ons61797 is None:
                self.ons61797 = ONS61797(port=PORT)
            else:
                self.ons61797.connect(port=PORT)
            return func(self, *args, **kwargs)
        finally:
            if self.ons61797 is not None:
                self.ons61797.close()

    return wrapper


class DCVoltageController:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.ons61797: ONS61797 = None  # type: ignore
        self._initialized = True

    def __del__(self):
        if self.ons61797 is not None:
            self.ons61797.close()

    @with_connection
    def on(self, channel: int) -> None:
        self.ons61797.on(channel=channel)

    @with_connection
    def off(self, channel: int) -> None:
        self.ons61797.off(channel=channel)

    @with_connection
    def get_output_state(self, channel: int) -> int:
        return self.ons61797.get_output_state(channel=channel)

    @with_connection
    def set_voltage(self, channel: int, voltage: float) -> None:
        self.ons61797.set_voltage(channel=channel, voltage=voltage)

    @with_connection
    def get_voltage(self, channel: int) -> float:
        return self.ons61797.get_voltage(channel=channel)

    @with_connection
    def get_device_information(self) -> str:
        return self.ons61797.get_device_information()

    @with_connection
    def reset(self) -> None:
        self.ons61797.reset()

    @contextmanager
    def connection(self):
        try:
            if self.ons61797 is None:
                self.ons61797 = ONS61797(port=PORT)
            else:
                self.ons61797.connect(port=PORT)
            yield self.ons61797
        finally:
            if self.ons61797 is not None:
                self.ons61797.close()
