"""
Copyright (c) 2024 NF Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import socket
import time
from typing import Optional

import serial


class ONS61797(object):
    """
    A class to interface with ONS61797 instruments via serial or socket communication.

    Parameters
    ----------
    port : str, optional
        The serial port to connect to the instrument, by default None.
    ip_address : str, optional
        The IP address to connect to the instrument, by default None.

    Attributes
    ----------
    instrument : serial.Serial or socket.socket
        The instrument connection object.
    port : str
        The serial port used for communication.
    baudrate : int
        The baudrate for serial communication, default is 115200.
    ip_address : str
        The IP address for socket communication.
    line_feed_code : str
        The line feed character used in communication.
    time_out : float
        Timeout value for communication, default is 10.0 seconds.
    time_interval : float
        The time interval between sending a command and reading a response, default is 0.1 seconds.
    """

    def __init__(self, port: Optional[str] = None, ip_address: Optional[str] = None):
        """
        Initializes the connection to the instrument.

        Parameters
        ----------
        port : str, optional
            The serial port to connect to the instrument, by default None.
        ip_address : str, optional
            The IP address to connect to the instrument, by default None.
        """
        self.instrument: Optional[socket.socket | serial.Serial] = None
        self.port = port
        self.baudrate = 115200
        self.ip_address = ip_address
        self.line_feed_code = "\n"
        self.time_out = 10.0
        self.time_interval = 0.1
        self.connect(port=self.port, ip_address=self.ip_address)

    def __del__(self) -> None:
        """Closes the connection to the instrument when the object is deleted."""
        if self.instrument:
            self.close()

    def connect(
        self, port: Optional[str] = None, ip_address: Optional[str] = None
    ) -> None:
        """
        Establishes a connection to the instrument via serial or socket.

        Parameters
        ----------
        port : str, optional
            The serial port to connect to the instrument, by default None.
        ip_address : str, optional
            The IP address to connect to the instrument, by default None.

        Raises
        ------
        ValueError
            If both port and ip_address are provided or neither is provided.
        """
        if self.instrument:
            self.close()

        self.port = port
        self.ip_address = ip_address

        if not port and not ip_address:
            raise ValueError("Either port or ip_address must be provided.")
        if port and ip_address:
            raise TypeError("Only one of port or ip_address should be provided.")

        if port:
            self.kind = "serial"
            self.instrument = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.time_out
            )
        elif ip_address:
            self.kind = "socket"
            self.instrument = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.instrument.connect((self.ip_address, 10001))

    def close(self) -> None:
        """Closes the connection to the instrument."""
        if self.instrument:
            self.instrument.close()
            self.instrument = None

    def write(self, cmd: str) -> None:
        """
        Sends a command to the instrument.

        Parameters
        ----------
        cmd : str
            The command to send to the instrument.
        """
        cmd = f"{cmd}{self.line_feed_code}"
        if self.kind == "serial":
            if isinstance(self.instrument, serial.Serial):
                self.instrument.write(cmd.encode("utf-8"))
            # self.instrument.write(cmd.encode("utf-8"))
        elif self.kind == "socket":
            if isinstance(self.instrument, socket.socket):
                self.instrument.sendall(cmd.encode("utf-8"))
            # self.instrument.sendall(cmd.encode("utf-8"))
        time.sleep(self.time_interval)  # コマンドの書き込み後、遅延を追加

    def read(self) -> str:
        """
        Reads a response from the instrument.

        Returns
        -------
        str
            The response from the instrument.
        """
        if self.kind == "serial":
            if isinstance(self.instrument, serial.Serial):
                res = self.instrument.read_until(
                    self.line_feed_code.encode("utf-8")
                ).decode("utf-8")[:-1]
                time.sleep(self.time_interval)  # コマンドの書き込み後、遅延を追加
                return res
            # return self.instrument.read_until(
            #     self.line_feed_code.encode("utf-8")
            # ).decode("utf-8")[:-1]
        elif self.kind == "socket":
            if isinstance(self.instrument, socket.socket):
                res = ""
                while True:
                    res_ = self.instrument.recv(1).decode("utf-8")
                    if res_ == self.line_feed_code:
                        break
                    res += res_
                time.sleep(self.time_interval)  # コマンドの書き込み後、遅延を追加
                return res
        raise ValueError("Invalid instrument connection.")

    def query(self, cmd: str) -> str:
        """
        Sends a command and reads the response from the instrument.

        Parameters
        ----------
        cmd : str
            The command to send.

        Returns
        -------
        str
            The response from the instrument.
        """
        self.write(cmd)
        time.sleep(self.time_interval)
        return self.read()

    def on(self, channel: int) -> None:
        """
        Turns on the specified output channel.

        Parameters
        ----------
        channel : int
            The output channel number.
        """
        cmd = f"OUT {channel},1"
        self.write(cmd=cmd)

    def off(self, channel: int) -> None:
        """
        Turns off the specified output channel.

        Parameters
        ----------
        channel : int
            The output channel number.
        """
        cmd = f"OUT {channel},0"
        self.write(cmd=cmd)

    def get_output_state(self, channel: int) -> int:
        """
        Gets the current state of the specified output channel.

        Parameters
        ----------
        channel : int
            The output channel number.

        Returns
        -------
        int
            The state of the output channel (0: off, 1: on).
        """
        cmd = f"OUT? {channel}"
        return int(self.query(cmd=cmd))

    def set_voltage(self, channel: int, voltage: float) -> None:
        """
        Sets the output voltage for the specified channel.

        Parameters
        ----------
        channel : int
            The output channel number.
        voltage : float
            The voltage to set.
        """
        cmd = f"VLT {channel},{voltage:.4f}"
        self.write(cmd=cmd)

    def get_voltage(self, channel: int) -> float:
        """
        Gets the current output voltage for the specified channel.

        Parameters
        ----------
        channel : int
            The output channel number.

        Returns
        -------
        float
            The output voltage.
        """
        cmd = f"VLT? {channel}"
        return float(self.query(cmd=cmd))

    def get_device_information(self) -> str:
        """
        Retrieves the device information.

        Returns
        -------
        str
            The device information string.
        """
        cmd = "*IDN?"
        return self.query(cmd=cmd)

    def reset(self) -> None:
        """Resets the instrument to its default settings."""
        cmd = "*RST"
        self.write(cmd=cmd)

    def set_ip_address(self, ip_address: str) -> None:
        """
        Sets the instrument's IP address.

        Parameters
        ----------
        ip_address : str
            The new IP address.
        """
        cmd = f"IPA {ip_address}"
        self.write(cmd=cmd)

    def get_ip_address(self) -> str:
        """
        Gets the current IP address of the instrument.

        Returns
        -------
        str
            The IP address.
        """
        cmd = "IPA?"
        return str(self.query(cmd=cmd))

    def set_subnet_mask(self, subnet_mask: str) -> None:
        """
        Sets the subnet mask for the instrument.

        Parameters
        ----------
        subnet_mask : str
            The new subnet mask.
        """
        cmd = f"SBM {subnet_mask}"
        self.write(cmd=cmd)

    def get_subnet_mask(self) -> str:
        """
        Gets the current subnet mask of the instrument.

        Returns
        -------
        str
            The subnet mask.
        """
        cmd = "SBM?"
        return str(self.query(cmd=cmd))

    def set_default_gateway(self, default_gateway: str) -> None:
        """
        Sets the default gateway for the instrument.

        Parameters
        ----------
        default_gateway : str
            The new default gateway.
        """
        cmd = f"DGW {default_gateway}"
        self.write(cmd=cmd)

    def get_default_gateway(self) -> str:
        """
        Gets the current default gateway of the instrument.

        Returns
        -------
        str
            The default gateway.
        """
        cmd = "DGW?"
        return str(self.query(cmd=cmd))
