from __future__ import annotations

import os
import re
from functools import cached_property
from typing import Any, Literal, Optional

import httpx
import numpy as np

from qubex.measurement import MeasureData, MeasureMode, MeasureResult
from qubex.pulse import PulseSchedule, Waveform

API_BASE_URL = "https://qiqb.ngrok.dev"


class PulseAPI:
    def __init__(
        self,
        *,
        chip_id: str,
        qubits: list[str],
        api_key: str | None = None,
        api_base_url: str | None = None,
    ):
        """
        Pulse API Client
        API documentation: https://qiqb.ngrok.dev/docs

        Parameters
        ----------
        chip_id: str
            The quantum chip ID.
        qubits: list[str]
            The qubits to measure.
        api_key: str
            The API key to use.
        api_base_url: str
            The base URL of the API.
        """
        self.chip_id = chip_id
        self.qubits = qubits
        self.api_key = self._get_api_key(api_key)
        self.api_base_url = api_base_url or API_BASE_URL
        self.headers = {"X-API-Key": self.api_key}
        self.client = httpx.Client()

    @staticmethod
    def _get_api_key(api_key: str | None) -> str:
        """Get the API key."""
        if api_key is None:
            api_key = os.getenv("PULSE_API_KEY")
        if api_key is None:
            raise ValueError("API key is required.")
        return api_key

    @cached_property
    def params(self) -> dict:
        """Get parameters of the control system."""
        result = self._request(
            "GET",
            "/api/params",
            params={"chip_id": self.chip_id},
        )
        return result

    @cached_property
    def targets(self) -> dict:
        """Get the available targets."""
        result = self._request(
            "GET",
            "/api/targets",
            params={"chip_id": self.chip_id},
        )
        targets = {}
        for label, target in result.items():
            if target["qubit"] in self.qubits:
                if target["type"] == "CTRL_CR":
                    if match := re.match(r"^(Q\d+)-(Q\d+)$", label):
                        cr_target_qubit = match.group(2)
                        if cr_target_qubit in self.qubits:
                            targets[label] = target
                else:
                    targets[label] = target
        return targets

    def reset(self) -> dict:
        """Restart the control system."""
        result = self._request(
            "GET",
            "/api/reset",
            params={"chip_id": self.chip_id},
        )
        return result

    def measure(
        self,
        waveforms: dict[str, Any] | PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int = 1024,
        interval: int = 100 * 1024,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_frequencies: dict[str, float] | None = None,
        readout_waveforms: dict[str, Any] | None = None,
    ) -> MeasureResult:
        """
        Measure the qubits using the control waveforms.

        Parameters
        ----------
        waveforms: dict[str, Any] | PulseSchedule
            The control waveforms for each qubit.
        frequencies: dict[str, float], optional
            The frequencies of the qubits.
        mode: Literal["single", "avg"], optional
            The measurement mode.
        shots: int, optional
            The number of shots.
        interval: int, optional
            The interval between measurements in ns.
        control_window : int, optional
            The control window in ns, by default None.
        capture_window : int, optional
            The capture window in ns, by default None.
        capture_margin : int, optional
            The capture margin in ns, by default None.
        readout_duration : int, optional
            The readout duration in ns, by default None.
        readout_amplitudes : dict[str, float], optional
            The readout amplitude for each qubit, by default None.
        readout_frequencies : dict[str, float], optional
            The readout frequency for each qubit, by default None.
        readout_waveforms : dict[str, Any], optional
            The readout waveforms for each qubit, by default None.

        Returns
        -------
        MeasureResult
            The measurement result.
        """
        if isinstance(waveforms, PulseSchedule):
            waveforms = waveforms.get_sampled_sequences()

        if not all(label in self.targets for label in waveforms):
            raise ValueError(
                f"Invalid qubit labels: {set(waveforms) - set(self.targets)}"
            )

        control_waveforms = {}
        for qubit, waveform in waveforms.items():
            if isinstance(waveform, Waveform):
                waveform = waveform.values
            control_waveforms[qubit] = {
                "I": np.real(waveform).tolist(),
                "Q": np.imag(waveform).tolist(),
            }

        if readout_waveforms is not None:
            readout_waveforms_iq = {}
            for qubit, waveform in readout_waveforms.items():
                if isinstance(waveform, Waveform):
                    waveform = waveform.values
                readout_waveforms_iq[qubit] = {
                    "I": np.real(waveform).tolist(),
                    "Q": np.imag(waveform).tolist(),
                }
        else:
            readout_waveforms_iq = None

        response = self._request(
            "POST",
            "/api/measure",
            json={
                "chip_id": self.chip_id,
                "waveforms": control_waveforms,
                "frequencies": frequencies,
                "mode": mode,
                "shots": shots,
                "interval": interval,
                "control_window": control_window,
                "capture_window": capture_window,
                "capture_margin": capture_margin,
                "readout_duration": readout_duration,
                "readout_amplitudes": readout_amplitudes,
                "readout_frequencies": readout_frequencies,
                "readout_waveforms": readout_waveforms_iq,
            },
        )

        def to_ndarray(data):
            return np.array(data["I"]) + 1j * np.array(data["Q"])

        measure_data = {
            qubit: MeasureData(
                target=qubit,
                mode=MeasureMode(response["mode"]),
                raw=to_ndarray(data["raw"]),
                kerneled=to_ndarray(data["kerneled"]),
                classified=data["classified"],
            )
            for qubit, data in response["data"].items()
        }
        return MeasureResult(
            mode=MeasureMode(response["mode"]),
            data=measure_data,
            config=response["config"],
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        params=None,
        json=None,
    ) -> dict:
        """Make an HTTP request to the API."""
        response = self.client.request(
            method=method,
            url=self.api_base_url + endpoint,
            params=params,
            json=json,
            headers=self.headers,
            timeout=60,
        )
        status_code = response.status_code
        if status_code == 401:
            print("Unauthorized. Please check your API key.")
            return {"error": {"message": "Unauthorized"}}
        response.raise_for_status()
        json = response.json()
        if "error" in json:
            raise RuntimeError(json["error"]["message"])
        return json["result"]
