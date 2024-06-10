from __future__ import annotations

import os
from typing import Literal, Optional

import httpx
import numpy as np
import numpy.typing as npt

from qubex.measurement_result import MeasureData, MeasureMode, MeasureResult
from qubex.pulse import Waveform

API_BASE_URL = "https://qiqb.ngrok.dev"


class PulseAPI:
    def __init__(
        self,
        *,
        chip_id: str,
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
        api_key: str
            The API key to use.
        api_base_url: str
            The base URL of the API.
        """
        self.chip_id = chip_id
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

    @property
    def targets(self) -> dict:
        """Get the available targets."""
        return self._request(
            "GET",
            "/api/targets",
            params={"chip_id": self.chip_id},
        )

    def measure(
        self,
        waveforms: dict[str, list | npt.NDArray | Waveform],
        *,
        frequencies: Optional[dict[str, float]] = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int = 1024,
        interval: int = 100 * 1024,
        control_window: int = 1024,
    ) -> MeasureResult:
        """
        Measure the waveforms.

        Parameters
        ----------
        waveforms: dict[str, list | npt.NDArray | Waveform]
            The waveforms to measure.
        frequencies: dict[str, float], optional
            The frequencies of the qubits.
        mode: Literal["single", "avg"], optional
            The measurement mode.
        shots: int, optional
            The number of shots.
        interval: int, optional
            The interval between measurements in ns.
        control_window: int, optional
            The control window in ns.

        Returns
        -------
        MeasureResult
            The measurement result.
        """

        normalized_waveforms = {}
        for qubit, waveform in waveforms.items():
            if isinstance(waveform, Waveform):
                waveform = waveform.values
            normalized_waveforms[qubit] = {
                "I": np.real(waveform).tolist(),
                "Q": np.imag(waveform).tolist(),
            }

        response = self._request(
            "POST",
            "/api/measure",
            json={
                "chip_id": self.chip_id,
                "waveforms": normalized_waveforms,
                "frequencies": frequencies,
                "mode": mode,
                "shots": shots,
                "interval": interval,
                "control_window": control_window,
            },
        )

        def to_ndarray(data):
            return np.array(data["I"]) + 1j * np.array(data["Q"])

        measure_data = {
            qubit: MeasureData(
                raw=to_ndarray(data["raw"]),
                kerneled=to_ndarray(data["kerneled"]),
                classified=np.array([]),
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
