from __future__ import annotations

from typing import Literal, Optional

import httpx
import numpy as np
import numpy.typing as npt
from qubex.measurement import MeasureData, MeasureMode, MeasureResult
from qubex.pulse import Waveform


class PulseAPI:
    def __init__(
        self,
        api_key: str,
        api_base_url: str,
    ):
        """
        Pulse API Client
        API documentation: https://qiqb.ngrok.dev/docs

        Parameters
        ----------
        api_key: str
            The API key to use.
        api_base_url: str
            The base URL of the API.
        """
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.headers = {"X-API-Key": self.api_key}
        self.client = httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def config(self) -> dict:
        """Get the current configuration."""
        return await self._request(
            "GET",
            "/api/config",
        )

    async def configure(self, chip_id: str) -> dict:
        """Configure the device."""
        return await self._request(
            "POST",
            "/api/configure",
            json={"chip_id": chip_id},
        )

    async def measure(
        self,
        waveforms: dict[str, list | npt.NDArray | Waveform],
        *,
        frequencies: Optional[dict[str, float]] = None,
        meas_mode: Literal["single", "avg"] = "avg",
        meas_level: int = 1,
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
        meas_mode: Literal["single", "avg"], optional
            The measurement mode.
        meas_level: int, optional
            The measurement level.
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

        response = await self._request(
            "POST",
            "/api/measure",
            json={
                "waveforms": normalized_waveforms,
                "frequencies": frequencies,
                "meas_mode": meas_mode,
                "meas_level": meas_level,
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

    async def _request(
        self,
        method: str,
        endpoint: str,
        json=None,
    ):
        """Make an HTTP request to the API."""
        response = await self.client.request(
            method=method,
            url=self.api_base_url + endpoint,
            json=json,
            headers=self.headers,
            timeout=60,
        )
        response.raise_for_status()
        json = response.json()
        if "error" in json:
            raise RuntimeError(json["error"]["message"])
        return json["result"]

    async def close(self):
        """Close the HTTP client session."""
        await self.client.aclose()
