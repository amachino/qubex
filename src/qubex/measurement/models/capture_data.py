"""Per-capture measurement data model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

from qubex.core import DataModel

from .classifier_ref import ClassifierRef
from .measurement_config import MeasurementConfig, ReturnItem

WaveformSeries: TypeAlias = NDArray[np.complexfloating[Any, Any]]
IQSeries: TypeAlias = NDArray[np.complexfloating[Any, Any]]
StateSeries: TypeAlias = NDArray[np.integer[Any]]
AveragedWaveform: TypeAlias = NDArray[np.complexfloating[Any, Any]]
AveragedIQ: TypeAlias = NDArray[np.complexfloating[Any, Any]]


def _format_array_preview(raw: NDArray) -> str:
    """Return compact repr text for a data array."""
    array = np.asarray(raw)
    if array.ndim == 0:
        return repr(array)
    flat = array.reshape(-1)
    preview = np.array2string(flat[:1], separator=", ")
    if preview.startswith("[") and preview.endswith("]") and flat.size > 1:
        preview = f"{preview[:-1]}, ...]"
    return f"array({preview}, shape={array.shape})"


def _normalize_payload_array(
    *,
    item: ReturnItem,
    data: NDArray[Any],
    n_shots: int,
) -> NDArray[Any]:
    """Return one payload array in canonical shape for its return-item kind."""
    array = np.asarray(data)

    match item:
        case ReturnItem.WAVEFORM_SERIES:
            if array.ndim == 0:
                raise ValueError("waveform_series must include a shot axis.")
            if array.ndim == 1:
                if n_shots != 1:
                    raise ValueError(
                        "waveform_series must include one waveform per shot."
                    )
                return array.reshape(1, -1)
            if array.shape[0] != n_shots:
                raise ValueError(
                    "waveform_series shot axis length must match config.n_shots."
                )
            return array.reshape(n_shots, -1)

        case ReturnItem.IQ_SERIES:
            squeezed = np.squeeze(array)
            if squeezed.ndim == 0:
                if n_shots != 1:
                    raise ValueError(
                        "iq_series must include one integrated IQ value per shot."
                    )
                squeezed = squeezed.reshape(1)
            if squeezed.ndim != 1:
                raise ValueError(
                    "iq_series must contain exactly one integrated IQ value per shot."
                )
            if squeezed.shape[0] != n_shots:
                raise ValueError("iq_series length must match config.n_shots.")
            return squeezed

        case ReturnItem.STATE_SERIES:
            squeezed = np.squeeze(array)
            if squeezed.ndim == 0:
                if n_shots != 1:
                    raise ValueError(
                        "state_series must include one classified state per shot."
                    )
                squeezed = squeezed.reshape(1)
            if squeezed.ndim != 1:
                raise ValueError(
                    "state_series must contain exactly one classified state per shot."
                )
            if squeezed.shape[0] != n_shots:
                raise ValueError("state_series length must match config.n_shots.")
            return squeezed

        case ReturnItem.AVERAGED_WAVEFORM:
            squeezed = np.squeeze(array)
            if squeezed.ndim == 0:
                return squeezed.reshape(1)
            return squeezed.reshape(-1)

        case ReturnItem.AVERAGED_IQ:
            squeezed = np.squeeze(array)
            if squeezed.ndim == 0:
                return squeezed
            if squeezed.size != 1:
                raise ValueError("averaged_iq must contain exactly one complex value.")
            return squeezed.reshape(())


class CapturePayload(DataModel):
    """Structured payload fields for capture data arrays."""

    waveform_series: WaveformSeries | None = None
    iq_series: IQSeries | None = None
    state_series: StateSeries | None = None
    averaged_waveform: AveragedWaveform | None = None
    averaged_iq: AveragedIQ | None = None


class CaptureData(DataModel):
    """
    Serializable per-capture measurement payload.

    Parameters
    ----------
    target : str
        Target label associated with this capture (for example, qubit label).
    config : MeasurementConfig
        Measurement configuration used to interpret payload fields.
        The primary payload field is selected by `config.primary_return_item`.
    payload : CapturePayload
        Data payload model that may contain one or more of:
        `waveform_series`, `iq_series`, `state_series`, `averaged_waveform`,
        and `averaged_iq`.
    sampling_period : float
        Sampling period in ns for waveform-domain visualization and derived axes.
        Must be positive.
    classifier_ref : ClassifierRef | None, default=None
        Optional classifier reference metadata for downstream classification.

    Notes
    -----
    - This model stores device-returned payloads and metadata only.
    - At least one entry must be present in `payload`.
    - Payload fields must be consistent with `config.return_items`.
    - Payload arrays are normalized to canonical shapes at model construction:
      `waveform_series -> (n_shots, capture_length)`,
      `iq_series -> (n_shots,)`,
      `state_series -> (n_shots,)`,
      `averaged_waveform -> (capture_length,)`,
      `averaged_iq -> ()`.
    - The `data` property returns the primary payload selected by
      `config.primary_return_item`.
    """

    target: str
    config: MeasurementConfig
    payload: CapturePayload
    sampling_period: float
    classifier_ref: ClassifierRef | None = None

    def __repr__(self) -> str:
        """Return a compact representation for notebook-friendly display."""
        classifier_ref = (
            "None" if self.classifier_ref is None else repr(self.classifier_ref)
        )
        return (
            "CaptureData("
            f"target={self.target!r}, "
            f"data={_format_array_preview(self.data)}, "
            f"sampling_period={self.sampling_period}, "
            f"classifier_ref={classifier_ref})"
        )

    @classmethod
    def from_primary_data(
        cls,
        *,
        target: str,
        data: NDArray,
        config: MeasurementConfig,
        sampling_period: float,
        classifier_ref: ClassifierRef | None = None,
    ) -> CaptureData:
        """Create capture data by placing data into the config-primary field."""
        primary_item = config.primary_return_item
        match primary_item:
            case ReturnItem.WAVEFORM_SERIES:
                payload = CapturePayload(waveform_series=data)
            case ReturnItem.IQ_SERIES:
                payload = CapturePayload(iq_series=data)
            case ReturnItem.STATE_SERIES:
                payload = CapturePayload(state_series=data)
            case ReturnItem.AVERAGED_WAVEFORM:
                payload = CapturePayload(averaged_waveform=data)
            case ReturnItem.AVERAGED_IQ:
                payload = CapturePayload(averaged_iq=data)
        return cls(
            target=target,
            config=config,
            payload=payload,
            classifier_ref=classifier_ref,
            sampling_period=sampling_period,
        )

    @model_validator(mode="after")
    def _validate_invariants(self) -> CaptureData:
        """Validate sampling period, payload consistency, and shape constraints."""
        if self.sampling_period <= 0:
            raise ValueError("sampling_period must be positive.")

        configured_items = set(self.config.return_items)
        payload_items: set[ReturnItem] = set()
        if self.payload.waveform_series is not None:
            payload_items.add(ReturnItem.WAVEFORM_SERIES)
        if self.payload.iq_series is not None:
            payload_items.add(ReturnItem.IQ_SERIES)
        if self.payload.state_series is not None:
            payload_items.add(ReturnItem.STATE_SERIES)
        if self.payload.averaged_waveform is not None:
            payload_items.add(ReturnItem.AVERAGED_WAVEFORM)
        if self.payload.averaged_iq is not None:
            payload_items.add(ReturnItem.AVERAGED_IQ)
        if not payload_items:
            raise ValueError("At least one capture payload field must be set.")

        config = self.config
        unexpected_items = sorted(
            payload_items - configured_items,
            key=lambda item: item.value,
        )
        if unexpected_items:
            joined = ", ".join(item.value for item in unexpected_items)
            raise ValueError(
                "Capture payload contains fields not requested by "
                f"config.return_items: {joined}."
            )

        normalized_payload = self.payload
        for item, field_name in (
            (ReturnItem.WAVEFORM_SERIES, "waveform_series"),
            (ReturnItem.IQ_SERIES, "iq_series"),
            (ReturnItem.STATE_SERIES, "state_series"),
            (ReturnItem.AVERAGED_WAVEFORM, "averaged_waveform"),
            (ReturnItem.AVERAGED_IQ, "averaged_iq"),
        ):
            value = getattr(normalized_payload, field_name)
            if value is None:
                continue
            object.__setattr__(
                normalized_payload,
                field_name,
                _normalize_payload_array(
                    item=item,
                    data=value,
                    n_shots=self.config.n_shots,
                ),
            )

        data = np.asarray(self.data)
        if config.shot_averaging:
            return self
        if data.ndim == 0:
            raise ValueError(
                "data must have at least one dimension when shot_averaging is disabled."
            )
        if config.time_integration and data.shape[0] != config.n_shots:
            raise ValueError(
                "data first-axis length must match config.n_shots "
                "when time_integration is enabled and shot_averaging is disabled."
            )
        if not config.time_integration and data.shape[0] != config.n_shots:
            raise ValueError(
                "data shot-axis length must match config.n_shots "
                "when waveform shots are retained."
            )
        state_series = self.payload.state_series
        if state_series is not None and np.asarray(state_series).ndim >= 1:
            if np.asarray(state_series).shape[0] != config.n_shots:
                raise ValueError(
                    "state_series first-axis length must match config.n_shots "
                    "when shot_averaging is disabled."
                )
        return self

    @property
    def data(self) -> NDArray:
        """Return the primary capture payload selected by config mode."""
        primary_field = self.config.primary_return_item
        match primary_field:
            case ReturnItem.WAVEFORM_SERIES:
                value = self.payload.waveform_series
            case ReturnItem.IQ_SERIES:
                value = self.payload.iq_series
            case ReturnItem.STATE_SERIES:
                value = self.payload.state_series
            case ReturnItem.AVERAGED_WAVEFORM:
                value = self.payload.averaged_waveform
            case ReturnItem.AVERAGED_IQ:
                value = self.payload.averaged_iq
        if value is None:
            raise ValueError(
                f"Primary payload field `{primary_field.value}` is not set."
            )
        return value

    @property
    def waveform_series(self) -> WaveformSeries | None:
        """Return non-averaged waveform payload."""
        return self.payload.waveform_series

    @property
    def iq_series(self) -> IQSeries | None:
        """Return non-averaged integrated IQ payload."""
        return self.payload.iq_series

    @property
    def state_series(self) -> StateSeries | None:
        """Return per-shot classified state payload."""
        return self.payload.state_series

    @property
    def averaged_waveform(self) -> AveragedWaveform | None:
        """Return shot-averaged waveform payload."""
        return self.payload.averaged_waveform

    @property
    def averaged_iq(self) -> AveragedIQ | None:
        """Return shot-averaged integrated IQ payload."""
        return self.payload.averaged_iq

    def save(
        self,
        path: str | Path,
    ) -> Path:
        """Alias of `save_netcdf`."""
        return self.save_netcdf(path)

    @classmethod
    def load(cls, path: str | Path) -> CaptureData:
        """Alias of `load_netcdf`."""
        return cls.load_netcdf(path)
