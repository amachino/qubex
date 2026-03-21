"""Converters between canonical and legacy measurement result models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .classifiers.state_classifier import StateClassifier
from .models.capture_data import CaptureData
from .models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult


def _as_read_only_array(data: Any) -> np.ndarray:
    """Return read-only NumPy array view for capture payloads."""
    array = np.asarray(data).view()
    array.setflags(write=False)
    return array


class MeasurementResultConverter:
    """Convert `MeasurementResult` to and from legacy result classes."""

    @staticmethod
    def _resolve_measure_mode(config: MeasurementConfig) -> MeasureMode:
        """Resolve legacy measure mode from canonical measurement config."""
        return MeasureMode.AVG if config.shot_averaging else MeasureMode.SINGLE

    @staticmethod
    def _resolve_capture_mode(capture: CaptureData) -> MeasureMode:
        """Resolve legacy measure mode from capture-level config."""
        return MeasureMode.AVG if capture.config.shot_averaging else MeasureMode.SINGLE

    @staticmethod
    def from_multiple(
        multiple: MultipleMeasureResult,
        *,
        measurement_config: MeasurementConfig,
    ) -> MeasurementResult:
        """
        Create a canonical result from legacy multi-capture data.

        Parameters
        ----------
        multiple : MultipleMeasureResult
            Legacy multi-capture result.
        measurement_config : MeasurementConfig
            Measurement configuration attached to the canonical result.

        Returns
        -------
        MeasurementResult
            Canonical serializable measurement result.
        """
        data = {}
        for target, captures in multiple.data.items():
            data[target] = [
                CaptureData.from_primary_data(
                    target=target,
                    data=_as_read_only_array(item.raw),
                    config=measurement_config,
                    sampling_period=item.sampling_period,
                )
                for item in captures
            ]
        return MeasurementResult(
            data=data,
            measurement_config=measurement_config,
            device_config=multiple.config,
        )

    @staticmethod
    def to_multiple_measure_result(
        result: MeasurementResult,
        *,
        config: dict[str, Any] | None = None,
        classifiers: Mapping[str, StateClassifier] | None = None,
        sampling_period: float | None = None,
    ) -> MultipleMeasureResult:
        """
        Convert a canonical result to the legacy multi-capture model.

        Parameters
        ----------
        result : MeasurementResult
            Canonical result.
        config : dict[str, Any] | None, optional
            Legacy config payload. Falls back to `result.device_config` or `{}`.
        classifiers : dict[str, StateClassifier] | None, optional
            Optional legacy classifiers keyed by target.

        Returns
        -------
        MultipleMeasureResult
            Legacy multi-capture result.
        """
        mode = MeasurementResultConverter._resolve_measure_mode(
            result.measurement_config
        )
        if config is None:
            resolved_config: dict[str, Any] = (
                {} if result.device_config is None else result.device_config
            )
        else:
            resolved_config = config
        classifier_map = {} if classifiers is None else classifiers
        resolved_classifiers: dict[str, StateClassifier | None] = {
            target: classifier_map.get(target) for target in result.data
        }
        legacy_data: dict[str, list[MeasureData]] = {}
        for target, captures in result.data.items():
            legacy_captures = [
                MeasureData(
                    target=target,
                    mode=mode,
                    raw=np.asarray(capture.data),
                    classifier=resolved_classifiers.get(target),
                    sampling_period=(
                        sampling_period
                        if sampling_period is not None
                        else capture.sampling_period
                    ),
                )
                for capture in captures
            ]
            legacy_data[target] = legacy_captures
        for target, captures in result.data.items():
            for capture in captures:
                capture_mode = MeasurementResultConverter._resolve_capture_mode(capture)
                if capture_mode != mode:
                    raise ValueError(
                        f"Capture mode mismatch for target {target}: "
                        f"result mode={mode.value} capture mode={capture_mode.value}."
                    )
        return MultipleMeasureResult(
            mode=mode,
            data=legacy_data,
            config=resolved_config,
        )

    @staticmethod
    def to_measure_result(
        result: MeasurementResult,
        *,
        index: int = 0,
        config: dict[str, Any] | None = None,
        classifiers: Mapping[str, StateClassifier] | None = None,
        sampling_period: float | None = None,
    ) -> MeasureResult:
        """
        Convert one capture index to a legacy per-target result.

        Parameters
        ----------
        result : MeasurementResult
            Canonical result.
        index : int, optional
            Capture index in each target's result list.
        config : dict[str, Any] | None, optional
            Legacy config payload. Falls back to `result.device_config` or `{}`.
        classifiers : dict[str, StateClassifier] | None, optional
            Optional legacy classifiers keyed by target.

        Returns
        -------
        MeasureResult
            Legacy per-target result.

        Raises
        ------
        IndexError
            If `index` is out of range for any target.
        """
        classifier_map = {} if classifiers is None else classifiers
        resolved_classifiers: dict[str, StateClassifier | None] = {
            target: classifier_map.get(target) for target in result.data
        }
        single_data: dict[str, MeasureData] = {}
        resolved_mode: MeasureMode | None = None
        for target, captures in result.data.items():
            if not (-len(captures) <= index < len(captures)):
                raise IndexError(
                    f"Capture index {index} is out of range for target {target}."
                )
            selected_capture = captures[index]
            selected_mode = MeasurementResultConverter._resolve_capture_mode(
                selected_capture
            )
            if resolved_mode is None:
                resolved_mode = selected_mode
            elif resolved_mode != selected_mode:
                raise ValueError(
                    "Cannot convert captures with mixed shot_averaging modes "
                    "to legacy MeasureResult."
                )
            single_data[target] = MeasureData(
                target=target,
                mode=selected_mode,
                raw=np.asarray(selected_capture.data),
                classifier=resolved_classifiers.get(target),
                sampling_period=(
                    sampling_period
                    if sampling_period is not None
                    else selected_capture.sampling_period
                ),
            )

        if config is None:
            resolved_config: dict[str, Any] = (
                {} if result.device_config is None else result.device_config
            )
        else:
            resolved_config = config

        return MeasureResult(
            mode=(
                resolved_mode
                if resolved_mode is not None
                else MeasurementResultConverter._resolve_measure_mode(
                    result.measurement_config
                )
            ),
            data=single_data,
            config=resolved_config,
        )
