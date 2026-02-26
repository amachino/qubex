"""Converters between canonical and legacy measurement result models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from qubex.backend.quel1 import SAMPLING_PERIOD

from .classifiers.state_classifier import StateClassifier
from .models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult


class MeasurementResultConverter:
    """Convert `MeasurementResult` to and from legacy result classes."""

    @staticmethod
    def _resolve_measure_mode(config: MeasurementConfig) -> MeasureMode:
        """Resolve legacy measure mode from canonical measurement config."""
        return MeasureMode.AVG if config.shot_averaging else MeasureMode.SINGLE

    @staticmethod
    def _resolve_sampling_period_ns(
        *,
        explicit: float | None,
        result: MeasurementResult,
    ) -> float:
        """Resolve sampling period for legacy result models."""
        if explicit is not None:
            return explicit
        if result.sampling_period_ns is not None:
            return float(result.sampling_period_ns)
        return SAMPLING_PERIOD

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
        data = {
            target: [np.asarray(item.raw) for item in captures]
            for target, captures in multiple.data.items()
        }
        first_data = next(iter(multiple.data.values()), [])
        first_capture = first_data[0] if first_data else None
        return MeasurementResult(
            data=data,
            measurement_config=measurement_config,
            device_config=multiple.config,
            sampling_period_ns=(
                first_capture.sampling_period_ns
                if isinstance(first_capture, MeasureData)
                else None
            ),
        )

    @staticmethod
    def to_multiple_measure_result(
        result: MeasurementResult,
        *,
        config: dict[str, Any] | None = None,
        classifiers: Mapping[str, StateClassifier] | None = None,
        sampling_period_ns: float | None = None,
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
        resolved_sampling_period = (
            MeasurementResultConverter._resolve_sampling_period_ns(
                explicit=sampling_period_ns,
                result=result,
            )
        )
        legacy_data = {
            target: [
                MeasureData(
                    target=target,
                    mode=mode,
                    raw=np.asarray(raw),
                    classifier=classifier_map.get(target),
                    sampling_period_ns=resolved_sampling_period,
                )
                for raw in captures
            ]
            for target, captures in result.data.items()
        }
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
        sampling_period_ns: float | None = None,
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
        mode = MeasurementResultConverter._resolve_measure_mode(
            result.measurement_config
        )
        classifier_map = {} if classifiers is None else classifiers
        resolved_sampling_period = (
            MeasurementResultConverter._resolve_sampling_period_ns(
                explicit=sampling_period_ns,
                result=result,
            )
        )
        single_data: dict[str, MeasureData] = {}
        for target, captures in result.data.items():
            if not (-len(captures) <= index < len(captures)):
                raise IndexError(
                    f"Capture index {index} is out of range for target {target}."
                )
            single_data[target] = MeasureData(
                target=target,
                mode=mode,
                raw=np.asarray(captures[index]),
                classifier=classifier_map.get(target),
                sampling_period_ns=resolved_sampling_period,
            )

        if config is None:
            resolved_config: dict[str, Any] = (
                {} if result.device_config is None else result.device_config
            )
        else:
            resolved_config = config

        return MeasureResult(
            mode=mode,
            data=single_data,
            config=resolved_config,
        )
