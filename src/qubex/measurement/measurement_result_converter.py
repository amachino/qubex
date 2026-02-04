"""Converters between canonical and legacy measurement result models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .classifiers.state_classifier import StateClassifier
from .models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from .models.measurement_result import MeasurementResult


class MeasurementResultConverter:
    """Convert `MeasurementResult` to and from legacy result classes."""

    @staticmethod
    def from_multiple(multiple: MultipleMeasureResult) -> MeasurementResult:
        """
        Create a canonical result from legacy multi-capture data.

        Parameters
        ----------
        multiple : MultipleMeasureResult
            Legacy multi-capture result.

        Returns
        -------
        MeasurementResult
            Canonical serializable measurement result.
        """
        data = {
            target: [np.asarray(item.raw) for item in captures]
            for target, captures in multiple.data.items()
        }
        return MeasurementResult(
            mode=multiple.mode.value,
            data=data,
            device_config=multiple.config,
        )

    @staticmethod
    def to_multiple_measure_result(
        result: MeasurementResult,
        *,
        config: dict[str, Any] | None = None,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> MultipleMeasureResult:
        """
        Convert a canonical result to the legacy multi-capture model.

        Parameters
        ----------
        result : MeasurementResult
            Canonical result.
        config : dict[str, Any] | None, optional
            Legacy config payload. Falls back to `result.device_config`.
        classifiers : dict[str, StateClassifier] | None, optional
            Optional legacy classifiers keyed by target.

        Returns
        -------
        MultipleMeasureResult
            Legacy multi-capture result.
        """
        mode = MeasureMode(result.mode)
        resolved_config: dict[str, Any] = (
            result.device_config if config is None else config
        )
        classifier_map = {} if classifiers is None else classifiers
        legacy_data = {
            target: [
                MeasureData(
                    target=target,
                    mode=mode,
                    raw=np.asarray(raw),
                    classifier=classifier_map.get(target),
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
            Legacy config payload. Falls back to `result.device_config`.
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
        mode = MeasureMode(result.mode)
        classifier_map = {} if classifiers is None else classifiers
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
            )

        return MeasureResult(
            mode=mode,
            data=single_data,
            config=result.device_config if config is None else config,
        )
