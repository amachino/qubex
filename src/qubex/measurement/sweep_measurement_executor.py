"""Executor for qxschema sweep measurement configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from qxschema import SweepMeasurementConfig, SweepMeasurementResult

from qubex.core import Value

from .measurement import Measurement
from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult
from .sweep_measurement_builder import SweepMeasurementBuilder


class SweepMeasurementExecutor:
    """Execute `qxschema.SweepMeasurementConfig` through the measurement runtime."""

    def __init__(
        self,
        *,
        measurement: Measurement,
        command_registry: Mapping[str, Any] | None = None,
    ) -> None:
        self._measurement = measurement
        self._command_registry = command_registry

    async def run(
        self,
        config: SweepMeasurementConfig,
    ) -> SweepMeasurementResult:
        """
        Execute the provided sweep configuration and return a qxschema result.

        Parameters
        ----------
        config : SweepMeasurementConfig
            External sweep measurement configuration.

        Returns
        -------
        SweepMeasurementResult
            Sweep result exported in qxschema format.
        """
        runtime_targets = self._validate_runtime_targets()
        self._validate_config_channels(config, runtime_targets)
        builder = SweepMeasurementBuilder(
            config=config,
            command_registry=self._command_registry,
        )
        measurement_config = self._create_measurement_config(config)

        results: list[MeasurementResult] = []
        for indices in builder.iterate():
            schedule = builder.build_measurement_schedule(indices)
            result = await self._measurement.run_measurement(
                schedule,
                config=measurement_config,
            )
            results.append(result)

        return self._convert_result(
            config=config,
            builder=builder,
            measurement_config=measurement_config,
            results=results,
        )

    def _validate_runtime_targets(self) -> set[str]:
        """Return the active runtime target names or raise when unavailable."""
        try:
            runtime_targets = set(self._measurement.targets)
            _ = self._measurement.backend_controller
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            raise RuntimeError(
                "Measurement runtime is not ready for sweep execution."
            ) from exc
        if len(runtime_targets) == 0:
            raise RuntimeError("Measurement runtime has no available targets.")
        return runtime_targets

    def _validate_config_channels(
        self,
        config: SweepMeasurementConfig,
        runtime_targets: set[str],
    ) -> None:
        """Validate that all config channel references exist in runtime targets."""
        referenced_channels = set(config.channel_list)
        referenced_channels.update(config.frequency.channel_to_frequency)
        referenced_channels.update(config.frequency.channel_to_frequency_reference)
        referenced_channels.update(config.frequency.channel_to_frequency_shift)
        referenced_channels.update(config.data_acquisition.channel_to_averaging_time)
        referenced_channels.update(config.data_acquisition.channel_to_averaging_window)
        for command in config.sequence.command_list:
            referenced_channels.update(command.channel_list)
        for sweep_content in config.sweep_parameter.sweep_content_list.values():
            if sweep_content.category == "frequency_shift":
                referenced_channels.update(sweep_content.sweep_target)

        unknown_channels = sorted(referenced_channels - runtime_targets)
        if unknown_channels:
            joined = ", ".join(unknown_channels)
            raise ValueError(f"Unknown channel(s) in sweep config: {joined}.")

    def _create_measurement_config(
        self,
        config: SweepMeasurementConfig,
    ) -> MeasurementConfig:
        """Create the shared runtime measurement configuration."""
        data_acquisition = config.data_acquisition
        return self._measurement.create_measurement_config(
            n_shots=int(data_acquisition.shot_count),
            shot_interval=self._to_float(data_acquisition.shot_repetition_margin),
            shot_averaging=bool(data_acquisition.flag_average_shots),
            time_integration=bool(data_acquisition.flag_average_waveform),
            state_classification=False,
        )

    def _convert_result(
        self,
        *,
        config: SweepMeasurementConfig,
        builder: SweepMeasurementBuilder,
        measurement_config: MeasurementConfig,
        results: list[MeasurementResult],
    ) -> SweepMeasurementResult:
        """Convert runtime measurement results into qxschema shape."""
        sweep_shape = builder.sweep_shape
        sweep_key_list = [
            key for axis in config.sweep_parameter.sweep_axis for key in axis
        ]
        data_key_list, data = self._stack_result_data(
            results=results,
            sweep_shape=sweep_shape,
        )
        metadata = {
            "measurement_config": measurement_config.to_dict(),
            "sampling_period": builder.sampling_period,
            "sweep_axis": [list(axis) for axis in config.sweep_parameter.sweep_axis],
            "channel_to_averaging_time": config.data_acquisition.channel_to_averaging_time,
            "channel_to_averaging_window": config.data_acquisition.channel_to_averaging_window,
            "channel_to_frequency_reference": config.frequency.channel_to_frequency_reference,
            "keep_oscillator_relative_phase": config.frequency.keep_oscillator_relative_phase,
            "data_axis": "after_sweep_axes",
        }
        return SweepMeasurementResult(
            metadata=metadata,
            data=data,
            data_shape=list(data.shape),
            sweep_key_list=sweep_key_list,
            data_key_list=data_key_list,
        )

    def _stack_result_data(
        self,
        *,
        results: list[MeasurementResult],
        sweep_shape: tuple[int, ...],
    ) -> tuple[list[str], np.ndarray]:
        """Stack runtime results into one dense tensor."""
        if len(results) == 0:
            empty = np.empty((*sweep_shape, 0), dtype=complex)
            return [], empty

        first = results[0]
        target_order = list(first.data)
        capture_counts = {target: len(first.data[target]) for target in target_order}
        data_key_list = [
            target if capture_counts[target] == 1 else f"{target}[{capture_index}]"
            for target in target_order
            for capture_index in range(capture_counts[target])
        ]
        per_key_series: dict[str, list[np.ndarray]] = {key: [] for key in data_key_list}
        payload_shape: tuple[int, ...] | None = None

        for point_index, result in enumerate(results):
            result_targets = list(result.data)
            if set(result_targets) != set(target_order):
                raise ValueError(
                    "All sweep points must contain the same targets for result "
                    f"conversion; mismatch at point index {point_index}."
                )
            for target in target_order:
                captures = result.data[target]
                expected_capture_count = capture_counts[target]
                if len(captures) != expected_capture_count:
                    raise ValueError(
                        "All sweep points must contain the same capture count for "
                        f"{target}; mismatch at point index {point_index}."
                    )
                for capture_index, capture in enumerate(captures):
                    key = (
                        target
                        if expected_capture_count == 1
                        else f"{target}[{capture_index}]"
                    )
                    array = np.asarray(capture.data)
                    if payload_shape is None:
                        payload_shape = array.shape
                    elif array.shape != payload_shape:
                        raise ValueError(
                            "All exported data keys must have the same payload "
                            f"shape; got {array.shape} and {payload_shape}."
                        )
                    per_key_series[key].append(array)

        if payload_shape is None:
            payload_shape = ()

        stacked_per_key = []
        for key in data_key_list:
            series = per_key_series[key]
            stacked = np.stack(series, axis=0).reshape((*sweep_shape, *payload_shape))
            stacked_per_key.append(stacked)

        if len(stacked_per_key) == 0:
            return data_key_list, np.empty((*sweep_shape, 0), dtype=complex)

        data = np.stack(stacked_per_key, axis=len(sweep_shape))
        return data_key_list, data

    @staticmethod
    def _to_float(value: Value | float | int | np.generic) -> float:
        """Convert sweep-related values to float."""
        if isinstance(value, Value):
            return float(value.value)
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)
