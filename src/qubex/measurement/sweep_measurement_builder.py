"""Builder for sweep measurement schedules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Protocol, TypeAlias

import numpy as np
from qxpulse import (
    Blank,
    Drag,
    FlatTop,
    Gaussian,
    PhaseShift,
    PulseChannel,
    PulseSchedule,
    RaisedCosine,
    Rect,
    VirtualZ,
    Waveform,
    set_sampling_period,
)

from qubex.core import Expression, Value, ValueArray
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.schema import (
    ParameterSweepConfig,
    ParametricSequencePulseCommand,
    SweepMeasurementConfig,
)

SweepCommandValue: TypeAlias = PulseSchedule | Waveform | PhaseShift


@dataclass(frozen=True)
class SweepState:
    """Resolved sweep state for a single sweep index."""

    sweep_indices: tuple[int, ...]
    sequence_variables: dict[str, float]
    frequency_shifts: dict[str, float]


@dataclass(frozen=True)
class SweepCommandContext:
    """Resolved context provided to custom command factories."""

    command: ParametricSequencePulseCommand
    resolved_argument_list: tuple[float, ...]
    sequence_variables: Mapping[str, float]
    frequency_shifts: Mapping[str, float]
    delta_time: float


class SweepCommandFactory(Protocol):
    """Protocol for sweep command factories."""

    def __call__(self, context: SweepCommandContext) -> SweepCommandValue:
        """Build a pulse object from a resolved command context."""
        ...


class SweepMeasurementBuilder:
    """
    Build measurement schedules from sweep measurement configurations.

    Parameters
    ----------
    config : SweepMeasurementConfig
        Sweep measurement configuration.
    command_registry : Mapping[str, Any] or None, optional
        Optional registry to override or extend pulse command factories.
    """

    _RESERVED_COMMANDS = frozenset({"barrier", "blank", "delay", "wait"})

    def __init__(
        self,
        *,
        config: SweepMeasurementConfig,
        command_registry: Mapping[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._command_registry = self._create_command_registry(command_registry)
        self._sweep_shape = self._compute_sweep_shape(config.sweep_parameter)

    @property
    def sweep_shape(self) -> tuple[int, ...]:
        """Tuple of int: Shape of the sweep grid across all axes."""
        return self._sweep_shape

    @property
    def sampling_period(self) -> float:
        """Return the configured sampling period in ns."""
        return self._to_float(self._config.sequence.delta_time)

    def iterate(self) -> Iterable[tuple[int, ...]]:
        """
        Iterate over all sweep indices of the sweep grid.

        Returns
        -------
        Iterable[tuple[int, ...]]
            Iterator over axis index tuples.
        """
        return np.ndindex(self._sweep_shape)

    def resolve_sweep_state(self, sweep_indices: Sequence[int]) -> SweepState:
        """
        Resolve sweep variables and frequency shifts for given indices.

        Parameters
        ----------
        sweep_indices : Sequence[int]
            Indices for each sweep axis.

        Returns
        -------
        SweepState
            Resolved variables and frequency shifts for the indices.

        Raises
        ------
        ValueError
            If axis length mismatches, sweep keys are unknown, or category is unsupported.
        """
        if len(sweep_indices) != len(self._config.sweep_parameter.sweep_axis):
            raise ValueError("sweep_indices length must match sweep_axis length.")

        sequence_variables: dict[str, float] = {}
        frequency_shifts: dict[str, float] = {}

        for idx, keys in enumerate(self._config.sweep_parameter.sweep_axis):
            for key in keys:
                sweep_content = self._config.sweep_parameter.sweep_content_list.get(key)
                if sweep_content is None:
                    raise ValueError(f"Unknown sweep key: {key}")

                value = self._value_at_index(
                    sweep_content.value_list,
                    sweep_indices[idx],
                )

                if sweep_content.category == "sequence_variable":
                    for target in sweep_content.sweep_target:
                        sequence_variables[target] = value
                elif sweep_content.category == "frequency_shift":
                    for target in sweep_content.sweep_target:
                        frequency_shifts[target] = value
                else:
                    raise ValueError(
                        f"Unsupported sweep category: {sweep_content.category}"
                    )

        return SweepState(
            sweep_indices=tuple(sweep_indices),
            sequence_variables=sequence_variables,
            frequency_shifts=frequency_shifts,
        )

    def build_schedule(self, indices: Sequence[int]) -> PulseSchedule:
        """
        Build a pulse schedule for the specified sweep indices.

        Parameters
        ----------
        indices : Sequence[int]
            Indices for each sweep axis.

        Returns
        -------
        PulseSchedule
            Schedule constructed with resolved variables and shifts.
        """
        state = self.resolve_sweep_state(indices)
        return self._build_pulse_schedule(state)

    def build_measurement_schedule(self, indices: Sequence[int]) -> MeasurementSchedule:
        """
        Build a measurement schedule for the specified sweep indices.

        Parameters
        ----------
        indices : Sequence[int]
            Indices for each sweep axis.

        Returns
        -------
        MeasurementSchedule
            Schedule constructed with resolved pulse and capture instructions.
        """
        state = self.resolve_sweep_state(indices)
        pulse_schedule = self._build_pulse_schedule(state)
        capture_schedule = self._build_capture_schedule(pulse_schedule)
        return MeasurementSchedule(
            pulse_schedule=pulse_schedule,
            capture_schedule=capture_schedule,
        )

    def _build_pulse_schedule(self, state: SweepState) -> PulseSchedule:
        """Build one pulse schedule from a resolved sweep state."""
        set_sampling_period(self.sampling_period)
        channels = self._create_channels(state.frequency_shifts)
        schedule = PulseSchedule(channels)
        self._populate_pulse_schedule(
            schedule=schedule,
            state=state,
        )
        return schedule

    def _create_channels(
        self,
        sweep_shifts: Mapping[str, float],
    ) -> list[PulseChannel]:
        """
        Create pulse channels with resolved frequencies.

        Parameters
        ----------
        sweep_shifts : Mapping[str, float]
            Frequency shifts keyed by channel label.

        Returns
        -------
        list of PulseChannel
            Channels with effective frequencies applied.
        """
        freq_config = self._config.frequency
        channels: list[PulseChannel] = []
        for channel in self._config.channel_list:
            base = freq_config.channel_to_frequency.get(channel)
            base_shift = freq_config.channel_to_frequency_shift.get(channel)
            sweep_shift = sweep_shifts.get(channel)
            parts = [
                self._to_float(value)
                for value in (base, base_shift, sweep_shift)
                if value is not None
            ]
            frequency = float(np.sum(parts)) if parts else None
            channels.append(PulseChannel(label=channel, frequency=frequency))
        return channels

    def _populate_pulse_schedule(
        self,
        *,
        schedule: PulseSchedule,
        state: SweepState,
    ) -> None:
        """
        Populate the pulse schedule by processing commands.

        Parameters
        ----------
        schedule : PulseSchedule
            Schedule to be populated.
        state : SweepState
            Resolved state for the current sweep point.

        Raises
        ------
        ValueError
            If a command is unknown or produces an unsupported object type.
        """
        for command in self._config.sequence.command_list:
            name = command.name

            if self._is_barrier(name):
                self._add_barrier(schedule, command)
                continue

            if self._is_blank(name):
                self._add_blank(schedule, command, state.sequence_variables)
                continue

            factory = self._lookup_command_factory(name)
            if factory is None:
                raise ValueError(f"Unknown pulse command: {name}")

            resolved_args = tuple(
                self._resolve_argument(arg, state.sequence_variables)
                for arg in command.argument_list
            )
            context = SweepCommandContext(
                command=command,
                resolved_argument_list=resolved_args,
                sequence_variables=dict(state.sequence_variables),
                frequency_shifts=dict(state.frequency_shifts),
                delta_time=self.sampling_period,
            )
            obj = self._invoke_command_factory(factory, context)

            if isinstance(obj, PulseSchedule):
                schedule.call(obj, copy=False)
            elif isinstance(obj, (Waveform, PhaseShift)):
                for channel in command.channel_list:
                    schedule.add(channel, obj)
            else:
                raise TypeError(
                    f"Unsupported command object type for {name}: {type(obj)}"
                )

        schedule.barrier()

    def _build_capture_schedule(self, pulse_schedule: PulseSchedule) -> CaptureSchedule:
        """Build capture schedule from acquisition settings and pulse ranges."""
        data_acquisition = self._config.data_acquisition
        capture_channel_set = set(data_acquisition.channel_to_averaging_window)
        averaging_time_channel_set = set(data_acquisition.channel_to_averaging_time)
        if len(capture_channel_set) == 0:
            raise ValueError(
                "data_acquisition must define at least one capture channel."
            )
        if capture_channel_set != averaging_time_channel_set:
            raise ValueError(
                "channel_to_averaging_window and channel_to_averaging_time must "
                "contain the same capture channels."
            )
        unknown_capture_channels = sorted(
            capture_channel_set - set(self._config.channel_list)
        )
        if unknown_capture_channels:
            joined = ", ".join(unknown_capture_channels)
            raise ValueError(
                f"Capture channels must be included in channel_list: {joined}."
            )

        capture_delay = self._to_float(data_acquisition.data_acquisition_delay)
        capture_duration = self._to_float(data_acquisition.data_acquisition_duration)
        if capture_duration <= 0:
            raise ValueError("data_acquisition_duration must be positive.")

        pulse_ranges = pulse_schedule.get_pulse_ranges(list(capture_channel_set))
        captures: list[Capture] = []
        for channel in data_acquisition.channel_to_averaging_window:
            ranges = pulse_ranges.get(channel, [])
            if len(ranges) == 0:
                raise ValueError(
                    f"No pulse ranges found for capture channel {channel}."
                )
            for pulse_range in ranges:
                start_time = pulse_range.start * self.sampling_period + capture_delay
                end_time = start_time + capture_duration
                if end_time > pulse_schedule.duration + 1e-9:
                    raise ValueError(
                        "Computed capture window exceeds pulse schedule duration for "
                        f"{channel}."
                    )
                captures.append(
                    Capture(
                        channels=[channel],
                        start_time=start_time,
                        duration=capture_duration,
                    )
                )
        return CaptureSchedule(captures=captures)

    def _lookup_command_factory(
        self,
        name: str,
    ) -> Callable[..., SweepCommandValue] | None:
        """Return command factory by normalized command name."""
        return self._command_registry.get(name.lower())

    def _add_barrier(
        self,
        schedule: PulseSchedule,
        command: ParametricSequencePulseCommand,
    ) -> None:
        """Insert a barrier into the schedule."""
        labels = command.channel_list or None
        schedule.barrier(labels=labels)

    def _add_blank(
        self,
        schedule: PulseSchedule,
        command: ParametricSequencePulseCommand,
        sequence_variables: Mapping[str, float],
    ) -> None:
        """Insert blank pulses into the schedule."""
        if len(command.argument_list) != 1:
            raise ValueError("Blank command expects a single duration argument.")
        duration = self._resolve_argument(command.argument_list[0], sequence_variables)
        for channel in command.channel_list:
            schedule.add(channel, Blank(duration=duration))

    @staticmethod
    def _resolve_argument(
        arg: str | float,
        sequence_variables: Mapping[str, float],
    ) -> float:
        """Resolve a command argument to a float."""
        if isinstance(arg, str):
            if arg in sequence_variables:
                return sequence_variables[arg]
            try:
                return float(arg)
            except ValueError:
                pass
            try:
                return float(Expression(arg).resolve(sequence_variables))
            except Exception as exc:
                raise ValueError(f"Failed to resolve argument: {arg}") from exc
        return float(arg)

    @staticmethod
    def _instantiate(factory: Any, args: Sequence[float]) -> Any:
        """Instantiate a legacy factory using positional argument mapping."""
        params = list(signature(factory).parameters.values())
        params = [param for param in params if param.name != "self"]
        if len(args) > len(params):
            raise ValueError(
                f"Too many arguments for {factory}: {len(args)} > {len(params)}"
            )
        kwargs = {params[idx].name: args[idx] for idx in range(len(args))}
        return factory(**kwargs)

    @classmethod
    def _invoke_command_factory(
        cls,
        factory: Callable[..., SweepCommandValue],
        context: SweepCommandContext,
    ) -> SweepCommandValue:
        """Invoke a command factory using context or legacy positional mode."""
        if cls._expects_context(factory):
            return factory(context)
        return cls._instantiate(factory, context.resolved_argument_list)

    @staticmethod
    def _expects_context(factory: Callable[..., SweepCommandValue]) -> bool:
        """Return whether the factory expects a single context argument."""
        params = [
            param
            for param in signature(factory).parameters.values()
            if param.name != "self"
        ]
        if len(params) != 1:
            return False
        param = params[0]
        return param.kind in (
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
        )

    @staticmethod
    def _is_barrier(name: str) -> bool:
        """Check whether a command name represents a barrier."""
        return name.lower() == "barrier"

    @staticmethod
    def _is_blank(name: str) -> bool:
        """Check whether a command name represents a blank/delay."""
        return name.lower() in {"blank", "delay", "wait"}

    @staticmethod
    def _to_float(value: Any) -> float:
        """Convert supported numeric types to float."""
        if isinstance(value, Value):
            return float(value.value)
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)

    @staticmethod
    def _value_at_index(values: Any, index: int) -> float:
        """Get a float value from a list or array at the given index."""
        if isinstance(values, ValueArray):
            values = values.value
        arr = np.asarray(values)
        return float(arr[index])

    @staticmethod
    def _compute_sweep_shape(sweep: ParameterSweepConfig) -> tuple[int, ...]:
        """Compute the sweep grid shape from configuration."""
        shape: list[int] = []
        for sweep_keys in sweep.sweep_axis:
            if not sweep_keys:
                raise ValueError("sweep_axis cannot contain empty axis.")
            first_key = sweep_keys[0]
            content = sweep.sweep_content_list.get(first_key)
            if content is None:
                raise ValueError(f"Unknown sweep key: {first_key}")
            axis_len = len(np.asarray(content.value_list))
            for sweep_key in sweep_keys[1:]:
                other = sweep.sweep_content_list.get(sweep_key)
                if other is None:
                    raise ValueError(f"Unknown sweep key: {sweep_key}")
                other_len = len(np.asarray(other.value_list))
                if other_len != axis_len:
                    raise ValueError(
                        "All sweep contents in the same axis must have the same length."
                    )
            shape.append(axis_len)
        return tuple(shape)

    @classmethod
    def _build_builtin_factory(
        cls,
        constructor: Callable[..., SweepCommandValue],
    ) -> Callable[[SweepCommandContext], SweepCommandValue]:
        """Wrap a legacy constructor as a context-based factory."""

        def _factory(context: SweepCommandContext) -> SweepCommandValue:
            return cls._instantiate(constructor, context.resolved_argument_list)

        return _factory

    @classmethod
    def _create_command_registry(
        cls,
        overrides: Mapping[str, Any] | None,
    ) -> dict[str, Callable[..., SweepCommandValue]]:
        """Create a command registry with optional overrides."""
        registry: dict[str, Callable[..., SweepCommandValue]] = {
            "rect": cls._build_builtin_factory(Rect),
            "gaussian": cls._build_builtin_factory(Gaussian),
            "flattop": cls._build_builtin_factory(FlatTop),
            "drag": cls._build_builtin_factory(Drag),
            "raisedcosine": cls._build_builtin_factory(RaisedCosine),
            "virtualz": cls._build_builtin_factory(VirtualZ),
        }
        if overrides is None:
            return registry

        normalized_keys: set[str] = set()
        for name in overrides:
            normalized = name.lower()
            if normalized in normalized_keys:
                raise ValueError(
                    "command_registry must not contain duplicate normalized command "
                    f"keys: {name}."
                )
            if normalized in cls._RESERVED_COMMANDS:
                raise ValueError(
                    f"command_registry cannot override reserved command: {name}."
                )
            normalized_keys.add(normalized)

        custom_registry = {name.lower(): factory for name, factory in overrides.items()}
        registry.update(custom_registry)
        return registry
