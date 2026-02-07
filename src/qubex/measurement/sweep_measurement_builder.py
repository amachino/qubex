"""Builder for sweep measurement pulse schedules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from inspect import signature
from typing import Any

import numpy as np
import tunits
from qxcore import Expression
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
from qxschema import (
    ParameterSweepConfig,
    ParametricSequencePulseCommand,
    SweepMeasurementConfig,
)


@dataclass(frozen=True)
class SweepState:
    """
    Resolved sweep state for a single sweep index.

    Attributes
    ----------
    sweep_indices : tuple of int
        Indices for each sweep axis.
    sequence_variables : dict of str to float
        Resolved sequence variable values keyed by variable name.
    frequency_shifts : dict of str to float
        Resolved frequency shifts keyed by channel label.
    """

    sweep_indices: tuple[int, ...]
    sequence_variables: dict[str, float]
    frequency_shifts: dict[str, float]


class SweepMeasurementBuilder:
    """
    Build pulse schedules from sweep measurement configurations.

    Parameters
    ----------
    config : SweepMeasurementConfig
        Sweep measurement configuration.
    command_registry : Mapping[str, Any] or None, optional
        Optional registry to override or extend pulse command factories.
    """

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

        delta_time = self._to_float(self._config.sequence.delta_time)
        set_sampling_period(delta_time)

        channels = self._create_channels(state.frequency_shifts)
        schedule = PulseSchedule(channels)
        self._build_schedule(schedule, state.sequence_variables)
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

    def _build_schedule(
        self,
        schedule: PulseSchedule,
        sequence_variables: Mapping[str, float],
    ) -> None:
        """
        Build the pulse schedule by processing commands.

        Parameters
        ----------
        schedule : PulseSchedule
            Schedule to be populated.
        sequence_variables : Mapping[str, float]
            Resolved variable values for the sweep point.

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
                self._add_blank(schedule, command, sequence_variables)
                continue

            factory = self._command_registry.get(name) or self._command_registry.get(
                name.lower()
            )
            if factory is None:
                raise ValueError(f"Unknown pulse command: {name}")

            args = [
                self._resolve_argument(arg, sequence_variables)
                for arg in command.argument_list
            ]
            obj = self._instantiate(factory, args)

            if isinstance(obj, PulseSchedule):
                schedule.call(obj, copy=False)
            elif isinstance(obj, (Waveform, PhaseShift)):
                for channel in command.channel_list:
                    schedule.add(channel, obj)
            else:
                raise TypeError(
                    f"Unsupported command object type for {name}: {type(obj)}"
                )

        # Finalize the schedule with a barrier
        schedule.barrier()

    def _add_barrier(
        self,
        schedule: PulseSchedule,
        command: ParametricSequencePulseCommand,
    ) -> None:
        """
        Insert a barrier into the schedule.

        Parameters
        ----------
        schedule : PulseSchedule
            Schedule to modify.
        command : ParametricSequencePulseCommand
            Barrier command definition.
        """
        labels = command.channel_list or None
        schedule.barrier(labels=labels)

    def _add_blank(
        self,
        schedule: PulseSchedule,
        command: ParametricSequencePulseCommand,
        sequence_variables: Mapping[str, float],
    ) -> None:
        """
        Insert blank (delay) pulses into the schedule.

        Parameters
        ----------
        schedule : PulseSchedule
            Schedule to modify.
        command : ParametricSequencePulseCommand
            Blank command definition.
        sequence_variables : Mapping[str, float]
            Resolved variable values for the sweep point.

        Raises
        ------
        ValueError
            If the command does not have exactly one argument.
        """
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
        """
        Resolve a command argument to a float.

        Parameters
        ----------
        arg : str or float
            Argument value or variable name.
        sequence_variables : Mapping[str, float]
            Resolved variable values.

        Returns
        -------
        float
            Resolved numeric value.

        Raises
        ------
        ValueError
            If a string argument is not a variable name or numeric literal.
        """
        # NOTE: When the argument is a string, it could be a literal parameter
        if isinstance(arg, str):
            # Try to resolve as a variable name
            if arg in sequence_variables:
                return sequence_variables[arg]
            try:
                return float(arg)
            except ValueError:
                pass
            # Try to resolve as a symbolic expression
            try:
                return float(Expression(arg).resolve(sequence_variables))
            except Exception as exc:
                raise ValueError(f"Failed to resolve argument: {arg}") from exc
        return float(arg)

    @staticmethod
    def _instantiate(factory: Any, args: Sequence[float]) -> Any:
        """
        Instantiate a factory using positional arguments by signature.

        Parameters
        ----------
        factory : Any
            Callable factory for a pulse object.
        args : Sequence[float]
            Positional arguments to map to the factory signature.

        Returns
        -------
        Any
            Instantiated object.

        Raises
        ------
        ValueError
            If more arguments are supplied than the factory accepts.
        """
        params = list(signature(factory).parameters.values())
        params = [param for param in params if param.name != "self"]
        if len(args) > len(params):
            raise ValueError(
                f"Too many arguments for {factory}: {len(args)} > {len(params)}"
            )
        kwargs = {params[idx].name: args[idx] for idx in range(len(args))}
        return factory(**kwargs)

    @staticmethod
    def _is_barrier(name: str) -> bool:
        """
        Check whether a command name represents a barrier.

        Parameters
        ----------
        name : str
            Command name.

        Returns
        -------
        bool
            True if the command is a barrier.
        """
        return name.lower() == "barrier"

    @staticmethod
    def _is_blank(name: str) -> bool:
        """
        Check whether a command name represents a blank/delay.

        Parameters
        ----------
        name : str
            Command name.

        Returns
        -------
        bool
            True if the command is a blank/delay.
        """
        return name.lower() in {"blank", "delay", "wait"}

    @staticmethod
    def _to_float(value: Any) -> float:
        """
        Convert supported numeric types to float.

        Parameters
        ----------
        value : Any
            Numeric-like value, including tunits and NumPy scalars.

        Returns
        -------
        float
            Converted float value.
        """
        if isinstance(value, tunits.Value):
            return float(value.value)
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)

    @staticmethod
    def _value_at_index(values: Any, index: int) -> float:
        """
        Get a float value from a list/array at the given index.

        Parameters
        ----------
        values : Any
            Sequence or array-like values (including tunits.ValueArray).
        index : int
            Index to retrieve.

        Returns
        -------
        float
            Value at the specified index.
        """
        if isinstance(values, tunits.ValueArray):
            values = values.value
        arr = np.asarray(values)
        return float(arr[index])

    @staticmethod
    def _compute_sweep_shape(sweep: ParameterSweepConfig) -> tuple[int, ...]:
        """
        Compute the sweep grid shape from configuration.

        Parameters
        ----------
        sweep : ParameterSweepConfig
            Sweep parameter configuration.

        Returns
        -------
        tuple of int
            Length of each sweep axis.

        Raises
        ------
        ValueError
            If an axis is empty, a sweep key is unknown, or axis lengths mismatch.
        """
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

    @staticmethod
    def _create_command_registry(
        overrides: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Create a command registry with optional overrides.

        Parameters
        ----------
        overrides : Mapping[str, Any] or None
            Registry overrides to merge.

        Returns
        -------
        dict of str to Any
            Command registry mapping names to factories.
        """
        registry: dict[str, Any] = {
            "Blank": Blank,
            "Rect": Rect,
            "Gaussian": Gaussian,
            "FlatTop": FlatTop,
            "Drag": Drag,
            "RaisedCosine": RaisedCosine,
            "VirtualZ": VirtualZ,
        }
        lower = {name.lower(): pulse for name, pulse in registry.items()}
        registry.update(lower)
        if overrides:
            registry.update(overrides)
        return registry
