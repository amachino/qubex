"""Pulse factory for measurement readout and pump waveforms."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from qxpulse import Blank, FlatTop, PulseArray, RampType

from qubex.system import ControlParameters, Mux, TargetRegistry

from .measurement_defaults import (
    DEFAULT_READOUT_DRAG_COEFF,
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_READOUT_RAMP_TIME,
    DEFAULT_READOUT_RAMP_TYPE,
)


class MeasurementPulseFactory:
    """Build readout and pump pulses from measurement control parameters."""

    def __init__(
        self,
        *,
        control_params: ControlParameters,
        mux_dict: Mapping[str, Mux],
        target_registry: TargetRegistry | None = None,
    ) -> None:
        """
        Initialize a measurement pulse factory.

        Parameters
        ----------
        control_params : ControlParameters
            Control parameters providing readout and pump amplitudes.
        mux_dict : Mapping[str, Mux]
            Mapping from qubit label to mux metadata.
        """
        self._control_params = control_params
        self._mux_dict = mux_dict
        self._target_registry = target_registry or TargetRegistry()

    def _resolve_qubit_label(self, target: str) -> str:
        """Resolve qubit label using target registry (legacy fallback enabled)."""
        resolver = self._target_registry.resolve_qubit_label
        try:
            return str(resolver(target, allow_legacy=True))
        except TypeError:
            return str(resolver(target))

    @staticmethod
    def _warn_deprecated_alias(
        *,
        old_name: str,
        new_name: str,
    ) -> None:
        """Emit a deprecation warning for an old option name."""
        warnings.warn(
            f"`{old_name}` is deprecated; use `{new_name}`.",
            DeprecationWarning,
            stacklevel=3,
        )

    @classmethod
    def _resolve_deprecated_alias(
        cls,
        *,
        new_value: Any,
        old_value: Any,
        old_name: str,
        new_name: str,
    ) -> Any:
        """Resolve old/new alias values and reject conflicting inputs."""
        if old_value is None:
            return new_value
        cls._warn_deprecated_alias(old_name=old_name, new_name=new_name)
        if new_value is not None and new_value != old_value:
            raise ValueError(
                f"`{old_name}` conflicts with `{new_name}`. Provide only `{new_name}`."
            )
        return old_value if new_value is None else new_value

    def readout_pulse(
        self,
        target: str,
        *,
        duration: float | None = None,
        amplitude: float | None = None,
        pre_margin: float | None = None,
        post_margin: float | None = None,
        ramp_time: float | None = None,
        ramp_type: RampType | None = None,
        drag_coeff: float | None = None,
        **deprecated_options: Any,
    ) -> PulseArray:
        """
        Build a readout pulse for a target.

        Parameters
        ----------
        target : str
            Target label.
        duration : float | None, optional
            Readout duration in ns.
        amplitude : float | None, optional
            Readout amplitude.
        pre_margin : float | None, optional
            Pre-readout margin.
        post_margin : float | None, optional
            Post-readout margin.
        ramp_time : float | None, optional
            Ramp time for the envelope.
        ramp_type : RampType | None, optional
            Ramp type name.
        drag_coeff : float | None, optional
            DRAG coefficient.

        Returns
        -------
        PulseArray
            Readout pulse array with margins.
        """
        legacy_ramp_time = deprecated_options.pop("ramptime", None)
        legacy_ramp_type = deprecated_options.pop("type", None)
        if deprecated_options:
            unexpected_args = ", ".join(sorted(deprecated_options))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected_args}.")
        ramp_time = self._resolve_deprecated_alias(
            new_value=ramp_time,
            old_value=legacy_ramp_time,
            old_name="ramptime",
            new_name="ramp_time",
        )
        ramp_type = self._resolve_deprecated_alias(
            new_value=ramp_type,
            old_value=legacy_ramp_type,
            old_name="type",
            new_name="ramp_type",
        )

        qubit = self._resolve_qubit_label(target)
        if duration is None:
            duration = DEFAULT_READOUT_DURATION
        if amplitude is None:
            amplitude = self._control_params.get_readout_amplitude(qubit)
        if pre_margin is None:
            pre_margin = DEFAULT_READOUT_PRE_MARGIN
        if post_margin is None:
            post_margin = DEFAULT_READOUT_POST_MARGIN
        if ramp_time is None:
            ramp_time = DEFAULT_READOUT_RAMP_TIME
        if ramp_type is None:
            ramp_type = DEFAULT_READOUT_RAMP_TYPE
        if drag_coeff is None:
            drag_coeff = DEFAULT_READOUT_DRAG_COEFF
        pulse = FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=ramp_time,
            beta=drag_coeff,
            type=ramp_type,
        )
        return PulseArray(
            [
                Blank(pre_margin),
                pulse.padded(
                    total_duration=duration + post_margin,
                    pad_side="right",
                ),
            ]
        )

    def pump_pulse(
        self,
        mux_index: int,
        duration: float | None = None,
        amplitude: float | None = None,
        ramp_time: float | None = None,
        ramp_type: RampType | None = None,
        **deprecated_options: Any,
    ) -> FlatTop:
        """
        Build a pump pulse for a mux.

        Parameters
        ----------
        mux_index : int
            Mux index.
        duration : float | None, optional
            Pump duration in ns.
        amplitude : float | None, optional
            Pump amplitude.
        ramp_time : float | None, optional
            Ramp time for the envelope.
        ramp_type : RampType | None, optional
            Ramp type name.

        Returns
        -------
        FlatTop
            Pump pulse.
        """
        legacy_ramp_time = deprecated_options.pop("ramptime", None)
        legacy_ramp_type = deprecated_options.pop("type", None)
        if deprecated_options:
            unexpected_args = ", ".join(sorted(deprecated_options))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected_args}.")
        ramp_time = self._resolve_deprecated_alias(
            new_value=ramp_time,
            old_value=legacy_ramp_time,
            old_name="ramptime",
            new_name="ramp_time",
        )
        ramp_type = self._resolve_deprecated_alias(
            new_value=ramp_type,
            old_value=legacy_ramp_type,
            old_name="type",
            new_name="ramp_type",
        )

        if duration is None:
            duration = DEFAULT_READOUT_DURATION
        if amplitude is None:
            amplitude = self._control_params.get_pump_amplitude(mux_index)
        if ramp_time is None:
            ramp_time = DEFAULT_READOUT_RAMP_TIME
        if ramp_type is None:
            ramp_type = DEFAULT_READOUT_RAMP_TYPE
        return FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=ramp_time,
            type=ramp_type,
        )
