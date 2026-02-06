"""Pulse factory for measurement readout and pump waveforms."""

from __future__ import annotations

from collections.abc import Mapping

from qxpulse import Blank, FlatTop, PulseArray, RampType

from qubex.backend import ControlParams, Mux, Target

from .measurement_defaults import (
    DEFAULT_READOUT_DRAG_COEFF,
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_READOUT_RAMP_TYPE,
    DEFAULT_READOUT_RAMPTIME,
)


class MeasurementPulseFactory:
    """Build readout and pump pulses from measurement control parameters."""

    def __init__(
        self,
        *,
        control_params: ControlParams,
        mux_dict: Mapping[str, Mux],
    ) -> None:
        """
        Initialize a measurement pulse factory.

        Parameters
        ----------
        control_params : ControlParams
            Control parameters providing readout and pump amplitudes.
        mux_dict : Mapping[str, Mux]
            Mapping from qubit label to mux metadata.
        """
        self._control_params = control_params
        self._mux_dict = mux_dict

    def readout_pulse(
        self,
        target: str,
        *,
        duration: float | None = None,
        amplitude: float | None = None,
        ramptime: float | None = None,
        type: RampType | None = None,
        drag_coeff: float | None = None,
        pre_margin: float | None = None,
        post_margin: float | None = None,
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
        ramptime : float | None, optional
            Ramp time for the envelope.
        type : RampType | None, optional
            Ramp type name.
        drag_coeff : float | None, optional
            DRAG coefficient.
        pre_margin : float | None, optional
            Pre-readout margin.
        post_margin : float | None, optional
            Post-readout margin.

        Returns
        -------
        PulseArray
            Readout pulse array with margins.
        """
        qubit = Target.qubit_label(target)
        if duration is None:
            duration = DEFAULT_READOUT_DURATION
        if amplitude is None:
            amplitude = self._control_params.get_readout_amplitude(qubit)
        if ramptime is None:
            ramptime = DEFAULT_READOUT_RAMPTIME
        if type is None:
            type = DEFAULT_READOUT_RAMP_TYPE
        if drag_coeff is None:
            drag_coeff = DEFAULT_READOUT_DRAG_COEFF
        if pre_margin is None:
            pre_margin = DEFAULT_READOUT_PRE_MARGIN
        if post_margin is None:
            post_margin = DEFAULT_READOUT_POST_MARGIN
        pulse = FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=ramptime,
            beta=drag_coeff,
            type=type,
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
        ramptime: float | None = None,
        type: RampType | None = None,
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
        ramptime : float | None, optional
            Ramp time for the envelope.
        type : RampType | None, optional
            Ramp type name.

        Returns
        -------
        FlatTop
            Pump pulse.
        """
        if duration is None:
            duration = DEFAULT_READOUT_DURATION
        if amplitude is None:
            amplitude = self._control_params.get_pump_amplitude(mux_index)
        if ramptime is None:
            ramptime = DEFAULT_READOUT_RAMPTIME
        if type is None:
            type = DEFAULT_READOUT_RAMP_TYPE
        return FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=ramptime,
            type=type,
        )
