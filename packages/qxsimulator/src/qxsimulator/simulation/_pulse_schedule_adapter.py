"""Adapters from pulse schedules to simulator controls."""

from __future__ import annotations

import numpy as np
from qxpulse import PulseSchedule

from .control import Control


def controls_from_pulse_schedule(
    pulse_schedule: PulseSchedule,
) -> list[Control]:
    """
    Convert every pulse-schedule channel into a simulator control.

    Parameters
    ----------
    pulse_schedule : PulseSchedule
        Schedule containing sampled channel sequences, cyclic frequencies,
        target labels, and logical-frame metadata.

    Returns
    -------
    list[Control]
        One control for each channel sequence in the schedule.

    Raises
    ------
    ValueError
        If any channel has no frequency or target, or its converted segment
        data violates the `Control` requirements.

    Notes
    -----
    Each sampled value retains the sampling period of its source waveform.
    `Control` copies each sequence and its frame-shift metadata. Intermediate
    virtual-Z operations are already reflected in the sampled sequence phase;
    the copied shifts describe the logical coordinates used to interpret
    simulation results.
    """
    controls = []
    for label, sequence in pulse_schedule.get_sequences(copy=False).items():
        frequency = pulse_schedule.get_frequency(label)
        if frequency is None:
            raise ValueError(f"Frequency for {label} is not provided.")

        target = pulse_schedule.get_target(label)
        if target is None:
            raise ValueError(f"Object for {label} is not provided.")

        waveforms = sequence.get_flattened_waveforms()
        durations = (
            np.concatenate(
                [
                    np.full(waveform.length, waveform.sampling_period)
                    for waveform in waveforms
                ]
            )
            if waveforms
            else np.array([], dtype=np.float64)
        )

        controls.append(
            Control(
                target=target,
                frequency=frequency,
                waveform=sequence,
                durations=durations,
                final_frame_shift=pulse_schedule.get_final_frame_shift(label),
                frame_shifts=sequence.frame_shifts,
            )
        )
    return controls
