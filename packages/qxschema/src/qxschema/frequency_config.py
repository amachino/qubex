"""Frequency configuration model."""

from __future__ import annotations

from qxcore import Frequency, Model


class FrequencyConfig(Model):
    """Frequency configuration for channels."""

    channel_to_frequency: dict[str, Frequency]
    channel_to_frequency_reference: dict[str, str]
    channel_to_frequency_shift: dict[str, Frequency]
    keep_oscillator_relative_phase: bool
