"""Data acquisition configuration model."""

from __future__ import annotations

from qxcore import Model, Time, ValueArrayLike


class DataAcquisitionConfig(Model):
    """Data acquisition configuration."""

    shot_count: int
    shot_repetition_margin: Time
    data_acquisition_duration: Time
    data_acquisition_delay: Time
    data_acquisition_timeout: Time
    flag_average_waveform: bool
    flag_average_shots: bool
    delta_time: Time
    channel_to_averaging_time: dict[str, Time]
    channel_to_averaging_window: dict[str, ValueArrayLike]
