"""QuEL-3 waveform registration options."""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class Quel3WaveformCompression(BaseModel):
    """
    Configure block sharing when compiling a pulse schedule into a payload.

    Sharing is enabled by default for long Rect and FlatTop pulses. Lengths
    count registered IQ samples after readout normalization. The target
    block length defaults to 5000 samples and is rounded down to a hardware
    boundary. A short tail absorbs the preceding block. Sharing requires at
    least two complete blocks. Every new fragment, including
    prefixes and tails, must reach the minimum of 1000 samples by default.
    Disable sharing to register each unique pulse shape in full.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    enabled: bool = True
    target_block_samples: int = Field(default=5000, gt=0)
    min_chunk_samples: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        if self.min_chunk_samples > self.target_block_samples:
            raise ValueError("min_chunk_samples must not exceed target_block_samples.")
        return self
