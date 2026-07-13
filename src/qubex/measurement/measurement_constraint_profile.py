"""Backend timing/constraint profile for measurement scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from qubex.backend.quel1 import (
    BLOCK_LENGTH,
    SAMPLING_PERIOD_NS,
    WORD_LENGTH,
)
from qubex.system.quel1.quel1_system_constants import (
    EXTRA_POST_BLANK_LENGTH,
    EXTRA_SUM_SECTION_LENGTH,
)


@dataclass(frozen=True)
class MeasurementConstraintProfile:
    """Backend timing and schedule-constraint configuration."""

    sampling_period_ns: float
    word_length_samples: int | None
    block_length_samples: int | None
    final_readout_guard_length_samples: int
    extra_sum_section_length_samples: int
    extra_post_blank_length_samples: int
    require_workaround_capture: bool
    enforce_word_alignment: bool
    enforce_block_alignment: bool
    enforce_capture_spacing: bool

    @classmethod
    def quel1(
        cls,
        sampling_period_ns: float = SAMPLING_PERIOD_NS,
    ) -> MeasurementConstraintProfile:
        """Create QuEL-1 constraints with the given sampling period."""
        return cls(
            sampling_period_ns=float(sampling_period_ns),
            word_length_samples=WORD_LENGTH,
            block_length_samples=BLOCK_LENGTH,
            final_readout_guard_length_samples=BLOCK_LENGTH,
            extra_sum_section_length_samples=EXTRA_SUM_SECTION_LENGTH,
            extra_post_blank_length_samples=EXTRA_POST_BLANK_LENGTH,
            require_workaround_capture=True,
            enforce_word_alignment=True,
            enforce_block_alignment=True,
            enforce_capture_spacing=True,
        )

    @classmethod
    def quel3(
        cls,
        sampling_period_ns: float,
    ) -> MeasurementConstraintProfile:
        """Create QuEL-3 constraints that only require sample-grid timing."""
        return cls(
            sampling_period_ns=float(sampling_period_ns),
            word_length_samples=None,
            block_length_samples=None,
            final_readout_guard_length_samples=0,
            extra_sum_section_length_samples=0,
            extra_post_blank_length_samples=0,
            require_workaround_capture=False,
            enforce_word_alignment=False,
            enforce_block_alignment=False,
            enforce_capture_spacing=False,
        )

    @property
    def word_duration_ns(self) -> float | None:
        """Return word duration in ns when word length is defined."""
        if self.word_length_samples is None:
            return None
        return self.word_length_samples * self.sampling_period_ns

    @property
    def block_duration_ns(self) -> float | None:
        """Return block duration in ns when block length is defined."""
        if self.block_length_samples is None:
            return None
        return self.block_length_samples * self.sampling_period_ns

    @property
    def workaround_capture_duration_ns(self) -> float:
        """Return workaround capture duration in ns."""
        return self.extra_sum_section_length_samples * self.sampling_period_ns

    @property
    def final_readout_guard_duration_ns(self) -> float:
        """Return the guard blank inserted before appended final readout."""
        return self.final_readout_guard_length_samples * self.sampling_period_ns

    @property
    def extra_capture_duration_ns(self) -> float:
        """Return additional capture padding duration in ns."""
        return (
            self.extra_sum_section_length_samples + self.extra_post_blank_length_samples
        ) * self.sampling_period_ns
