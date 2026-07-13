"""System-scoped partial measurement-default models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MeasurementExecutionDefaults(BaseModel):
    """Partial execution defaults loaded from configuration."""

    model_config = ConfigDict(extra="forbid")

    n_shots: int | None = None
    shot_interval_ns: float | None = None

    @model_validator(mode="after")
    def _validate_positive_values(self) -> MeasurementExecutionDefaults:
        """Validate that configured execution defaults are positive."""
        if self.n_shots is not None and self.n_shots <= 0:
            raise ValueError("execution.n_shots must be positive.")
        if self.shot_interval_ns is not None and self.shot_interval_ns <= 0:
            raise ValueError("execution.shot_interval_ns must be positive.")
        return self


class ReadoutDefaults(BaseModel):
    """Partial readout-timing defaults loaded from configuration."""

    model_config = ConfigDict(extra="forbid")

    duration_ns: float | None = None
    ramp_time_ns: float | None = None
    pre_margin_ns: float | None = None
    post_margin_ns: float | None = None

    @model_validator(mode="after")
    def _validate_non_negative_values(self) -> ReadoutDefaults:
        """Validate that configured readout defaults are non-negative."""
        for name in (
            "duration_ns",
            "ramp_time_ns",
            "pre_margin_ns",
            "post_margin_ns",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"readout.{name} must be non-negative.")
        return self


class MeasurementDefaults(BaseModel):
    """Partial measurement defaults loaded from `measurement_defaults.yaml`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    execution: MeasurementExecutionDefaults = Field(
        default_factory=MeasurementExecutionDefaults
    )
    readout: ReadoutDefaults = Field(default_factory=ReadoutDefaults)

    @model_validator(mode="after")
    def _validate_schema_version(self) -> MeasurementDefaults:
        """Validate the supported schema version."""
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported measurement defaults schema_version: {self.schema_version}."
            )
        return self
