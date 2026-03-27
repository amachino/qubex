"""Measurement configuration model."""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from qubex.core import Model


class ReturnItem(str, Enum):
    """Return payload item kinds."""

    WAVEFORM_SERIES = "waveform_series"
    IQ_SERIES = "iq_series"
    STATE_SERIES = "state_series"
    AVERAGED_WAVEFORM = "averaged_waveform"
    AVERAGED_IQ = "averaged_iq"


class MeasurementConfig(Model):
    """Measurement configuration model."""

    n_shots: int
    shot_interval: float
    shot_averaging: bool
    time_integration: bool
    state_classification: bool
    classification_source: str | None = None
    return_items: tuple[ReturnItem, ...] = ()

    @model_validator(mode="after")
    def _validate_invariants(self) -> MeasurementConfig:
        """Validate mode invariants and `return_items` consistency."""
        if self.n_shots <= 0:
            raise ValueError("n_shots must be positive.")

        if self.classification_source is not None:
            if self.classification_source != "gmm_linear":
                raise ValueError(
                    "classification_source must be `gmm_linear` when provided."
                )
            if not self.state_classification:
                raise ValueError(
                    "classification_source='gmm_linear' requires state_classification=True."
                )
            if self.shot_averaging:
                raise ValueError(
                    "classification_source='gmm_linear' requires shot_averaging=False."
                )
            if not self.time_integration:
                raise ValueError(
                    "classification_source='gmm_linear' requires time_integration=True."
                )

        return_items = tuple(self.return_items)
        if len(return_items) == 0:
            inferred = self._infer_return_items()
            object.__setattr__(self, "return_items", inferred)
            return_items = inferred

        if len(return_items) != len(set(return_items)):
            raise ValueError("return_items must not contain duplicates.")

        allowed = self._allowed_return_items()
        disallowed = [item for item in return_items if item not in allowed]
        if disallowed:
            joined = ", ".join(item.value for item in disallowed)
            raise ValueError(
                f"return_items contains unsupported entries for this mode: {joined}."
            )

        required = {self._primary_return_item()}
        if (
            self.state_classification
            and self._primary_return_item() != ReturnItem.STATE_SERIES
        ):
            required.add(ReturnItem.STATE_SERIES)
        missing = [item for item in required if item not in return_items]
        if missing:
            joined = ", ".join(item.value for item in missing)
            raise ValueError(
                f"return_items must include required entries for this mode: {joined}."
            )
        return self

    @property
    def primary_return_item(self) -> ReturnItem:
        """Return the primary requested payload item for this configuration."""
        return self._primary_return_item()

    def _primary_return_item(self) -> ReturnItem:
        """Return the primary return item inferred from legacy mode flags."""
        if self.classification_source == "gmm_linear":
            return ReturnItem.STATE_SERIES
        match (self.shot_averaging, self.time_integration):
            case (True, True):
                return ReturnItem.AVERAGED_IQ
            case (True, False):
                return ReturnItem.AVERAGED_WAVEFORM
            case (False, True):
                return ReturnItem.IQ_SERIES
            case (False, False):
                return ReturnItem.WAVEFORM_SERIES

    def _infer_return_items(self) -> tuple[ReturnItem, ...]:
        """Infer default return items from legacy flags."""
        primary = self._primary_return_item()
        items: list[ReturnItem] = [primary]
        if self.state_classification and primary != ReturnItem.STATE_SERIES:
            items.append(ReturnItem.STATE_SERIES)
        return tuple(items)

    def _allowed_return_items(self) -> set[ReturnItem]:
        """Return allowed return-item set for the configured mode."""
        if self.classification_source == "gmm_linear":
            return {ReturnItem.STATE_SERIES}
        if self.shot_averaging:
            allowed: set[ReturnItem] = {
                ReturnItem.AVERAGED_WAVEFORM,
                ReturnItem.AVERAGED_IQ,
            }
        else:
            allowed = {
                ReturnItem.WAVEFORM_SERIES,
                ReturnItem.IQ_SERIES,
            }
        if self.state_classification:
            allowed.add(ReturnItem.STATE_SERIES)
        return allowed
