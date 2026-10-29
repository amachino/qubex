"""Input validation for legacy measurement APIs."""

from __future__ import annotations


def reject_legacy_state_classification_request(
    *,
    state_classification: bool | None,
    enable_dsp_classification: object = None,
) -> None:
    """Reject classified-state output requested through IQ-only legacy APIs."""
    if state_classification is True or enable_dsp_classification is True:
        raise ValueError(
            "Legacy `measure()` and `execute()` results cannot represent "
            "preclassified state data. Use `await run_measurement(..., "
            "state_classification=True)` and keep its MeasurementResult instead."
        )
