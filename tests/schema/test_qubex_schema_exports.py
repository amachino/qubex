"""Tests for qubex.schema exports."""

from __future__ import annotations


def test_qubex_schema_exports_sweep_measurement_models() -> None:
    """Given qubex.schema, when importing models, then expected schema classes are available."""
    from qubex.schema import (
        DataAcquisitionConfig,
        FrequencyConfig,
        ParameterSweepConfig,
        ParameterSweepContent,
        ParametricSequenceConfig,
        ParametricSequencePulseCommand,
        SweepMeasurementConfig,
        SweepMeasurementResult,
    )

    assert DataAcquisitionConfig.__name__ == "DataAcquisitionConfig"
    assert FrequencyConfig.__name__ == "FrequencyConfig"
    assert ParameterSweepConfig.__name__ == "ParameterSweepConfig"
    assert ParameterSweepContent.__name__ == "ParameterSweepContent"
    assert ParametricSequenceConfig.__name__ == "ParametricSequenceConfig"
    assert ParametricSequencePulseCommand.__name__ == "ParametricSequencePulseCommand"
    assert SweepMeasurementConfig.__name__ == "SweepMeasurementConfig"
    assert SweepMeasurementResult.__name__ == "SweepMeasurementResult"
