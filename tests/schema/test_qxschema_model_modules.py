"""Tests for qxschema model module boundaries."""

from __future__ import annotations


def test_frequency_config_is_defined_in_dedicated_module() -> None:
    """Given qxschema, when loading FrequencyConfig, then dedicated module defines it."""
    # Arrange
    from qxschema import FrequencyConfig

    # Act
    module_name = FrequencyConfig.__module__

    # Assert
    assert module_name == "qxschema.frequency_config"


def test_data_acquisition_config_is_defined_in_dedicated_module() -> None:
    """Given qxschema, when loading DataAcquisitionConfig, then dedicated module defines it."""
    # Arrange
    from qxschema import DataAcquisitionConfig

    # Act
    module_name = DataAcquisitionConfig.__module__

    # Assert
    assert module_name == "qxschema.data_acquisition_config"
