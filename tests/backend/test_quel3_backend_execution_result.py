"""Tests for QuEL-3 backend execution result model."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from qubex.backend.quel3.quel3_backend_execution_result import (
    Quel3BackendExecutionResult,
)


def test_backend_result_rejects_device_config_argument() -> None:
    """Given legacy device_config argument, backend result raises TypeError."""
    with pytest.raises(TypeError, match="device_config"):
        cast(Any, Quel3BackendExecutionResult)(
            mode="avg",
            data={"alias": [np.array([1.0 + 0.0j], dtype=np.complex128)]},
            device_config={},
        )


def test_backend_result_rejects_measurement_config_argument() -> None:
    """Given legacy measurement_config argument, backend result raises TypeError."""
    with pytest.raises(TypeError, match="measurement_config"):
        cast(Any, Quel3BackendExecutionResult)(
            mode="avg",
            data={"alias": [np.array([1.0 + 0.0j], dtype=np.complex128)]},
            measurement_config={},
        )
