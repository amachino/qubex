"""Experiment utility helpers."""

from __future__ import annotations

import io
import sys
from collections.abc import Collection, Iterator
from contextlib import contextmanager

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qubex.backend.quel1 import SAMPLING_PERIOD as DEFAULT_BACKEND_SAMPLING_PERIOD
from qubex.system import SystemManager


class ExperimentUtil:
    """Utility functions for experiment workflows."""

    @staticmethod
    @contextmanager
    def no_output() -> Iterator[None]:
        """Suppress stdout and stderr within the context."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    @staticmethod
    def resolve_sampling_period(
        sampling_period: float | None = None,
    ) -> float:
        """
        Resolve sampling period from explicit value or backend controller.

        Parameters
        ----------
        sampling_period : float | None, optional
            Explicit sampling period in ns.

        Returns
        -------
        float
            Resolved sampling period in ns.
        """
        if isinstance(sampling_period, (int, float)):
            return float(sampling_period)
        backend_controller = getattr(SystemManager.shared(), "backend_controller", None)
        backend_sampling_period = getattr(
            backend_controller,
            "sampling_period",
            None,
        )
        if isinstance(backend_sampling_period, (int, float)):
            return float(backend_sampling_period)
        return float(DEFAULT_BACKEND_SAMPLING_PERIOD)

    @staticmethod
    def discretize_time_range(
        time_range: ArrayLike,
        sampling_period: float | None = None,
    ) -> NDArray[np.float64]:
        """
        Discretizes the time range.

        Parameters
        ----------
        time_range : ArrayLike
            Time range to discretize in ns.
        sampling_period : float, optional
            Sampling period in ns. Defaults to backend-defined sampling period.

        Returns
        -------
        NDArray[np.float64]
            Discretized time range.
        """
        resolved_sampling_period = ExperimentUtil.resolve_sampling_period(
            sampling_period
        )
        discretized_range = np.array(time_range)
        discretized_range = (
            np.round(discretized_range / resolved_sampling_period)
            * resolved_sampling_period
        )
        return discretized_range

    @staticmethod
    def split_frequency_range(
        frequency_range: ArrayLike,
        subrange_width: float | None = None,
    ) -> list[NDArray[np.float64]]:
        """
        Split the frequency range into sub-ranges.

        Parameters
        ----------
        frequency_range : ArrayLike
            Frequency range to split in GHz.
        ubrange_width : float, optional
            Width of the sub-ranges. Defaults to 0.3 GHz.

        Returns
        -------
        list[NDArray[np.float64]]
            Sub-ranges.
        """
        if subrange_width is None:
            subrange_width = 0.3
        frequency_range = np.array(frequency_range)
        range_count = (frequency_range[-1] - frequency_range[0]) // subrange_width + 1
        sub_ranges = np.array_split(frequency_range, range_count)
        return sub_ranges

    @staticmethod
    def create_qubit_subgroups(
        qubits: Collection[str],
    ) -> list[list[str]]:
        """
        Create subgroups of qubits.

        Parameters
        ----------
        qubits : Collection[str]
            Collection of qubits.

        Returns
        -------
        list[list[str]]
            Subgroups of qubits.
        """
        # TODO: Implement a more general method
        qubit_labels = list(qubits)
        system = SystemManager.shared().experiment_system
        qubit_objects = [system.get_qubit(qubit) for qubit in qubit_labels]
        group03 = [qubit.label for qubit in qubit_objects if qubit.index % 4 in [0, 3]]
        group12 = [qubit.label for qubit in qubit_objects if qubit.index % 4 in [1, 2]]
        return [group03, group12]
