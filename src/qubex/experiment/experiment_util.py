from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from typing import Collection

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..backend import SAMPLING_PERIOD, SystemManager


class ExperimentUtil:
    @staticmethod
    @contextmanager
    def no_output():
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
    def discretize_time_range(
        time_range: ArrayLike,
        sampling_period: float = SAMPLING_PERIOD,
    ) -> NDArray[np.float64]:
        """
        Discretizes the time range.

        Parameters
        ----------
        time_range : ArrayLike
            Time range to discretize in ns.
        sampling_period : float, optional
            Sampling period in ns. Defaults to SAMPLING_PERIOD.

        Returns
        -------
        NDArray[np.float64]
            Discretized time range.
        """
        discretized_range = np.array(time_range)
        discretized_range = (
            np.round(discretized_range / sampling_period) * sampling_period
        )
        return discretized_range

    @staticmethod
    def split_frequency_range(
        frequency_range: ArrayLike,
        subrange_width: float = 0.3,
    ) -> list[NDArray[np.float64]]:
        """
        Splits the frequency range into sub-ranges.

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
        frequency_range = np.array(frequency_range)
        range_count = (frequency_range[-1] - frequency_range[0]) // subrange_width + 1
        sub_ranges = np.array_split(frequency_range, range_count)
        return sub_ranges

    @staticmethod
    def create_qubit_subgroups(
        qubits: Collection[str],
    ) -> list[list[str]]:
        """
        Creates subgroups of qubits.

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
