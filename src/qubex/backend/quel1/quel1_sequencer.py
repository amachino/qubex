"""Qubex-specific Sequencer wrapper on top of qubecalib's Sequencer."""

from __future__ import annotations

from qubecalib.qubecalib import Sequencer
from typing_extensions import override


class Quel1Sequencer(Sequencer):
    """
    Sequencer variant that keeps qubecalib behavior except first-padding insertion.

    Notes
    -----
    We intentionally rely on the base qubecalib.qubecalib.Sequencer
    implementation for port-direction checks and PortConfigAcquirer setup, so
    driver is used in the same way as upstream qubecalib.

    The only behavioral difference is that automatic first-padding insertion is
    disabled. Qubex already prepares waveform timing beforehand, and adding extra
    padding at the qubecalib layer would modify those prepared timings again.
    """

    @override
    def calc_first_padding(self) -> int:
        """
        Override Sequencer.calc_first_padding to disable first padding.

        Returns
        -------
        int
            Always 0.

        Examples
        --------
        >>> sequencer.calc_first_padding()
        0
        """
        return 0


# TODO: Remove this alias in future versions.
SequencerMod = Quel1Sequencer
