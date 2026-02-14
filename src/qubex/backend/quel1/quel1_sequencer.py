"""Qubex-specific Sequencer wrapper on top of qxdriver_quel Sequencer."""

from __future__ import annotations

from qxdriver_quel.qubecalib import Sequencer
from typing_extensions import override


class Quel1Sequencer(Sequencer):
    """
    Sequencer variant that keeps qxdriver_quel behavior except first-padding insertion.

    Notes
    -----
    We intentionally rely on the base qxdriver_quel.qubecalib.Sequencer
    implementation for port-direction checks and PortConfigAcquirer setup, so
    driver is used in the same way as the compatibility runtime.

    The only behavioral difference is that automatic first-padding insertion is
    disabled. Qubex already prepares waveform timing beforehand, and adding extra
    padding at the qxdriver_quel layer would modify those prepared timings again.
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
