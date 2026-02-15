"""Qubex-specific Sequencer wrapper on top of selected driver Sequencer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from .driver_loader import load_quel_driver

if TYPE_CHECKING:
    from .driver_protocols import SequencerProtocol as Sequencer
else:
    Sequencer = load_quel_driver().Sequencer


class Quel1Sequencer(Sequencer):
    """
    Sequencer variant that keeps driver behavior except first-padding insertion.

    Notes
    -----
    We intentionally rely on the selected driver package's base Sequencer
    implementation for port-direction checks and PortConfigAcquirer setup.

    The only behavioral difference is that automatic first-padding insertion is
    disabled. Qubex already prepares waveform timing beforehand, and adding extra
    padding at the driver layer would modify those prepared timings again.
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
