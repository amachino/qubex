"""QuEL-3 system assembly helpers used by ConfigLoader."""

from __future__ import annotations

from qubex.system.quel1.quel1_system_loader import Quel1SystemLoader


class Quel3SystemLoader(Quel1SystemLoader):
    """
    Assemble QuEL-3 control-system and wiring models.

    Notes
    -----
    This loader currently reuses the legacy port-centric assembly flow.
    Backend-specific divergence points are isolated in this class for future
    unit/instrument model expansion.
    """
