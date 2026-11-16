"""QuEL-3 instrument settings for backend-owned frequency sweep scopes."""

from dataclasses import dataclass

from qubex.backend.frequency_sweep import SweepConfiguration
from qubex.backend.quel3.models import InstrumentConfiguration


@dataclass(frozen=True)
class Quel3SweepConfiguration(SweepConfiguration):
    """Carry native QuEL-3 instrument settings through common sweep APIs."""

    settings: InstrumentConfiguration
