"""Typed native frequency settings for QuEL-1 sweep scopes."""

from dataclasses import dataclass
from typing import TypedDict

from typing_extensions import NotRequired

from qubex.backend.frequency_sweep import SweepConfiguration


class Quel1ChannelFrequencySettings(TypedDict):
    """Store a generator or capture channel's FNCO frequency in Hz."""

    fnco_freq: int | None


class Quel1PortFrequencySettings(TypedDict):
    """Store one port's LO, CNCO, and selected channel settings in Hz."""

    lo_freq: int | None
    cnco_freq: int | None
    channels: NotRequired[dict[int, Quel1ChannelFrequencySettings]]
    runits: NotRequired[dict[int, Quel1ChannelFrequencySettings]]


class Quel1BoxFrequencySettings(TypedDict):
    """Collect selected native frequency settings by port number."""

    ports: dict[int | tuple[int, int], Quel1PortFrequencySettings]


@dataclass(frozen=True)
class Quel1SweepConfiguration(SweepConfiguration):
    """Carry native QuEL-1 frequency settings through common sweep APIs."""

    settings: dict[str, Quel1BoxFrequencySettings]
