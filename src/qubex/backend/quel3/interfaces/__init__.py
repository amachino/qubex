"""Protocol interfaces for QuEL-3 <-> quelware-client integration."""

from __future__ import annotations

from qubex.backend.quel3.interfaces.client import (
    FixedTimelineProfileFactory,
    FixedTimelineProfileProtocol,
    InstrumentDefinitionFactory,
    InstrumentDefinitionProtocol,
    InstrumentInfoProtocol,
    InstrumentModeNamespaceProtocol,
    InstrumentModeProtocol,
    InstrumentRoleNamespaceProtocol,
    InstrumentRoleProtocol,
    QuelwareClientFactory,
    QuelwareClientProtocol,
    ResourceIdProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.interfaces.directives import (
    CaptureModeNamespaceProtocol,
    CaptureModeProtocol,
    DirectiveProtocol,
    SetCaptureModeFactory,
    SetFrequencyFactory,
)
from qubex.backend.quel3.interfaces.driver import (
    InstrumentConfigProtocol,
    InstrumentDriverFactory,
    InstrumentDriverProtocol,
    IqWaveformResultProtocol,
    ResultContainerProtocol,
)
from qubex.backend.quel3.interfaces.resolver import (
    InstrumentResolverFactory,
    InstrumentResolverProtocol,
)
from qubex.backend.quel3.interfaces.sequencer import SequencerProtocol

__all__ = [
    "CaptureModeNamespaceProtocol",
    "CaptureModeProtocol",
    "DirectiveProtocol",
    "FixedTimelineProfileFactory",
    "FixedTimelineProfileProtocol",
    "InstrumentConfigProtocol",
    "InstrumentDefinitionFactory",
    "InstrumentDefinitionProtocol",
    "InstrumentDriverFactory",
    "InstrumentDriverProtocol",
    "InstrumentInfoProtocol",
    "InstrumentModeNamespaceProtocol",
    "InstrumentModeProtocol",
    "InstrumentResolverFactory",
    "InstrumentResolverProtocol",
    "InstrumentRoleNamespaceProtocol",
    "InstrumentRoleProtocol",
    "IqWaveformResultProtocol",
    "QuelwareClientFactory",
    "QuelwareClientProtocol",
    "ResourceIdProtocol",
    "ResultContainerProtocol",
    "SequencerProtocol",
    "SessionProtocol",
    "SetCaptureModeFactory",
    "SetFrequencyFactory",
]
