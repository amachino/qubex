"""Protocol interfaces for QuEL-3 <-> quelware-client integration."""

from __future__ import annotations

from qubex.backend.quel3.interfaces.client import (
    InstrumentDefinitionProtocol,
    InstrumentInfoProtocol,
    QuelwareClientFactory,
    QuelwareClientProtocol,
    ResourceIdProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.interfaces.directives import (
    CaptureModeNamespaceProtocol,
    CaptureModeProtocol,
    CaptureModeValue,
    DirectiveProtocol,
    SetCaptureModeFactory,
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
    "CaptureModeValue",
    "DirectiveProtocol",
    "InstrumentConfigProtocol",
    "InstrumentDefinitionProtocol",
    "InstrumentDriverFactory",
    "InstrumentDriverProtocol",
    "InstrumentInfoProtocol",
    "InstrumentResolverFactory",
    "InstrumentResolverProtocol",
    "IqWaveformResultProtocol",
    "QuelwareClientFactory",
    "QuelwareClientProtocol",
    "ResourceIdProtocol",
    "ResultContainerProtocol",
    "SequencerProtocol",
    "SessionProtocol",
    "SetCaptureModeFactory",
]
