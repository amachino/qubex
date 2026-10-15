"""Shared helpers for loading quelware runtime dependencies."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Final, Literal, cast

from qubex.backend.quel3.interfaces import (
    CaptureModeNamespaceProtocol,
    CaptureModeProtocol,
    DirectiveProtocol,
    FixedTimelineProfileFactory,
    InstrumentDefinitionFactory,
    InstrumentDriverFactory,
    InstrumentModeNamespaceProtocol,
    InstrumentRoleNamespaceProtocol,
    InstrumentRoleProtocol,
    QuelwareClientFactory,
    SequencerFactoryProtocol,
    SequencerProtocol,
    SetCaptureModeFactory,
    SetFrequencyFactory,
)
from qubex.backend.quel3.interfaces.directives import SetTimingOffsetFactory
from qubex.backend.quel3.models import InstrumentRoleName, Quel3CaptureMode

from .quelware_transport_config import (
    Quel3HttpTransportConfig,
    Quel3Transport,
    validate_quel3_transport_config,
)

Quel3ClientMode = Literal["server"]
SUPPORTED_QUEL3_CLIENT_MODES: Final[frozenset[Quel3ClientMode]] = frozenset({"server"})

_CHANNEL_OVERRIDE_LOCK = Lock()


def normalize_quel3_client_mode(value: object) -> Quel3ClientMode | None:
    """Normalize one QuEL-3 client-mode value to the canonical literal."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SUPPORTED_QUEL3_CLIENT_MODES:
            return cast(Quel3ClientMode, normalized)
    return None


def validate_quelware_client_runtime(
    *,
    client_mode: str,
) -> Quel3ClientMode:
    """Validate one QuEL-3 client runtime and return normalized mode."""
    normalized_client_mode = normalize_quel3_client_mode(client_mode)
    if normalized_client_mode is None:
        raise ValueError(f"Unsupported QuEL-3 client mode: {client_mode!r}")
    return normalized_client_mode


def load_quelware_client_factory(
    *,
    client_mode: Quel3ClientMode,
    pat_path: str | None = None,
    transport: str = "grpc",
    http_transport: Quel3HttpTransportConfig | None = None,
) -> QuelwareClientFactory:
    """Load one quelware client factory for the configured runtime mode."""
    validate_quelware_client_runtime(client_mode=client_mode)
    normalized_transport = validate_quel3_transport_config(
        transport=transport,
        http_transport=http_transport,
    )

    client_module = importlib.import_module("quelware_client.client")
    grpc_factory_module = importlib.import_module("quelware_client.client._grpc")
    channel_factory = _load_channel_factory(
        transport=normalized_transport,
        http_transport=http_transport,
    )

    pat_provider: Callable[[], str] | None = None
    if pat_path is not None:
        path = Path(pat_path).expanduser()
        pat_provider = lambda: path.read_text(encoding="utf-8").rstrip("\r\n")

    def _create_client(endpoint: str, port: int | None):
        with _CHANNEL_OVERRIDE_LOCK:
            original_channel = grpc_factory_module.__dict__["Channel"]
            grpc_factory_module.__dict__["Channel"] = channel_factory
            try:
                if pat_provider is not None:
                    return client_module.create_quelware_client(
                        endpoint,
                        port,
                        pat=pat_provider,
                    )
                return client_module.create_quelware_client(endpoint, port)
            finally:
                grpc_factory_module.__dict__["Channel"] = original_channel

    return cast(QuelwareClientFactory, _create_client)


def _load_channel_factory(
    *,
    transport: Quel3Transport,
    http_transport: Quel3HttpTransportConfig | None,
) -> object:
    if transport == "grpc":
        grpclib_client_module = importlib.import_module("grpclib.client")
        return grpclib_client_module.Channel

    from .quelware_http_transport import ProtobufHttpChannel

    options = http_transport or Quel3HttpTransportConfig()
    secret_headers = {
        name: _read_secret(path) for name, path in options.secret_header_paths.items()
    }
    proxy_url = (
        _read_secret(options.proxy_url_path)
        if options.proxy_url_path is not None
        else None
    )
    return partial(
        ProtobufHttpChannel,
        scheme=transport,
        base_path=options.base_path,
        default_timeout_seconds=options.default_timeout_seconds,
        proxy_url=proxy_url,
        secret_headers=secret_headers,
    )


def _read_secret(path: str) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8").rstrip("\r\n")


@dataclass(frozen=True)
class QuelwareInstrumentEntities:
    """Lazy-loaded quelware instrument entities needed for deployment."""

    fixed_timeline_profile_factory: FixedTimelineProfileFactory
    instrument_definition_factory: InstrumentDefinitionFactory
    instrument_mode_namespace: InstrumentModeNamespaceProtocol
    instrument_role_namespace: InstrumentRoleNamespaceProtocol

    def role_value(self, role: InstrumentRoleName) -> InstrumentRoleProtocol:
        """Return quelware instrument-role value for one deploy role name."""
        if role == "TRANSMITTER":
            return self.instrument_role_namespace.TRANSMITTER
        if role == "TRANSCEIVER":
            return self.instrument_role_namespace.TRANSCEIVER
        if role == "TRANSCEIVER_LOOPBACK":
            return self.instrument_role_namespace.TRANSCEIVER_LOOPBACK
        if role == "RECEIVER":
            return self.instrument_role_namespace.RECEIVER
        raise ValueError(f"Unsupported QuEL-3 instrument role: {role!r}")


@dataclass(frozen=True)
class QuelwareExecutionApi:
    """Lazy-loaded quelware API symbols needed for fixed-timeline execution."""

    client_factory: QuelwareClientFactory
    sequencer_factory: SequencerFactoryProtocol[SequencerProtocol]
    fixed_timeline_driver_factory: InstrumentDriverFactory
    set_frequency_directive_factory: SetFrequencyFactory
    set_capture_mode_directive_factory: SetCaptureModeFactory
    capture_mode_namespace: CaptureModeNamespaceProtocol
    set_timing_offset_directive_factory: SetTimingOffsetFactory | None = None

    def build_capture_mode_directive(
        self,
        capture_mode: Quel3CaptureMode,
    ) -> DirectiveProtocol:
        """Build one capture-mode directive from payload capture mode."""
        if capture_mode is Quel3CaptureMode.UNSPECIFIED:
            raise ValueError(f"Unsupported capture mode: {capture_mode}.")
        try:
            mode = cast(
                CaptureModeProtocol,
                getattr(self.capture_mode_namespace, capture_mode.name),
            )
        except AttributeError as exc:
            raise RuntimeError(
                "quelware runtime does not expose required "
                f"`CaptureMode.{capture_mode.name}`."
            ) from exc
        return self.set_capture_mode_directive_factory(mode=mode)


def load_quelware_execution_api(
    *, client_factory: QuelwareClientFactory
) -> QuelwareExecutionApi:
    """Load quelware execution symbols and bind the configured client factory."""
    sequencer_module = importlib.import_module(
        "quelware_client.client.helpers.sequencer"
    )
    directive_module = importlib.import_module("quelware_core.entities.directives")
    driver_module = importlib.import_module("quelware_client.core.instrument_driver")
    return QuelwareExecutionApi(
        client_factory=client_factory,
        sequencer_factory=sequencer_module.Sequencer,
        fixed_timeline_driver_factory=driver_module.create_instrument_driver_fixed_timeline,
        capture_mode_namespace=directive_module.CaptureMode,
        set_timing_offset_directive_factory=getattr(
            directive_module, "SetTimingOffset", None
        ),
        set_frequency_directive_factory=directive_module.SetFrequency,
        set_capture_mode_directive_factory=directive_module.SetCaptureMode,
    )


def load_quelware_instrument_entities() -> QuelwareInstrumentEntities:
    """Load instrument entity factories from the optional quelware core package."""
    instrument_module = importlib.import_module("quelware_core.entities.instrument")
    return QuelwareInstrumentEntities(
        fixed_timeline_profile_factory=instrument_module.FixedTimelineProfile,
        instrument_definition_factory=instrument_module.InstrumentDefinition,
        instrument_mode_namespace=instrument_module.InstrumentMode,
        instrument_role_namespace=instrument_module.InstrumentRole,
    )
