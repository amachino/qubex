"""QuEL-3 instrument deployment manager used by push-time configuration."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from qubex.backend.quel3.infra.quelware_imports import (
    import_module_with_workspace_fallback,
)
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge
from qubex.system.target import TargetType

if TYPE_CHECKING:
    from qubex.system.control_system import GenPort
    from qubex.system.experiment_system import ExperimentSystem
    from qubex.system.target import Target

T = TypeVar("T")
RoleName = Literal["TRANSMITTER", "TRANSCEIVER"]


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-configuration")
    return bridge.run(factory, timeout=timeout)


@dataclass(frozen=True)
class InstrumentDeployRequest:
    """One deploy request derived from grouped logical targets."""

    port_id: str
    role: RoleName
    frequency_range_min_hz: float
    frequency_range_max_hz: float
    alias: str
    target_labels: tuple[str, ...]


class Quel3ConfigurationManager:
    """Build and deploy QuEL-3 instruments from logical target registry."""

    def __init__(
        self,
        *,
        quelware_endpoint: str,
        quelware_port: int,
    ) -> None:
        self._quelware_endpoint = quelware_endpoint
        self._quelware_port = quelware_port
        self._last_deployed_instrument_infos: dict[str, tuple[object, ...]] = {}
        self._target_alias_map: dict[str, str] = {}

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint used for deployment."""
        return self._quelware_endpoint

    @property
    def quelware_port(self) -> int:
        """Return quelware port used for deployment."""
        return self._quelware_port

    @property
    def last_deployed_instrument_infos(self) -> dict[str, tuple[object, ...]]:
        """Return last deployed instrument infos keyed by alias."""
        return dict(self._last_deployed_instrument_infos)

    @property
    def target_alias_map(self) -> dict[str, str]:
        """Return last deployed target-to-alias mapping."""
        return dict(self._target_alias_map)

    def deploy_instruments_from_target_registry(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
    ) -> dict[str, tuple[object, ...]]:
        """Deploy instruments from selected boxes in target registry."""
        return _run_async(
            lambda: self._deploy_instruments_from_target_registry(
                experiment_system=experiment_system,
                box_ids=box_ids,
            )
        )

    async def _deploy_instruments_from_target_registry(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
    ) -> dict[str, tuple[object, ...]]:
        """Deploy grouped target-based instruments through quelware session APIs."""
        requests = self._build_instrument_deploy_requests(
            experiment_system=experiment_system,
            box_ids=box_ids,
        )
        if len(requests) == 0:
            self._last_deployed_instrument_infos = {}
            self._target_alias_map = {}
            return {}

        client_factory = self._load_quelware_client_factory()
        (
            fixed_timeline_profile_cls,
            instrument_definition_cls,
            instrument_mode,
            instrument_role,
        ) = self._load_instrument_entities()

        deployed: dict[str, tuple[object, ...]] = {}
        target_alias_map: dict[str, str] = {}
        async with client_factory(
            self._quelware_endpoint,
            self._quelware_port,
        ) as client:
            for request in requests:
                async with client.create_session([request.port_id]) as session:
                    profile = fixed_timeline_profile_cls(
                        frequency_range_min=request.frequency_range_min_hz,
                        frequency_range_max=request.frequency_range_max_hz,
                    )
                    definition = instrument_definition_cls(
                        alias=request.alias,
                        mode=instrument_mode.FIXED_TIMELINE,
                        role=getattr(instrument_role, request.role),
                        profile=profile,
                    )
                    inst_infos = await session.deploy_instruments(
                        request.port_id,
                        definitions=[definition],
                    )
                    deployed[request.alias] = tuple(inst_infos)
                    for target_label in request.target_labels:
                        target_alias_map[target_label] = request.alias

        self._last_deployed_instrument_infos = dict(deployed)
        self._target_alias_map = target_alias_map
        return deployed

    def _build_instrument_deploy_requests(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
    ) -> tuple[InstrumentDeployRequest, ...]:
        """Build deterministic deploy requests from selected generator targets."""
        selected_box_ids = set(box_ids)
        grouped_targets: dict[tuple[str, RoleName], list[Target]] = defaultdict(list)

        for _label, target in sorted(experiment_system.gen_targets.items()):
            port = target.channel.port
            if port.box_id not in selected_box_ids:
                continue
            role = self._resolve_instrument_role(target.type)
            port_id = self._resolve_port_id(
                experiment_system=experiment_system,
                target=target,
            )
            grouped_targets[(port_id, role)].append(target)

        requests: list[InstrumentDeployRequest] = []
        for (port_id, role), targets in sorted(grouped_targets.items()):
            frequency_hz = [
                self._resolve_target_frequency_hz(target=target) for target in targets
            ]
            freq_min = min(frequency_hz)
            freq_max = max(frequency_hz)
            if freq_min > freq_max:
                raise ValueError(
                    "Invalid frequency range derived from target group: "
                    f"port_id={port_id} role={role} min={freq_min} max={freq_max}"
                )
            alias = self._build_alias(port_id=port_id, role=role)
            requests.append(
                InstrumentDeployRequest(
                    port_id=port_id,
                    role=role,
                    frequency_range_min_hz=freq_min,
                    frequency_range_max_hz=freq_max,
                    alias=alias,
                    target_labels=tuple(sorted(target.label for target in targets)),
                )
            )
        return tuple(requests)

    def _resolve_port_id(
        self,
        *,
        experiment_system: ExperimentSystem,
        target: Target,
    ) -> str:
        """Resolve quelware port ID from one logical generator target."""
        port = target.channel.port
        port_number = self._resolve_port_number(port=port)

        if target.type == TargetType.READ:
            read_out_port_number = port_number
            read_in_port_number = self._resolve_read_in_port_number(
                experiment_system=experiment_system,
                read_out_port=port,
            )
            return f"{port.box_id}:trx_p{read_in_port_number:02d}p{read_out_port_number:02d}"
        return f"{port.box_id}:tx_p{port_number:02d}"

    @staticmethod
    def _resolve_port_number(*, port: GenPort) -> int:
        """Resolve validated integer port number from one generator port."""
        if not isinstance(port.number, int):
            raise TypeError(f"Port number must be int for QuEL-3 deployment: {port}")
        return port.number

    def _resolve_read_in_port_number(
        self,
        *,
        experiment_system: ExperimentSystem,
        read_out_port: GenPort,
    ) -> int:
        """Resolve paired read-in port number for one read-out generator port."""
        mux = experiment_system.get_mux_by_readout_port(read_out_port)
        if mux is None:
            raise ValueError(f"Readout mux is not found for port `{read_out_port.id}`.")

        for read_in_mux, cap_port in experiment_system.wiring_info.read_in:
            if read_in_mux.index != mux.index:
                continue
            if not isinstance(cap_port.number, int):
                raise TypeError(
                    "Capture port number must be int for QuEL-3 readout deployment."
                )
            return cap_port.number
        raise ValueError(f"Read-in pair is not found for readout mux `{mux.index}`.")

    @staticmethod
    def _resolve_instrument_role(target_type: TargetType) -> RoleName:
        """Resolve instrument role name from logical target type."""
        if target_type == TargetType.READ:
            return "TRANSCEIVER"
        if target_type in (
            TargetType.CTRL_GE,
            TargetType.CTRL_EF,
            TargetType.CTRL_CR,
            TargetType.PUMP,
        ):
            return "TRANSMITTER"
        raise ValueError(f"Unsupported target type for deployment: {target_type}.")

    @staticmethod
    def _resolve_target_frequency_hz(*, target: Target) -> float:
        """Resolve validated target frequency in Hz from GHz value."""
        frequency_hz = float(target.frequency) * 1e9
        if not math.isfinite(frequency_hz):
            raise ValueError(
                f"Target frequency must be finite: label={target.label} frequency={target.frequency}"
            )
        return frequency_hz

    @staticmethod
    def _build_alias(*, port_id: str, role: RoleName) -> str:
        """Build deterministic instrument alias from port and role."""
        normalized_port_id = port_id.replace(":", "_")
        return f"inst_{role.lower()}_{normalized_port_id}"

    @staticmethod
    def _load_quelware_client_factory() -> Callable[[str, int], Any]:
        """Import quelware client factory lazily."""
        client_module = import_module_with_workspace_fallback("quelware_client.client")
        return client_module.create_quelware_client

    @staticmethod
    def _load_instrument_entities() -> tuple[type[Any], type[Any], Any, Any]:
        """Import instrument entities lazily from quelware core package."""
        instrument_module = import_module_with_workspace_fallback(
            "quelware_core.entities.instrument"
        )
        return (
            instrument_module.FixedTimelineProfile,
            instrument_module.InstrumentDefinition,
            instrument_module.InstrumentMode,
            instrument_module.InstrumentRole,
        )
