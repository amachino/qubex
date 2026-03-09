"""Configuration manager for QuEL-3 backend instrument deployment."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from qubex.backend.quel3.infra.quelware_imports import (
    import_module_with_workspace_fallback,
)
from qubex.backend.quel3.models import InstrumentDeployRequest
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

T = TypeVar("T")


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-configuration")
    return bridge.run(factory, timeout=timeout)


class Quel3ConfigurationManager:
    """Deploy QuEL-3 instruments through quelware configuration APIs."""

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

    def deploy_instruments(
        self,
        *,
        requests: Sequence[InstrumentDeployRequest],
    ) -> dict[str, tuple[object, ...]]:
        """Deploy instruments for the provided QuEL-3 requests."""
        return _run_async(lambda: self._deploy_instruments(requests=tuple(requests)))

    async def _deploy_instruments(
        self,
        *,
        requests: tuple[InstrumentDeployRequest, ...],
    ) -> dict[str, tuple[object, ...]]:
        """Deploy instruments through quelware session APIs."""
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
