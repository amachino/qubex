"""Runtime context shared across QuEL-1 backend managers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qubex.backend.quel1.quel1_qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        Quel1SystemProtocol as Quel1System,
    )


@runtime_checkable
class Quel1RuntimeContextReader(Protocol):
    """Read-only interface for QuEL-1 runtime state shared by managers."""

    @property
    def is_connected(self) -> bool:
        """Return whether runtime resources are connected."""
        ...

    @property
    def boxpool(self) -> BoxPool | None:
        """Return connected boxpool when available."""
        ...

    @property
    def quel1system(self) -> Quel1System | None:
        """Return connected Quel1System when available."""
        ...

    @property
    def cap_resource_map(self) -> dict[str, dict] | None:
        """Return capture resource map when available."""
        ...

    @property
    def gen_resource_map(self) -> dict[str, dict] | None:
        """Return generator resource map when available."""
        ...


class Quel1RuntimeContext:
    """Mutable runtime state shared by QuEL-1 backend managers."""

    def __init__(self) -> None:
        self._boxpool: BoxPool | None = None
        self._quel1system: Quel1System | None = None
        self._cap_resource_map: dict[str, dict] | None = None
        self._gen_resource_map: dict[str, dict] | None = None

    @property
    def is_connected(self) -> bool:
        """Return whether runtime resources are connected."""
        return self._quel1system is not None

    @property
    def boxpool(self) -> BoxPool | None:
        """Return connected boxpool when available."""
        return self._boxpool

    @property
    def quel1system(self) -> Quel1System | None:
        """Return connected Quel1System when available."""
        return self._quel1system

    @property
    def cap_resource_map(self) -> dict[str, dict] | None:
        """Return capture resource map when available."""
        return self._cap_resource_map

    @property
    def gen_resource_map(self) -> dict[str, dict] | None:
        """Return generator resource map when available."""
        return self._gen_resource_map

    def set_connected_state(
        self,
        *,
        boxpool: BoxPool | None,
        quel1system: Quel1System | None,
        cap_resource_map: dict[str, dict] | None,
        gen_resource_map: dict[str, dict] | None,
    ) -> None:
        """
        Replace full connected runtime state.

        Parameters
        ----------
        boxpool : BoxPool | None
            Connected boxpool.
        quel1system : Quel1System | None
            Connected Quel1System.
        cap_resource_map : dict[str, dict] | None
            Capture resource map.
        gen_resource_map : dict[str, dict] | None
            Generator resource map.
        """
        self._boxpool = boxpool
        self._quel1system = quel1system
        self._cap_resource_map = cap_resource_map
        self._gen_resource_map = gen_resource_map

    def set_boxpool(self, boxpool: BoxPool | None) -> None:
        """Update only boxpool state."""
        self._boxpool = boxpool

    def set_quel1system(self, quel1system: Quel1System | None) -> None:
        """Update only Quel1System state."""
        self._quel1system = quel1system

    def set_cap_resource_map(self, resource_map: dict[str, dict] | None) -> None:
        """Update only capture resource map state."""
        self._cap_resource_map = resource_map

    def set_gen_resource_map(self, resource_map: dict[str, dict] | None) -> None:
        """Update only generator resource map state."""
        self._gen_resource_map = resource_map

    def clear_connected_state(self) -> None:
        """Clear connected runtime state."""
        self.set_connected_state(
            boxpool=None,
            quel1system=None,
            cap_resource_map=None,
            gen_resource_map=None,
        )
