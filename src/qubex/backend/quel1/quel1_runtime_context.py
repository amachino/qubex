"""Runtime context shared across QuEL-1 backend managers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast, runtime_checkable

from .compat.driver_loader import load_quel1_driver

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        QubeCalibProtocol as QubeCalib,
        Quel1SystemProtocol as Quel1System,
        QuelDriverClassesProtocol,
    )

_TConnectedResource = TypeVar("_TConnectedResource")


NOT_CONNECTED_ERROR_MESSAGE = "Boxes not connected. Call connect() method first."


@runtime_checkable
class Quel1RuntimeContextReader(Protocol):
    """Read-only interface for QuEL-1 runtime state shared by managers."""

    @property
    def driver(self) -> QuelDriverClassesProtocol:
        """Return loaded QuEL-1 driver class bundle."""
        ...

    @property
    def qubecalib(self) -> QubeCalib:
        """Return configured qubecalib instance."""
        ...

    @property
    def box_options(self) -> Mapping[str, tuple[str, ...]]:
        """Return per-box relink option labels."""
        ...

    @property
    def available_boxes(self) -> list[str]:
        """Return currently defined box names."""
        ...

    def validate_box_availability(self, box_name: str) -> None:
        """Validate that the target box exists in current system configuration."""
        ...

    @property
    def sampling_period(self) -> float:
        """Return backend sampling period in ns."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return whether runtime resources are connected."""
        ...

    @property
    def boxpool(self) -> BoxPool:
        """Return connected boxpool."""
        ...

    @property
    def quel1system(self) -> Quel1System:
        """Return connected Quel1System."""
        ...

    @property
    def cap_resource_map(self) -> dict[str, dict]:
        """Return capture resource map."""
        ...

    @property
    def gen_resource_map(self) -> dict[str, dict]:
        """Return generator resource map."""
        ...


class Quel1RuntimeContext:
    """Mutable runtime state shared by QuEL-1 backend managers."""

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        driver: QuelDriverClassesProtocol,
        qubecalib: QubeCalib | None,
        sampling_period: float,
    ) -> None:
        self._driver = driver
        self._qubecalib = qubecalib
        self._sampling_period = sampling_period
        self._box_options: dict[str, tuple[str, ...]] = {}
        self._boxpool: BoxPool | None = None
        self._quel1system: Quel1System | None = None
        self._cap_resource_map: dict[str, dict] | None = None
        self._gen_resource_map: dict[str, dict] | None = None

    @classmethod
    def create(cls, *, config_path: str | Path | None = None) -> Quel1RuntimeContext:
        """Create runtime context from driver-loader and optional config file."""
        driver = cast("QuelDriverClassesProtocol", load_quel1_driver())
        qubecalib: QubeCalib | None
        try:
            if config_path is None:
                qubecalib = driver.QubeCalib()
            else:
                try:
                    qubecalib = driver.QubeCalib(str(config_path))
                except FileNotFoundError:
                    cls.logger.warning(f"Configuration file {config_path} not found.")
                    raise
        except Exception:
            qubecalib = None
        return cls(
            driver=driver,
            qubecalib=qubecalib,
            sampling_period=driver.DEFAULT_SAMPLING_PERIOD,
        )

    @property
    def driver(self) -> QuelDriverClassesProtocol:
        """Return loaded QuEL-1 driver class bundle."""
        return self._driver

    @property
    def qubecalib(self) -> QubeCalib:
        """Return configured qubecalib instance."""
        qubecalib = self._qubecalib
        if qubecalib is None:
            raise ModuleNotFoundError(name="qubecalib")
        return qubecalib

    @property
    def box_options(self) -> Mapping[str, tuple[str, ...]]:
        """Return per-box relink option labels."""
        return self._box_options

    @property
    def available_boxes(self) -> list[str]:
        """Return currently defined box names."""
        system_config = self.qubecalib.system_config_database.asdict()
        box_settings = system_config.get("box_settings", {})
        return list(box_settings.keys())

    @property
    def sampling_period(self) -> float:
        """Return backend sampling period in ns."""
        return self._sampling_period

    @property
    def is_connected(self) -> bool:
        """Return whether runtime resources are connected."""
        return self._quel1system is not None

    @property
    def boxpool(self) -> BoxPool:
        """Return connected boxpool."""
        return self._require_connected_resource(self._boxpool)

    @property
    def quel1system(self) -> Quel1System:
        """Return connected Quel1System."""
        return self._require_connected_resource(self._quel1system)

    @property
    def cap_resource_map(self) -> dict[str, dict]:
        """Return capture resource map."""
        return self._require_connected_resource(self._cap_resource_map)

    @property
    def gen_resource_map(self) -> dict[str, dict]:
        """Return generator resource map."""
        return self._require_connected_resource(self._gen_resource_map)

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

    def set_box_options(self, box_options: Mapping[str, tuple[str, ...]]) -> None:
        """Replace per-box relink option labels."""
        self._box_options = {
            box_name: tuple(option_labels)
            for box_name, option_labels in box_options.items()
        }

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

    @staticmethod
    def _require_connected_resource(
        resource: _TConnectedResource | None,
    ) -> _TConnectedResource:
        """Return connected resource or raise a consistent error."""
        if resource is None:
            raise ValueError(NOT_CONNECTED_ERROR_MESSAGE)
        return resource

    def validate_box_availability(self, box_name: str) -> None:
        """Validate that the target box exists in current system configuration."""
        available_boxes = self.available_boxes
        if box_name not in available_boxes:
            raise ValueError(
                f"Box {box_name} not in available boxes: {available_boxes}"
            )
