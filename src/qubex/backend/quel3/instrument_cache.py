"""Single source of cached QuEL-3 instrument identities and runtime information."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence

from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.models import InstrumentConfiguration, InstrumentSpec

logger = logging.getLogger(__name__)


class InstrumentCache:
    """
    Store complete hardware instrument information by logical target alias.

    The controller owns this cache. Entries retain the original hardware objects,
    including their resource IDs and driver configuration. Replacements validate
    the entire candidate state before publishing it and never perform hardware IO.
    """

    def __init__(self) -> None:
        self._instruments: dict[str, InstrumentInfoProtocol] = {}

    @staticmethod
    def alias_for(instrument_info: InstrumentInfoProtocol) -> str:
        """Return the local alias, stripping only the instrument's own unit prefix."""
        alias = instrument_info.definition.alias.strip()
        unit_label, separator, _ = instrument_info.port_id.partition(":")
        if separator:
            alias = alias.removeprefix(f"{unit_label}:").strip()
        if not alias:
            raise ValueError("Instrument alias must not be empty.")
        return alias

    @property
    def hash(self) -> int:
        """Return a fingerprint of the cached instrument identities and ports."""
        return hash(
            tuple(
                sorted(
                    (alias, info.id, info.port_id)
                    for alias, info in self._instruments.items()
                )
            )
        )

    def get(self, alias: str) -> InstrumentInfoProtocol:
        """Return one complete instrument info or explain the required cache update."""
        try:
            return self._instruments[alias]
        except KeyError as exc:
            raise ValueError(
                f"Instrument `{alias}` is not cached; deploy or refresh "
                "instruments before execution."
            ) from exc

    def snapshot(self) -> dict[str, InstrumentInfoProtocol]:
        """Return a copy of the alias index containing the original hardware objects."""
        return dict(self._instruments)

    def export_configuration(self) -> InstrumentConfiguration:
        """Extract deployable settings from the last confirmed hardware information."""
        specifications: list[InstrumentSpec] = []
        for alias, info in sorted(self._instruments.items()):
            definition = info.definition
            mode = getattr(definition.mode, "name", definition.mode)
            if mode != "FIXED_TIMELINE":
                raise ValueError(
                    f"Cannot export instrument `{alias}` with mode {mode!r}."
                )
            profile = definition.profile
            if profile is None:
                raise ValueError(
                    f"Cannot export instrument `{alias}` without a profile."
                )
            try:
                specification = InstrumentSpec.model_validate(
                    {
                        "port_id": info.port_id,
                        "alias": alias,
                        "role": getattr(definition.role, "name", definition.role),
                        "frequency_range_min_hz": profile.frequency_range_min,
                        "frequency_range_max_hz": profile.frequency_range_max,
                    }
                )
            except ValueError as exc:
                raise ValueError(f"Cannot export instrument `{alias}`: {exc}") from exc
            specifications.append(specification)
        return InstrumentConfiguration(instruments=tuple(specifications))

    def clear(self) -> None:
        """Discard all cached instruments."""
        self._instruments = {}

    def replace_all(
        self,
        *,
        instrument_infos: Iterable[InstrumentInfoProtocol],
        allow_duplicate_aliases: bool = False,
    ) -> None:
        """
        Replace all instruments after validating the complete candidate state.

        With `allow_duplicate_aliases=True`, warn and retain the last input for
        each normalized alias. Other identity checks remain strict. Failed
        validation leaves the previous cache unchanged.
        """
        self._instruments = self._index(
            instrument_infos, allow_duplicate_aliases=allow_duplicate_aliases
        )

    def replace_ports(
        self,
        *,
        port_ids: Sequence[str],
        instrument_infos: Iterable[InstrumentInfoProtocol],
    ) -> None:
        """Replace selected ports, retaining instruments on every other port."""
        selected = set(port_ids)
        self._replace_scope(
            instrument_infos=instrument_infos,
            includes=lambda info: info.port_id in selected,
        )

    def replace_units(
        self,
        *,
        unit_labels: Sequence[str],
        instrument_infos: Iterable[InstrumentInfoProtocol],
    ) -> None:
        """Replace selected units, including instruments on ports that disappeared."""
        selected = set(unit_labels)
        self._replace_scope(
            instrument_infos=instrument_infos,
            includes=lambda info: info.port_id.partition(":")[0] in selected,
        )

    def _replace_scope(
        self,
        *,
        instrument_infos: Iterable[InstrumentInfoProtocol],
        includes: Callable[[InstrumentInfoProtocol], bool],
    ) -> None:
        """Validate and publish one scoped replacement atomically."""
        incoming = tuple(instrument_infos)
        if any(not includes(info) for info in incoming):
            raise ValueError("Instrument info lies outside the replacement scope.")
        retained = tuple(
            info for info in self._instruments.values() if not includes(info)
        )
        self._instruments = self._index((*retained, *incoming))

    @classmethod
    def _index(
        cls,
        instrument_infos: Iterable[InstrumentInfoProtocol],
        *,
        allow_duplicate_aliases: bool = False,
    ) -> dict[str, InstrumentInfoProtocol]:
        """Build a validated alias index without copying hardware information."""
        instruments: dict[str, InstrumentInfoProtocol] = {}
        resource_ids: set[str] = set()
        for info in instrument_infos:
            if not info.id.strip() or not info.port_id.strip():
                raise ValueError(
                    "Instrument resource ID and port ID must not be empty."
                )
            alias = cls.alias_for(info)
            if alias in instruments:
                if not allow_duplicate_aliases:
                    raise ValueError(f"Duplicate instrument alias `{alias}`.")
                previous = instruments[alias]
                logger.warning(
                    "Duplicate instrument alias `%s`; replacing resource `%s` "
                    "on port `%s` with resource `%s` on port `%s` in the cache.",
                    alias,
                    previous.id,
                    previous.port_id,
                    info.id,
                    info.port_id,
                )
            if info.id in resource_ids:
                raise ValueError(f"Duplicate instrument resource ID `{info.id}`.")
            instruments[alias] = info
            resource_ids.add(info.id)
        return instruments
