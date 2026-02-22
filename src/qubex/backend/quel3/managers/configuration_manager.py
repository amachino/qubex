"""Configuration-definition manager for QuEL-3 backend controller."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

PortNumber = int | tuple[int, int]


@dataclass(frozen=True)
class _BoxDefinition:
    ipaddr_wss: str
    boxtype: str


@dataclass(frozen=True)
class _PortDefinition:
    box_name: str
    port_number: PortNumber


@dataclass(frozen=True)
class _ChannelDefinition:
    port_name: str
    channel_number: int
    ndelay_or_nwait: int


@dataclass(frozen=True)
class _TargetDefinition:
    channel_name: str
    target_frequency: float


@dataclass(frozen=True)
class _PortConfig:
    lo_freq: float | None
    cnco_freq: float | None
    vatt: int | None
    sideband: str | None
    fullscale_current: int | None
    rfswitch: str | None


@dataclass(frozen=True)
class _ChannelConfig:
    fnco_freq: float | None


@dataclass(frozen=True)
class _RunitConfig:
    fnco_freq: float | None


class Quel3ConfigurationManager:
    """Hold backend-side QuEL-3 configuration definitions and updates."""

    def __init__(self) -> None:
        self._clockmaster_ipaddr: str | None = None
        self._box_options: dict[str, tuple[str, ...]] = {}
        self._boxes: dict[str, _BoxDefinition] = {}
        self._ports: dict[str, _PortDefinition] = {}
        self._channels: dict[str, _ChannelDefinition] = {}
        self._channel_target_relations: dict[str, str] = {}
        self._targets: dict[str, _TargetDefinition] = {}
        self._port_configs: dict[tuple[str, PortNumber], _PortConfig] = {}
        self._channel_configs: dict[tuple[str, PortNumber, int], _ChannelConfig] = {}
        self._runit_configs: dict[tuple[str, PortNumber, int], _RunitConfig] = {}
        self._command_queue: list[str] = []

    @property
    def hash(self) -> int:
        """Return stable hash for QuEL-3 configuration state."""
        return hash(
            (
                self._clockmaster_ipaddr,
                tuple(sorted(self._box_options.items())),
                tuple(
                    sorted(
                        (name, definition.ipaddr_wss, definition.boxtype)
                        for name, definition in self._boxes.items()
                    )
                ),
                tuple(
                    sorted(
                        (
                            name,
                            definition.box_name,
                            self._freeze_port_number(definition.port_number),
                        )
                        for name, definition in self._ports.items()
                    )
                ),
                tuple(
                    sorted(
                        (
                            name,
                            definition.port_name,
                            definition.channel_number,
                            definition.ndelay_or_nwait,
                        )
                        for name, definition in self._channels.items()
                    )
                ),
                tuple(sorted(self._channel_target_relations.items())),
                tuple(
                    sorted(
                        (
                            target_name,
                            definition.channel_name,
                            definition.target_frequency,
                        )
                        for target_name, definition in self._targets.items()
                    )
                ),
                tuple(
                    sorted(
                        (
                            box_name,
                            self._freeze_port_number(port_number),
                            config.lo_freq,
                            config.cnco_freq,
                            config.vatt,
                            config.sideband,
                            config.fullscale_current,
                            config.rfswitch,
                        )
                        for (
                            box_name,
                            port_number,
                        ), config in self._port_configs.items()
                    )
                ),
                tuple(
                    sorted(
                        (
                            box_name,
                            self._freeze_port_number(port_number),
                            channel_number,
                            config.fnco_freq,
                        )
                        for (
                            box_name,
                            port_number,
                            channel_number,
                        ), config in self._channel_configs.items()
                    )
                ),
                tuple(
                    sorted(
                        (
                            box_name,
                            self._freeze_port_number(port_number),
                            runit_number,
                            config.fnco_freq,
                        )
                        for (
                            box_name,
                            port_number,
                            runit_number,
                        ), config in self._runit_configs.items()
                    )
                ),
                tuple(self._command_queue),
            )
        )

    def set_box_options(self, box_options: Mapping[str, Sequence[str]]) -> None:
        """Set optional per-box options."""
        self._box_options = {
            box_name: tuple(options) for box_name, options in box_options.items()
        }

    def define_clockmaster(self, *, ipaddr: str, reset: bool = True) -> None:
        """Store clockmaster definition."""
        del reset
        self._clockmaster_ipaddr = ipaddr

    def define_box(
        self,
        *,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
    ) -> None:
        """Define one backend box."""
        self._boxes[box_name] = _BoxDefinition(ipaddr_wss=ipaddr_wss, boxtype=boxtype)

    def define_port(
        self,
        *,
        port_name: str,
        box_name: str,
        port_number: PortNumber,
    ) -> None:
        """Define one backend port."""
        self._ports[port_name] = _PortDefinition(
            box_name=box_name,
            port_number=port_number,
        )

    def define_channel(
        self,
        *,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> None:
        """Define one backend channel."""
        self._channels[channel_name] = _ChannelDefinition(
            port_name=port_name,
            channel_number=channel_number,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def add_channel_target_relation(self, channel_name: str, target_name: str) -> None:
        """Define one channel-to-target relation."""
        self._channel_target_relations[channel_name] = target_name

    def define_target(
        self,
        *,
        target_name: str,
        channel_name: str,
        target_frequency: float,
    ) -> None:
        """Define one backend target."""
        self._targets[target_name] = _TargetDefinition(
            channel_name=channel_name,
            target_frequency=target_frequency,
        )

    def clear_command_queue(self) -> None:
        """Clear pending backend command queue."""
        self._command_queue.clear()

    def clear_cache(self) -> None:
        """Clear transient backend-side configuration cache."""
        self._port_configs.clear()
        self._channel_configs.clear()
        self._runit_configs.clear()

    def modify_target_frequencies(self, frequencies: Mapping[str, float]) -> None:
        """Update target-frequency definitions."""
        for target_name, frequency in frequencies.items():
            target = self._targets.get(target_name)
            if target is None:
                continue
            self._targets[target_name] = _TargetDefinition(
                channel_name=target.channel_name,
                target_frequency=frequency,
            )

    def config_port(
        self,
        box_name: str,
        *,
        port: PortNumber,
        lo_freq: float | None = None,
        cnco_freq: float | None = None,
        vatt: int | None = None,
        sideband: str | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
    ) -> None:
        """Store one port-configuration update."""
        self._port_configs[(box_name, port)] = _PortConfig(
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )
        self._command_queue.append(
            f"config_port:{box_name}:{self._freeze_port_number(port)}"
        )

    def config_channel(
        self,
        box_name: str,
        *,
        port: PortNumber,
        channel: int,
        fnco_freq: float | None = None,
    ) -> None:
        """Store one channel-configuration update."""
        self._channel_configs[(box_name, port, channel)] = _ChannelConfig(
            fnco_freq=fnco_freq,
        )
        self._command_queue.append(
            f"config_channel:{box_name}:{self._freeze_port_number(port)}:{channel}"
        )

    def config_runit(
        self,
        box_name: str,
        *,
        port: PortNumber,
        runit: int,
        fnco_freq: float | None = None,
    ) -> None:
        """Store one runit-configuration update."""
        self._runit_configs[(box_name, port, runit)] = _RunitConfig(
            fnco_freq=fnco_freq,
        )
        self._command_queue.append(
            f"config_runit:{box_name}:{self._freeze_port_number(port)}:{runit}"
        )

    @staticmethod
    def initialize_awg_and_capunits(
        box_names: str | Collection[str],
        *,
        parallel: bool | None = None,
    ) -> None:
        """Keep QuEL-1 compatibility API as a no-op for QuEL-3."""
        del box_names, parallel

    @staticmethod
    def _freeze_port_number(port_number: PortNumber) -> tuple[int, ...]:
        """Normalize port-number key to a tuple for stable hashing."""
        if isinstance(port_number, tuple):
            return port_number
        return (port_number,)
