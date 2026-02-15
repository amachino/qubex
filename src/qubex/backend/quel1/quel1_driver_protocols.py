"""Type-checking protocols for QuEL driver symbols resolved by loader."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from os import PathLike
from types import ModuleType
from typing import Any, ClassVar, Protocol


class Quel1ConfigOptionProtocol(Protocol):
    """Protocol for Quel1 config option enum-like classes."""

    _value2member_map_: ClassVar[Mapping[str, Any]]


class QuBEMasterClientProtocol(Protocol):
    """Protocol for clock-master client compatibility wrappers."""

    def __init__(
        self,
        master_ipaddr: str | None = None,
        *,
        ipaddr: str | None = None,
    ) -> None:
        """Create a clock-master client."""
        ...

    def kick_clock_synch(self, box_sss_ipaddrs: list[str]) -> None:
        """Trigger clock synchronization for the given SSS endpoints."""
        ...

    def read_clock(self) -> tuple[bool, int]:
        """Read clock-master counter value."""
        ...

    def reset(self) -> bool:
        """Reset clock master when supported by the backend."""
        ...


class SequencerClientProtocol(Protocol):
    """Protocol for sequencer clock readers."""

    def __init__(self, target_ipaddr: str, *, box: Any | None = None) -> None:
        """Create a sequencer client."""
        ...

    def read_clock(self) -> tuple[bool, int, int]:
        """Read current and SYSREF counters."""
        ...


class Quel1BoxProtocol(Protocol):
    """Protocol for Quel1 box objects used by qubex backend flows."""

    boxtype: str

    def reconnect(self, *args: Any, **kwargs: Any) -> Any:
        """Reconnect links and return backend-specific status payload."""
        ...

    def relinkup(self, *args: Any, **kwargs: Any) -> Any:
        """Relink box JESD links with optional config options."""
        ...

    def link_status(self) -> dict[int, bool]:
        """Return link status by lane/group index."""
        ...

    def initialize_all_awgunits(self) -> None:
        """Initialize all AWG units."""
        ...

    def initialize_all_capunits(self) -> None:
        """Initialize all capture units."""
        ...

    def get_input_ports(self) -> Any:
        """Return iterable of input port identifiers."""
        ...

    def get_output_ports(self) -> Any:
        """Return iterable of output port identifiers."""
        ...

    def dump_box(self) -> dict[str, Any]:
        """Return current box configuration dump."""
        ...

    def dump_port(self, port: Any) -> dict[str, Any]:
        """Return current port configuration dump."""
        ...

    def config_port(self, *args: Any, **kwargs: Any) -> None:
        """Apply port-level configuration."""
        ...

    def config_channel(self, *args: Any, **kwargs: Any) -> None:
        """Apply channel-level configuration."""
        ...

    def config_runit(self, *args: Any, **kwargs: Any) -> None:
        """Apply runit-level configuration."""
        ...


class BoxPoolProtocol(Protocol):
    """Protocol for legacy-style box pool state and helpers."""

    _boxes: dict[str, tuple[Quel1BoxProtocol, Any]]
    _linkstatus: dict[str, bool]
    _box_config_cache: dict[str, dict[str, Any]]

    def create_clock_master(self, *, ipaddr: str) -> None:
        """Create clock master client for the pool."""
        ...

    def create(
        self,
        box_name: str,
        *,
        ipaddr_wss: str,
        ipaddr_sss: str,
        ipaddr_css: str,
        boxtype: Any,
    ) -> Quel1BoxProtocol:
        """Create and register one box instance."""
        ...


class Quel1SystemProtocol(Protocol):
    """Protocol for direct multi-box system objects used by qubex."""

    config_cache: dict[str, dict[str, Any]]
    config_fetched_at: Any

    @classmethod
    def create(
        cls,
        *,
        clockmaster: Any,
        boxes: list[Any],
        update_copnfig_cache: bool = False,
    ) -> Any:
        """Create a system from pre-constructed clockmaster/box objects."""
        ...


class AwgSettingProtocol(Protocol):
    """Marker protocol for driver AWG setting objects."""


class RunitSettingProtocol(Protocol):
    """Marker protocol for driver runit setting objects."""


class TriggerSettingProtocol(Protocol):
    """Marker protocol for driver trigger setting objects."""


class QubeCalibProtocol(Protocol):
    """Protocol for QubeCalib facade objects consumed by qubex."""

    _executor: Any

    def __init__(
        self, path_to_database_file: str | PathLike[str] | None = None
    ) -> None:
        """Create a QubeCalib session with optional config path."""
        ...

    @property
    def system_config_database(self) -> Any:
        """Return system configuration database object."""
        ...

    @property
    def sysdb(self) -> Any:
        """Return system configuration database alias."""
        ...

    @property
    def executor(self) -> Any:
        """Return queued-command executor."""
        ...

    def define_clockmaster(self, ipaddr: str, reset: bool) -> Any:
        """Define clock-master endpoint."""
        ...

    def define_box(
        self,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
        ipaddr_sss: str | None = None,
        ipaddr_css: str | None = None,
        config_root: str | None = None,
        config_options: Any = None,
    ) -> Any:
        """Define one box entry."""
        ...

    def define_port(self, *args: Any, **kwargs: Any) -> Any:
        """Define one port entry."""
        ...

    def define_channel(self, *args: Any, **kwargs: Any) -> Any:
        """Define one channel entry."""
        ...

    def define_target(self, *args: Any, **kwargs: Any) -> Any:
        """Define one target mapping."""
        ...

    def modify_target_frequency(self, target_name: str, frequency: float) -> None:
        """Update target frequency."""
        ...

    def show_command_queue(self) -> Any:
        """Show queued commands."""
        ...

    def clear_command_queue(self) -> None:
        """Clear queued commands."""
        ...

    def step_execute(
        self,
        repeats: int = 1,
        interval: float = 10240,
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
        """Execute queued commands step-by-step."""
        ...


class SequencerProtocol(Protocol):
    """Protocol for qube-calib sequencer compatibility objects."""

    interval: int | None
    cap_sampled_sequence: Mapping[str, Any]
    gen_sampled_sequence: Mapping[str, Any]

    def __init__(
        self,
        *,
        gen_sampled_sequence: dict[str, Any],
        cap_sampled_sequence: dict[str, Any],
        resource_map: dict[str, list[dict]],
        interval: int,
        sysdb: Any,
        driver: Any,
    ) -> None:
        """Create sequencer from sampled sequences and runtime objects."""
        ...

    def execute(
        self, boxpool: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Execute against the given box pool."""
        ...

    def select_trigger(self, system: Any, settings: list[Any]) -> list[Any]:
        """Select trigger settings for action execution."""
        ...

    def parse_capture_results(
        self,
        *,
        status: dict[str, Any],
        results: dict[str, Any],
        action: Any,
        crmap: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Parse low-level capture results into qubex-compatible payload."""
        ...

    def calc_first_padding(self) -> int:
        """Return first-padding length applied before waveform emission."""
        ...


class QuelDriverModulesProtocol(Protocol):
    """Protocol for driver-loader module bundle consumed by qubex."""

    package_name: str
    root_module: ModuleType
    clockmaster_module: ModuleType
    quel1_module: ModuleType
    driver_module: ModuleType
    tool_module: ModuleType
    neopulse_module: ModuleType
    qubecalib_module: ModuleType
    direct_multi_module: ModuleType
    direct_single_module: ModuleType
    QubeCalib: type[QubeCalibProtocol]
    Sequencer: type[SequencerProtocol]
    QuBEMasterClient: type[QuBEMasterClientProtocol]
    SequencerClient: type[SequencerClientProtocol]
    Quel1System: type[Quel1SystemProtocol]
    Action: Any
    AwgId: Any
    AwgSetting: type[AwgSettingProtocol]
    NamedBox: Any
    RunitId: Any
    RunitSetting: type[RunitSettingProtocol]
    TriggerSetting: type[TriggerSettingProtocol]
    Skew: Any
    DEFAULT_SAMPLING_PERIOD: Any
    CapSampledSequence: Any
    GenSampledSequence: Any
    BoxPool: type[BoxPoolProtocol]
    CaptureParamTools: Any
    Converter: Any
    WaveSequenceTools: Any
    Quel1Box: type[Quel1BoxProtocol]
    Quel1ConfigOption: type[Quel1ConfigOptionProtocol]
