"""Protocols for qubecalib backward compatibility symbols."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from os import PathLike
from typing import Any, ClassVar, Protocol, TypeAlias

from typing_extensions import Self

PortType = int | tuple[int, int]
StatusMap = dict[str, Any]
DataMap = dict[str, Any]
ConfigMap = dict[str, Any]
CaptureStatusKey: TypeAlias = tuple[str, PortType]
CaptureResultKey: TypeAlias = tuple[str, PortType, int]
RawCaptureStatusMap: TypeAlias = Mapping[CaptureStatusKey, Any]
RawCaptureResultsMap: TypeAlias = Mapping[CaptureResultKey, Any]
CaptureParamMap: TypeAlias = Mapping[CaptureResultKey, Any]
CaptureResourceEntry: TypeAlias = Mapping[str, Any]
CaptureResourceMap: TypeAlias = Mapping[str, CaptureResourceEntry]


class ActionProtocol(Protocol):
    """Protocol for action class symbols and their built instances."""

    _action: Any

    @classmethod
    def build(
        cls,
        *,
        system: Quel1SystemProtocol,
        settings: list[
            RunitSettingProtocol | AwgSettingProtocol | TriggerSettingProtocol
        ],
    ) -> ActionProtocol:
        """Build an action Any from driver settings."""
        ...

    def action(
        self,
    ) -> tuple[
        dict[CaptureStatusKey, Any],
        dict[CaptureResultKey, Any],
    ]:
        """Execute action and return status/data maps."""
        ...


class AwgIdProtocol(Protocol):
    """Protocol for AWG identifier objects."""

    box: str
    port: PortType
    channel: int

    def __init__(self, box: str, port: PortType, channel: int) -> None:
        """Create an AWG identifier."""
        ...


class AwgSettingProtocol(Protocol):
    """Protocol for driver AWG setting objects."""

    awg: AwgIdProtocol
    wseq: Any

    def __init__(self, awg: AwgIdProtocol, wseq: Any) -> None:
        """Create one AWG setting."""
        ...


class SingleAwgIdProtocol(Protocol):
    """Protocol for single-box AWG identifier objects."""

    port: PortType
    channel: int

    def __init__(self, port: PortType, channel: int) -> None:
        """Create a single-box AWG identifier."""
        ...


class SingleAwgSettingProtocol(Protocol):
    """Protocol for single-box AWG setting objects."""

    awg: SingleAwgIdProtocol
    wseq: Any

    def __init__(self, awg: SingleAwgIdProtocol, wseq: Any) -> None:
        """Create one single-box AWG setting."""
        ...


class BoxPoolProtocol(Protocol):
    """Protocol for legacy-style box pool state and helpers."""

    _boxes: dict[str, tuple[Quel1BoxCommonProtocol, SequencerClientProtocol]]
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
        boxtype: str,
    ) -> Quel1BoxCommonProtocol:
        """Create and register one box instance."""
        ...

    def get_box(
        self,
        box_name: str,
    ) -> tuple[Quel1BoxCommonProtocol, SequencerClientProtocol]:
        """Return box and sequencer-client pair for one registered box."""
        ...

    def get_port_direction(
        self,
        box_name: str,
        port: PortType,
    ) -> str:
        """Return cached port direction (`in` or `out`)."""
        ...

    def ensure_box_config_cache(
        self,
        *,
        box_name: str,
        box: Quel1BoxCommonProtocol,
    ) -> dict[str, Any]:
        """Ensure per-box dump cache and return cached box payload."""
        ...


class BoxSettingProtocol(Protocol):
    """Protocol for one box setting row in system config database."""

    box_name: str
    boxtype: str
    ipaddr_wss: Any
    ipaddr_sss: Any
    ipaddr_css: Any


class CapSampledSequenceProtocol(Protocol):
    """Protocol for capture sampled-sequence objects."""

    modulation_frequency: float | None


class CapSampledSubSequenceProtocol(Protocol):
    """Protocol for capture sub-sequence objects."""

    capture_slots: list[CaptureSlotsProtocol]


class CaptureParamToolsProtocol(Protocol):
    """Protocol for capture-parameter helper class symbols."""

    @classmethod
    def create(
        cls,
        *,
        sequence: CapSampledSequenceProtocol,
        capture_delay_words: int,
        repeats: int,
        interval_samples: int,
    ) -> Any:
        """Create one capture parameter Any."""
        ...

    @classmethod
    def enable_integration(cls, *, capprm: Any) -> None:
        """Enable integration mode on capture parameters."""
        ...

    @classmethod
    def enable_demodulation(cls, *, capprm: Any, f_GHz: float) -> None:
        """Enable DSP demodulation with the given IF frequency."""
        ...


class CaptureSlotsProtocol(Protocol):
    """Protocol for one capture slot entry."""

    def __init__(
        self,
        *,
        duration: int,
        post_blank: int,
        original_duration: int | None,
        original_post_blank: int | None,
    ) -> None:
        """Create one capture slot."""
        ...


class ClockmasterSettingProtocol(Protocol):
    """Protocol for clockmaster setting row in system config database."""

    ipaddr: Any


class ConverterProtocol(Protocol):
    """Protocol for waveform conversion helper class symbols."""

    @classmethod
    def multiplex(
        cls,
        *,
        sequences: Mapping[str, GenSampledSequenceProtocol],
        modfreqs: Mapping[str, float],
    ) -> Any:
        """Multiplex generator sequences into one waveform sequence."""
        ...

    @classmethod
    def convert_to_cap_device_specific_sequence(
        cls,
        *,
        gen_sampled_sequence: Mapping[str, GenSampledSequenceProtocol],
        cap_sampled_sequence: Mapping[str, CapSampledSequenceProtocol],
        resource_map: Mapping[str, Mapping[str, Any]],
        port_config: Mapping[str, PortConfigAcquirerProtocol],
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
        enable_sum: bool,
        enable_classification: bool = False,
        line_param0: tuple[float, float, float] = (1.0, 0.0, 0.0),
        line_param1: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> dict[tuple[str, PortType, int], Any]:
        """Convert capture sampled sequences into per-runit capture params."""
        ...

    @classmethod
    def convert_to_gen_device_specific_sequence(
        cls,
        *,
        gen_sampled_sequence: Mapping[str, GenSampledSequenceProtocol],
        cap_sampled_sequence: Mapping[str, CapSampledSequenceProtocol],
        resource_map: Mapping[str, Mapping[str, Any]],
        port_config: Mapping[str, PortConfigAcquirerProtocol],
        repeats: int,
        interval: float,
    ) -> dict[tuple[str, PortType, int], Any]:
        """Convert generation sampled sequences into per-awg wave sequences."""
        ...


class ExecutorProtocol(Protocol):
    """Protocol for queued command executor used by qubecalib."""

    def add_command(self, command: Any) -> None:
        """Append one command to queue."""
        ...


class GenSampledSequenceProtocol(Protocol):
    """Protocol for generation sampled-sequence objects."""

    modulation_frequency: float | None


class GenSampledSubSequenceProtocol(Protocol):
    """Protocol for generation sub-sequence objects."""


class MultiActionProtocol(Protocol):
    """Protocol for multi-action class symbols."""

    @classmethod
    def _mod_by_sysref(cls, t: int) -> int:
        """Map raw clock counter into SYSREF-period local offset."""
        ...

    @classmethod
    def _get_reference_box_name(cls, actions: Mapping[str, Any]) -> str:
        """Return the reference box name used for timing alignment."""
        ...

    @classmethod
    def _measure_average_offset_at_sysref_clock(cls, box: Any) -> int:
        """Measure average SYSREF offset for one box."""
        ...


class NamedBoxProtocol(Protocol):
    """Protocol for named-box wrapper objects."""

    name: str
    box: Quel1BoxCommonProtocol

    def __init__(self, *, name: str, box: Quel1BoxCommonProtocol) -> None:
        """Create a named box wrapper."""
        ...


class PortSettingProtocol(Protocol):
    """Protocol for one port setting row in system config database."""

    port: PortType


class PortConfigAcquirerProtocol(Protocol):
    """Protocol for port-config snapshots used during sequence conversion."""

    dump_config: dict[str, Any]
    lo_freq: float | None
    cnco_freq: float
    fnco_freq: float
    sideband: str | None
    box_name: str
    port: PortType
    channel: int

    def __init__(
        self,
        boxpool: BoxPoolProtocol,
        box_name: str,
        box: Quel1BoxCommonProtocol,
        port: PortType,
        channel: int,
        *,
        driver: Quel1SystemProtocol | None = None,
    ) -> None:
        """Capture effective port settings for one target channel mapping."""
        ...


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

    def kick_clock_synch(self, box_sss_ipaddrs: Sequence[str]) -> None:
        """Trigger clock synchronization for the given SSS endpoints."""
        ...

    def read_clock(self) -> tuple[bool, int]:
        """Read clock-master counter value."""
        ...

    def reset(self) -> bool:
        """Reset clock master when supported by the backend."""
        ...


class QubeCalibProtocol(Protocol):
    """Protocol for QubeCalib facade objects consumed by qubex."""

    _executor: ExecutorProtocol

    def __init__(
        self, path_to_database_file: str | PathLike[str] | None = None
    ) -> None:
        """Create a QubeCalib session with optional config path."""
        ...

    @property
    def system_config_database(self) -> SystemConfigDatabaseProtocol:
        """Return system configuration database Any."""
        ...

    @property
    def sysdb(self) -> SysdbProtocol:
        """Return system configuration database alias."""
        ...

    @property
    def executor(self) -> ExecutorProtocol:
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
        config_options: Sequence[Quel1ConfigOptionProtocol] | None = None,
    ) -> Any:
        """Define one box entry."""
        ...

    def define_port(
        self,
        *,
        port_name: str,
        box_name: str,
        port_number: int,
    ) -> Any:
        """Define one port entry."""
        ...

    def define_channel(
        self,
        *,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> Any:
        """Define one channel entry."""
        ...

    def define_target(
        self,
        *,
        target_name: str,
        channel_name: str,
        target_frequency: float | None = None,
    ) -> Any:
        """Define one target mapping."""
        ...

    def modify_target_frequency(self, target_name: str, frequency: float) -> None:
        """Update target frequency."""
        ...

    def show_command_queue(self) -> str:
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
    ) -> Iterator[tuple[StatusMap, DataMap, ConfigMap]]:
        """Execute queued commands step-by-step."""
        ...


class Quel1BoxCommonProtocol(Protocol):
    """Protocol for APIs shared by Quel1Box and Quel1BoxWithRawWss."""

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

    def get_input_ports(self) -> Sequence[PortType]:
        """Return iterable of input port identifiers."""
        ...

    def get_output_ports(self) -> Sequence[PortType]:
        """Return iterable of output port identifiers."""
        ...

    def dump_box(self) -> dict[str, Any]:
        """Return current box configuration dump."""
        ...

    def dump_port(self, port: PortType) -> dict[str, Any]:
        """Return current port configuration dump."""
        ...

    def config_port(
        self,
        *,
        port: PortType,
        lo_freq: float | None = None,
        cnco_freq: float | None = None,
        vatt: int | None = None,
        sideband: str | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
    ) -> None:
        """Apply port-level configuration."""
        ...

    def config_channel(
        self,
        *,
        port: PortType,
        channel: int,
        fnco_freq: float | None = None,
    ) -> None:
        """Apply channel-level configuration."""
        ...

    def config_runit(
        self,
        *,
        port: PortType,
        runit: int,
        fnco_freq: float | None = None,
    ) -> None:
        """Apply runit-level configuration."""
        ...


class Quel1BoxProtocol(Quel1BoxCommonProtocol, Protocol):
    """Protocol for box APIs additionally required by parallel action builder."""

    def get_current_timecounter(self) -> int:
        """Return current box timecounter."""
        ...

    def get_latest_sysref_timecounter(self) -> int:
        """Return latest latched SYSREF timecounter."""
        ...

    def start_wavegen(
        self,
        channels: set[tuple[PortType, int]],
        timecounter: int | None = None,
    ) -> Any:
        """Start wave generation immediately or at reserved time."""
        ...


class Quel1ConfigOptionProtocol(Protocol):
    """Protocol for Quel1 config option enum-like classes."""

    _value2member_map_: ClassVar[Mapping[str, Any]]


class Quel1SystemProtocol(Protocol):
    """Protocol for multi-box system objects used by qubex."""

    boxes: Mapping[str, Quel1BoxProtocol]
    box: Mapping[str, Quel1BoxProtocol]
    _clockmaster: QuBEMasterClientProtocol
    timing_shift: dict[str, int]
    displacement: int
    config_cache: dict[str, dict[str, Any]]
    config_fetched_at: datetime | None

    @classmethod
    def create(
        cls,
        *,
        clockmaster: QuBEMasterClientProtocol,
        boxes: Sequence[NamedBoxProtocol],
        update_copnfig_cache: bool = False,
    ) -> Quel1SystemProtocol:
        """Create a system from pre-constructed clockmaster/box objects."""
        ...


class QuelDriverClassesProtocol(Protocol):
    """Protocol for driver-loader class bundle consumed by qubex."""

    package_name: str
    DEFAULT_SAMPLING_PERIOD: float | int
    Action: type[ActionProtocol]
    AwgId: type[AwgIdProtocol]
    AwgSetting: type[AwgSettingProtocol]
    BoxPool: type[BoxPoolProtocol]
    CapSampledSequence: type[CapSampledSequenceProtocol]
    CapSampledSubSequence: type[CapSampledSubSequenceProtocol]
    CaptureParamTools: type[CaptureParamToolsProtocol]
    CaptureSlots: type[CaptureSlotsProtocol]
    Converter: type[ConverterProtocol]
    GenSampledSequence: type[GenSampledSequenceProtocol]
    GenSampledSubSequence: type[GenSampledSubSequenceProtocol]
    MultiAction: type[MultiActionProtocol]
    NamedBox: type[NamedBoxProtocol]
    QuBEMasterClient: type[QuBEMasterClientProtocol]
    QubeCalib: type[QubeCalibProtocol]
    Quel1Box: type[Quel1BoxCommonProtocol]
    Quel1ConfigOption: type[Quel1ConfigOptionProtocol]
    Quel1System: type[Quel1SystemProtocol]
    RunitId: type[RunitIdProtocol]
    RunitSetting: type[RunitSettingProtocol]
    Sequencer: type[SequencerProtocol]
    SequencerClient: type[SequencerClientProtocol]
    SingleAction: type[SingleActionProtocol]
    SingleAwgId: type[SingleAwgIdProtocol]
    SingleAwgSetting: type[SingleAwgSettingProtocol]
    SingleRunitId: type[SingleRunitIdProtocol]
    SingleRunitSetting: type[SingleRunitSettingProtocol]
    SingleTriggerSetting: type[SingleTriggerSettingProtocol]
    Skew: type[SkewProtocol]
    TriggerSetting: type[TriggerSettingProtocol]
    WaveSequenceTools: type[WaveSequenceToolsProtocol]


class RunitIdProtocol(Protocol):
    """Protocol for runit identifier objects."""

    box: str
    port: PortType
    runit: int

    def __init__(self, box: str, port: PortType, runit: int) -> None:
        """Create a runit identifier."""
        ...


class RunitSettingProtocol(Protocol):
    """Protocol for driver runit setting objects."""

    runit: RunitIdProtocol
    cprm: Any

    def __init__(self, runit: RunitIdProtocol, cprm: Any) -> None:
        """Create one runit setting."""
        ...


class SingleRunitIdProtocol(Protocol):
    """Protocol for single-box runit identifier objects."""

    port: PortType
    runit: int

    def __init__(self, port: PortType, runit: int) -> None:
        """Create a single-box runit identifier."""
        ...


class SingleRunitSettingProtocol(Protocol):
    """Protocol for single-box runit setting objects."""

    runit: SingleRunitIdProtocol
    cprm: Any

    def __init__(self, runit: SingleRunitIdProtocol, cprm: Any) -> None:
        """Create one single-box runit setting."""
        ...


class SequencerClientProtocol(Protocol):
    """Protocol for sequencer clock readers."""

    def __init__(self, target_ipaddr: str, *, box: Any | None = None) -> None:
        """Create a sequencer client."""
        ...

    def read_clock(self) -> tuple[bool, int, int]:
        """Read current and SYSREF counters."""
        ...


class SequencerProtocol(Protocol):
    """Protocol for qube-calib sequencer compatibility objects."""

    interval: int | None
    resource_map: Mapping[str, Sequence[dict[str, Any]]]
    cap_sampled_sequence: Mapping[str, CapSampledSequenceProtocol]
    gen_sampled_sequence: Mapping[str, GenSampledSequenceProtocol]
    repeats: int
    integral_mode: str
    dsp_demodulation: bool
    software_demodulation: bool
    enable_sum: bool
    enable_classification: bool
    line_param0: tuple[float, float, float]
    line_param1: tuple[float, float, float]
    driver: Quel1SystemProtocol | None

    def __init__(
        self,
        *,
        gen_sampled_sequence: dict[str, GenSampledSequenceProtocol],
        cap_sampled_sequence: dict[str, CapSampledSequenceProtocol],
        resource_map: dict[str, list[dict[str, Any]]],
        interval: int,
        sysdb: SysdbProtocol,
        driver: Quel1SystemProtocol | None = None,
    ) -> None:
        """Create sequencer from sampled sequences and runtime objects."""
        ...

    def execute(self, boxpool: BoxPoolProtocol) -> tuple[StatusMap, DataMap, ConfigMap]:
        """Execute against the given box pool."""
        ...

    def set_measurement_option(self, **kwargs: Any) -> None:
        """Apply measurement options before execution."""
        ...

    def generate_e7_settings(
        self,
        boxpool: BoxPoolProtocol,
    ) -> tuple[RawCaptureResultsMap, RawCaptureResultsMap, CaptureResourceMap]:
        """Generate driver capture and waveform settings."""
        ...

    def select_trigger(
        self,
        system: Quel1SystemProtocol,
        settings: list[
            RunitSettingProtocol | AwgSettingProtocol | TriggerSettingProtocol
        ],
    ) -> list[TriggerSettingProtocol]:
        """Select trigger settings for action execution."""
        ...

    def parse_capture_results(
        self,
        *,
        status: RawCaptureStatusMap,
        results: RawCaptureResultsMap,
        action: ActionProtocol,
        crmap: CaptureResourceMap,
    ) -> tuple[StatusMap, DataMap, ConfigMap]:
        """Parse low-level capture results into qubex-compatible payload."""
        ...

    def parse_capture_result(
        self,
        status: Any,
        data: Any,
        cprm: Any,
    ) -> tuple[Any, Any]:
        """Parse one backend capture payload with one capture parameter."""
        ...

    def calc_first_padding(self) -> int:
        """Return first-padding length applied before waveform emission."""
        ...


class SingleActionProtocol(Protocol):
    """Protocol for single-action class symbols and their built instances."""

    _cprms: Mapping[SingleRunitIdProtocol, Any]
    _wseqs: Mapping[Any, Any]
    _triggers: Mapping[Any, Any]
    box: Quel1BoxProtocol

    @classmethod
    def build(
        cls,
        *,
        box: Quel1BoxCommonProtocol,
        settings: list[
            SingleRunitSettingProtocol
            | SingleAwgSettingProtocol
            | SingleTriggerSettingProtocol
        ],
    ) -> Self:
        """Build one single-box action from settings."""
        ...

    def capture_start(self) -> dict[PortType, Any]:
        """Start capture and return future map."""
        ...

    def capture_stop(
        self,
        futures: dict[PortType, Any],
    ) -> tuple[dict[PortType, Any], dict[tuple[PortType, int], Any]]:
        """Stop capture and collect status/data."""
        ...


class SkewProtocol(Protocol):
    """Protocol for skew tool class symbols."""

    @classmethod
    def from_yaml(
        cls,
        path: str,
        *,
        box_yaml: str,
        clockmaster_ip: str,
        boxes: Sequence[str],
    ) -> SkewRuntimeProtocol:
        """Create skew Any from YAML and connectivity settings."""
        ...


class SkewRuntimeProtocol(Protocol):
    """Protocol for skew runtime Any returned from `Skew.from_yaml`."""

    system: SkewSystemProtocol

    def measure(self) -> None:
        """Run skew measurement."""
        ...

    def estimate(self) -> None:
        """Estimate skew model from measurements."""
        ...

    def plot(self) -> Any:
        """Return plotly figure Any."""
        ...


class SkewSystemProtocol(Protocol):
    """Protocol for skew-associated system Any."""

    def resync(self) -> None:
        """Resynchronize device clocks."""
        ...


class SysdbProtocol(Protocol):
    """Protocol for low-level sysdb helpers used by qubex."""

    _relation_channel_target: list[tuple[str, str]]

    def load_skew_yaml(self, path: str) -> None:
        """Load skew YAML file into sysdb."""
        ...


class SystemConfigDatabaseProtocol(Protocol):
    """Protocol for qubecalib system configuration database Any."""

    _target_settings: dict[str, Any]
    _box_settings: dict[str, BoxSettingProtocol]
    _port_settings: dict[str, PortSettingProtocol]
    _clockmaster_setting: ClockmasterSettingProtocol | None

    def asdict(self) -> dict[str, Any]:
        """Return whole configuration as dictionary."""
        ...

    def asjson(self) -> str:
        """Return whole configuration as JSON."""
        ...

    def get_channels_by_target(self, target: str) -> list[str]:
        """Return channel names linked to the target."""
        ...

    def get_channel(self, channel_name: str) -> tuple[str, str, int]:
        """Return `(box_name, port_name, channel_number)` for one channel."""
        ...

    def create_box(
        self, box_name: str, *, reconnect: bool = True
    ) -> Quel1BoxCommonProtocol:
        """Create one box Any from database settings."""
        ...


class TriggerSettingProtocol(Protocol):
    """Protocol for driver trigger setting objects."""

    trigger_awg: AwgIdProtocol
    triggerd_port: PortType

    def __init__(self, trigger_awg: AwgIdProtocol, triggerd_port: PortType) -> None:
        """Create one trigger setting."""
        ...


class SingleTriggerSettingProtocol(Protocol):
    """Protocol for single-box trigger setting objects."""

    trigger_awg: SingleAwgIdProtocol
    triggerd_port: PortType

    def __init__(
        self,
        trigger_awg: SingleAwgIdProtocol,
        triggerd_port: PortType,
    ) -> None:
        """Create one single-box trigger setting."""
        ...


class WaveSequenceToolsProtocol(Protocol):
    """Protocol for waveform-sequence helper class symbols."""

    @classmethod
    def create(
        cls,
        *,
        sequence: Any,
        wait_words: int,
        repeats: int,
        interval_samples: int,
    ) -> Any:
        """Create one wave-sequence Any from multiplexed samples."""
        ...
