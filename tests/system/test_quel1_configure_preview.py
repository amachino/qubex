"""Tests for QuEL-1 configure preview behavior."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from quel_ic_config import QUEL1_BOXTYPE_ALIAS

from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext
from qubex.system import ConfigurePreview, ConfigureStateChange
from qubex.system.control_system import BoxType, PortType
from qubex.system.quel1.quel1_configure_preview import Quel1BoxPreviewContext
from qubex.system.quel1.quel1_system_synchronizer import Quel1SystemSynchronizer


class _HardwareQuel1Box:
    snapshots_by_ip: ClassVar[dict[str, dict]] = {}
    config_calls: ClassVar[list[str]] = []

    def __init__(self, *, ipaddr_wss: str, ipaddr_sss: str, boxtype: str) -> None:
        self._state = deepcopy(self.snapshots_by_ip[ipaddr_wss])
        self.boxtype = boxtype
        self.wss = SimpleNamespace(ipaddr_sss=ipaddr_sss)

    @classmethod
    def create(
        cls,
        *,
        ipaddr_wss: str,
        ipaddr_sss: str,
        boxtype: str,
        **_: object,
    ) -> _HardwareQuel1Box:
        return cls(
            ipaddr_wss=ipaddr_wss,
            ipaddr_sss=ipaddr_sss,
            boxtype=boxtype,
        )

    def reconnect(self, **_: object) -> dict[int, bool]:
        return {}

    def dump_box(self) -> dict:
        return deepcopy(self._state)

    def config_port(self, **_: object) -> None:
        self.config_calls.append("port")

    def config_channel(self, **_: object) -> None:
        self.config_calls.append("channel")

    def config_runit(self, **_: object) -> None:
        self.config_calls.append("runit")


Quel1Box = _HardwareQuel1Box


class _SystemConfigDatabaseStub:
    def __init__(self, *, box_type: BoxType) -> None:
        self._box_settings = {
            "A": SimpleNamespace(
                ipaddr_wss="192.0.2.1",
                ipaddr_sss="192.0.2.2",
                ipaddr_css="192.0.2.3",
                boxtype=QUEL1_BOXTYPE_ALIAS[box_type.value],
            )
        }
        self.created_box_classes: list[type[object]] = []

    def asdict(self) -> dict[str, object]:
        return {"box_settings": self._box_settings}

    def create_box(
        self,
        box_name: str,
        reconnect: bool = True,
    ) -> object:
        del reconnect
        setting = self._box_settings[box_name]
        self.created_box_classes.append(Quel1Box)
        return Quel1Box.create(
            ipaddr_wss=setting.ipaddr_wss,
            ipaddr_sss=setting.ipaddr_sss,
            ipaddr_css=setting.ipaddr_css,
            boxtype=setting.boxtype,
            skip_init=False,
        )


class _ExperimentSystemStub:
    def __init__(self, boxes: list[Any]) -> None:
        self._boxes = {box.id: box for box in boxes}

    def get_box(self, box_id: str) -> Any:
        return self._boxes[box_id]

    @property
    def hash(self) -> int:
        return 0


def _make_system(
    *,
    box_type: BoxType = BoxType.QUEL1SE_R8,
    port_number: int = 1,
    port_type: PortType = PortType.CTRL,
    lo_freq: int | None = 10_000_000_000,
    cnco_freq: int = 1_500_000_000,
    fnco_freq: int | None = 100_000_000,
    rfswitch: str = "pass",
    vatt: int | None = 2048,
    sideband: str | None = "L",
    fullscale_current: int | None = 40527,
) -> _ExperimentSystemStub:
    channel = SimpleNamespace(number=0, fnco_freq=fnco_freq)
    port = SimpleNamespace(
        number=port_number,
        type=port_type,
        lo_freq=lo_freq,
        cnco_freq=cnco_freq,
        vatt=vatt,
        sideband=sideband,
        fullscale_current=fullscale_current,
        rfswitch=rfswitch,
        channels=(channel,),
    )
    return _ExperimentSystemStub(
        [SimpleNamespace(id="A", name="Alpha", type=box_type, ports=(port,))]
    )


def _backend_settings(
    *,
    port_number: int = 1,
    lo_freq: int | None = 10_000_000_000,
    cnco_freq: int = 1_500_000_000,
    fnco_freq: int = 100_000_000,
    rfswitch: str = "pass",
    vatt: int | None = 2048,
    sideband: str | None = "L",
    fullscale_current: int | None = 40527,
) -> dict[str, dict]:
    return {
        "A": {
            "ports": {
                port_number: {
                    "direction": "out",
                    "lo_freq": lo_freq,
                    "cnco_freq": cnco_freq,
                    "vatt": vatt,
                    "sideband": sideband,
                    "fullscale_current": fullscale_current,
                    "rfswitch": rfswitch,
                    "channels": {0: {"fnco_freq": fnco_freq}},
                }
            }
        }
    }


def _make_port(
    *,
    number: int,
    port_type: PortType,
    lo_freq: int | None = None,
    rfswitch: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        number=number,
        type=port_type,
        lo_freq=lo_freq,
        cnco_freq=None,
        vatt=None,
        sideband=None,
        fullscale_current=None,
        rfswitch=rfswitch,
        channels=(),
    )


def _make_shared_system(
    *,
    box_type: BoxType,
    ports: list[SimpleNamespace],
) -> _ExperimentSystemStub:
    return _ExperimentSystemStub(
        [
            SimpleNamespace(
                id="A",
                name="Alpha",
                type=box_type,
                ports=tuple(sorted(ports, key=lambda port: port.number)),
            )
        ]
    )


def _shared_backend_settings(
    *port_configs: tuple[int | tuple[int, int], str, int | None, str | None],
    include_rfswitch: bool = True,
) -> dict[str, dict]:
    ports: dict[int | tuple[int, int], dict[str, object]] = {}
    for port_number, direction, lo_freq, rfswitch in port_configs:
        config: dict[str, object] = {
            "direction": direction,
            "lo_freq": lo_freq,
        }
        if include_rfswitch:
            config["rfswitch"] = rfswitch
        config["channels" if direction == "out" else "runits"] = {}
        ports[port_number] = config
    return {"A": {"ports": ports}}


def _preview(
    *,
    experiment_system: _ExperimentSystemStub,
    backend_settings: dict[str, dict],
) -> ConfigurePreview:
    backend_controller, _ = _make_backend_controller(
        backend_settings=backend_settings,
        box_type=experiment_system.get_box("A").type,
    )
    synchronizer = Quel1SystemSynchronizer(backend_controller=backend_controller)
    return synchronizer.preview_configure(
        experiment_system=cast(Any, experiment_system),
        box_ids=["A"],
        mode="ge-cr-cr",
        parallel=False,
    )


def _make_backend_controller(
    *,
    backend_settings: dict[str, dict],
    box_type: BoxType,
    connected: bool = True,
) -> tuple[Quel1BackendController, _SystemConfigDatabaseStub]:
    database = _SystemConfigDatabaseStub(box_type=box_type)
    _HardwareQuel1Box.snapshots_by_ip = {
        "192.0.2.1": deepcopy(backend_settings.get("A", {}))
    }
    _HardwareQuel1Box.config_calls = []
    driver = SimpleNamespace(
        DEFAULT_SAMPLING_PERIOD=2.0,
        Quel1Box=_HardwareQuel1Box,
    )
    qubecalib = SimpleNamespace(system_config_database=database)
    runtime_context = Quel1RuntimeContext(
        driver=cast(Any, driver),
        qubecalib=cast(Any, qubecalib),
    )
    if connected:
        hardware_box = _HardwareQuel1Box.create(
            ipaddr_wss="192.0.2.1",
            ipaddr_sss="192.0.2.2",
            boxtype=box_type.value,
        )
        runtime_context.set_connected_state(
            boxpool=cast(
                Any,
                SimpleNamespace(_boxes={"A": (hardware_box, object())}),
            ),
            quel1system=cast(Any, object()),
            cap_resource_map={},
            gen_resource_map={},
        )
    return Quel1BackendController(runtime_context=runtime_context), database


def test_preview_configure_routes_configuration_away_from_hardware() -> None:
    """Preview should configure only a mock of an already-connected box."""
    system = _make_system(lo_freq=11_000_000_000)
    backend_settings = _backend_settings(lo_freq=10_000_000_000)
    backend_controller, database = _make_backend_controller(
        backend_settings=backend_settings,
        box_type=BoxType.QUEL1SE_R8,
    )
    synchronizer = Quel1SystemSynchronizer(backend_controller=backend_controller)

    preview = synchronizer.preview_configure(
        experiment_system=cast(Any, system),
        box_ids=["A"],
        mode="ge-cr-cr",
        parallel=False,
    )

    assert database.created_box_classes == []
    assert Quel1Box is _HardwareQuel1Box
    assert _HardwareQuel1Box.config_calls == []
    assert preview.changes[0].after == 11_000_000_000


def test_preview_configure_rejects_disconnected_before_hardware_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview should fail before fetching hardware when the backend is disconnected."""
    system = _make_system()
    backend_controller, _ = _make_backend_controller(
        backend_settings=_backend_settings(),
        box_type=BoxType.QUEL1SE_R8,
        connected=False,
    )
    dump_calls: list[str] = []

    def _dump_box(box_id: str) -> dict:
        dump_calls.append(box_id)
        return _backend_settings()[box_id]

    monkeypatch.setattr(backend_controller, "dump_box", _dump_box)
    synchronizer = Quel1SystemSynchronizer(backend_controller=backend_controller)

    with pytest.raises(RuntimeError, match="requires all target boxes to be connected"):
        synchronizer.preview_configure(
            experiment_system=cast(Any, system),
            box_ids=["A"],
            mode="ge-cr-cr",
            parallel=False,
        )

    assert dump_calls == []


def test_preview_configure_rejects_box_missing_from_pool_before_hardware_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview should fail before fetching a target absent from the connected pool."""
    system = _make_system()
    backend_controller, _ = _make_backend_controller(
        backend_settings=_backend_settings(),
        box_type=BoxType.QUEL1SE_R8,
    )
    pooled_boxes = vars(backend_controller.boxpool)["_boxes"]
    pooled_boxes.clear()
    dump_calls: list[str] = []

    def _dump_box(box_id: str) -> dict:
        dump_calls.append(box_id)
        return _backend_settings()[box_id]

    monkeypatch.setattr(backend_controller, "dump_box", _dump_box)
    synchronizer = Quel1SystemSynchronizer(backend_controller=backend_controller)

    with pytest.raises(RuntimeError, match="missing from the connected pool: A"):
        synchronizer.preview_configure(
            experiment_system=cast(Any, system),
            box_ids=["A"],
            mode="ge-cr-cr",
            parallel=False,
        )

    assert dump_calls == []


def test_preview_configure_uses_hardware_sync_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview should execute the same system-to-hardware orchestration as push."""
    system = _make_system(lo_freq=11_000_000_000)
    backend_controller, _ = _make_backend_controller(
        backend_settings=_backend_settings(lo_freq=10_000_000_000),
        box_type=BoxType.QUEL1SE_R8,
    )
    synchronizer = Quel1SystemSynchronizer(backend_controller=backend_controller)
    sync_calls: list[tuple[list[str], bool | None, tuple[str, ...] | None]] = []
    original_sync = synchronizer.sync_experiment_system_to_hardware

    def _record_sync(**kwargs: Any) -> None:
        sync_calls.append(
            (
                [box.id for box in kwargs["boxes"]],
                kwargs["parallel"],
                tuple(kwargs["target_labels"]),
            )
        )
        original_sync(**kwargs)

    monkeypatch.setattr(
        synchronizer,
        "sync_experiment_system_to_hardware",
        _record_sync,
    )

    synchronizer.preview_configure(
        experiment_system=cast(Any, system),
        box_ids=["A"],
        mode="ge-cr-cr",
        parallel=False,
        target_labels=["Q00"],
    )

    assert sync_calls == [(["A"], False, ("Q00",))]


def test_preview_configure_restores_connected_boxpool() -> None:
    """Preview should restore connected boxpool entries after mock configuration."""
    system = _make_system(lo_freq=11_000_000_000)
    backend_controller, _ = _make_backend_controller(
        backend_settings=_backend_settings(lo_freq=10_000_000_000),
        box_type=BoxType.QUEL1SE_R8,
        connected=True,
    )
    boxpool = cast(Any, backend_controller.boxpool)
    pooled_boxes = vars(boxpool)["_boxes"]
    original_box = pooled_boxes["A"][0]
    synchronizer = Quel1SystemSynchronizer(backend_controller=backend_controller)

    preview = synchronizer.preview_configure(
        experiment_system=cast(Any, system),
        box_ids=["A"],
        mode="ge-cr-cr",
        parallel=False,
    )

    assert pooled_boxes["A"][0] is original_box
    assert _HardwareQuel1Box.config_calls == []
    assert preview.changes[0].after == 11_000_000_000


def test_preview_context_restores_quel1_box_after_exception() -> None:
    """Preview context should restore the real box class after an exception."""
    backend_controller, _ = _make_backend_controller(
        backend_settings=_backend_settings(),
        box_type=BoxType.QUEL1SE_R8,
    )
    context = Quel1BoxPreviewContext(
        backend_controller=backend_controller,
        backend_settings=_backend_settings(),
        box_types={"A": BoxType.QUEL1SE_R8},
    )

    def _raise_inside_context() -> None:
        with context:
            assert Quel1Box is not _HardwareQuel1Box
            raise RuntimeError("stop preview")

    with pytest.raises(RuntimeError, match="stop preview"):
        _raise_inside_context()

    assert Quel1Box is _HardwareQuel1Box


def test_preview_configure_reports_no_changes() -> None:
    """Given matching hardware and config, preview should report no changes."""
    preview = _preview(
        experiment_system=_make_system(),
        backend_settings=_backend_settings(),
    )

    assert preview.is_complete is True
    assert preview.has_changes is False
    assert preview.has_frequency_changes is False
    assert preview.changes == ()
    assert len(preview.entries) > 0
    assert all(not entry.has_change for entry in preview.entries)


def test_preview_configure_detects_frequency_changes() -> None:
    """Given changed LO frequency, preview should mark frequency changes."""
    preview = _preview(
        experiment_system=_make_system(lo_freq=11_000_000_000),
        backend_settings=_backend_settings(lo_freq=10_000_000_000),
    )

    assert preview.has_changes is True
    assert preview.has_frequency_changes is True
    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="lo_freq",
            before=10_000_000_000,
            after=11_000_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_before_uses_fetched_dump() -> None:
    """Given changed CNCO frequency, preview before should use fetched hardware state."""
    preview = _preview(
        experiment_system=_make_system(cnco_freq=2_109_375_000),
        backend_settings=_backend_settings(cnco_freq=2_320_312_500),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="cnco_freq",
            before=2_320_312_500,
            after=2_109_375_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_detects_non_frequency_changes() -> None:
    """Given changed RF switch, preview should not mark frequency changes."""
    preview = _preview(
        experiment_system=_make_system(rfswitch="pass"),
        backend_settings=_backend_settings(rfswitch="block"),
    )

    assert preview.has_changes is True
    assert preview.has_frequency_changes is False
    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="rfswitch",
            before="block",
            after="pass",
            unit=None,
            is_frequency=False,
        ),
    )


def test_preview_configure_uses_effective_r8_generator_port_values() -> None:
    """Given R8 non-mixer CTRL port VATT, preview should not report backend-ignored values."""
    preview = _preview(
        experiment_system=_make_system(
            port_number=6,
            lo_freq=10_000_000_000,
            vatt=3072,
            sideband="L",
            fnco_freq=0,
        ),
        backend_settings=_backend_settings(
            port_number=6,
            lo_freq=None,
            vatt=None,
            sideband=None,
            fnco_freq=0,
        ),
    )

    assert preview.changes == ()


def test_preview_configure_ignores_unspecified_fnco() -> None:
    """Given planned FNCO is unspecified, preview should not show zero-to-blank changes."""
    preview = _preview(
        experiment_system=_make_system(fnco_freq=None),
        backend_settings=_backend_settings(fnco_freq=0),
    )

    assert preview.changes == ()


def test_preview_configure_ignores_unspecified_port_fields() -> None:
    """Given a planned port field is unspecified, preview should preserve hardware."""
    preview = _preview(
        experiment_system=_make_system(
            box_type=BoxType.QUEL1SE_A,
            lo_freq=None,
        ),
        backend_settings=_backend_settings(lo_freq=10_000_000_000),
    )

    assert preview.changes == ()


def test_preview_configure_marks_missing_fetch_incomplete() -> None:
    """Given missing hardware fetch result, preview should be incomplete."""
    preview = _preview(
        experiment_system=_make_system(),
        backend_settings={},
    )

    assert preview.is_complete is False
    assert preview.missing_box_ids == ("A",)
    assert preview.changes == ()


@pytest.mark.parametrize(
    ("box_type", "capture_port_number", "generator_port_number"),
    [
        (BoxType.QUEL1SE_A, 0, 1),
        (BoxType.QUEL1SE_A, 5, 3),
        (BoxType.QUEL1SE_A, 7, 8),
        (BoxType.QUEL1SE_A, 12, 10),
        (BoxType.QUEL1SE_B, 5, 3),
        (BoxType.QUEL1SE_B, 12, 10),
        (BoxType.QUEL1_A, 0, 1),
        (BoxType.QUEL1_A, 5, 3),
        (BoxType.QUEL1_A, 7, 8),
        (BoxType.QUEL1_A, 12, 10),
        (BoxType.QUEL1_B, 5, 2),
        (BoxType.QUEL1_B, 12, 9),
        (BoxType.QUEL1SE_R8, 0, 1),
        (BoxType.QUBE_RIKEN_A, 1, 0),
        (BoxType.QUBE_RIKEN_A, 4, 2),
        (BoxType.QUBE_RIKEN_A, 12, 13),
        (BoxType.QUBE_RIKEN_A, 9, 11),
        (BoxType.QUBE_RIKEN_B, 4, 2),
        (BoxType.QUBE_RIKEN_B, 9, 11),
        (BoxType.QUBE_OU_A, 1, 0),
        (BoxType.QUBE_OU_A, 12, 13),
    ],
)
def test_preview_configure_uses_final_generator_lo_for_shared_resource(
    box_type: BoxType,
    capture_port_number: int,
    generator_port_number: int,
) -> None:
    """Given a shared LO restored by a generator, preview should use its final value."""
    system = _make_shared_system(
        box_type=box_type,
        ports=[
            _make_port(
                number=capture_port_number,
                port_type=PortType.READ_IN,
                lo_freq=9_000_000_000,
            ),
            _make_port(
                number=generator_port_number,
                port_type=PortType.CTRL,
                lo_freq=11_000_000_000,
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (capture_port_number, "in", 11_000_000_000, None),
            (generator_port_number, "out", 11_000_000_000, None),
        ),
    )

    assert preview.changes == ()
    assert preview.has_frequency_changes is False


def test_preview_configure_uses_last_capture_lo_for_shared_r8_resource() -> None:
    """Given R8 monitor ports sharing an LO, preview should use the last capture write."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1SE_R8,
        ports=[
            _make_port(
                number=4,
                port_type=PortType.MNTR_IN,
                lo_freq=9_000_000_000,
            ),
            _make_port(
                number=10,
                port_type=PortType.MNTR_IN,
                lo_freq=11_000_000_000,
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (4, "in", 11_000_000_000, None),
            (10, "in", 11_000_000_000, None),
        ),
    )

    assert preview.changes == ()


def test_preview_configure_reports_final_shared_lo_for_each_port() -> None:
    """Given a changing shared LO, preview should retain rows for every logical port."""
    system = _make_shared_system(
        box_type=BoxType.QUBE_RIKEN_B,
        ports=[
            _make_port(
                number=2,
                port_type=PortType.CTRL,
                lo_freq=11_000_000_000,
            ),
            _make_port(
                number=4,
                port_type=PortType.MNTR_IN,
                lo_freq=None,
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (2, "out", 10_000_000_000, None),
            (4, "in", 10_000_000_000, None),
        ),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 2",
            field="lo_freq",
            before=10_000_000_000,
            after=11_000_000_000,
            unit="Hz",
            is_frequency=True,
        ),
        ConfigureStateChange(
            box_id="A",
            component="port 4",
            field="lo_freq",
            before=10_000_000_000,
            after=11_000_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_normalizes_shared_rfswitch_values() -> None:
    """Given a restored shared RF switch, preview should compare its final state."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1SE_A,
        ports=[
            _make_port(
                number=0,
                port_type=PortType.READ_IN,
                rfswitch="loop",
            ),
            _make_port(
                number=1,
                port_type=PortType.READ_OUT,
                rfswitch="pass",
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (0, "in", None, "open"),
            (1, "out", None, "pass"),
        ),
    )

    assert preview.changes == ()


def test_preview_configure_reports_final_shared_rfswitch_for_each_port() -> None:
    """Given a changing shared RF switch, preview should decode each logical port."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1SE_A,
        ports=[
            _make_port(
                number=0,
                port_type=PortType.READ_IN,
                rfswitch="open",
            ),
            _make_port(
                number=1,
                port_type=PortType.READ_OUT,
                rfswitch="block",
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (0, "in", None, "open"),
            (1, "out", None, "pass"),
        ),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 0",
            field="rfswitch",
            before="open",
            after="loop",
            unit=None,
            is_frequency=False,
        ),
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="rfswitch",
            before="pass",
            after="block",
            unit=None,
            is_frequency=False,
        ),
    )


def test_preview_configure_reports_r8_fogi_shared_rfswitch_change() -> None:
    """R8 preview should report the FOGI port affected by a readout switch."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1SE_R8,
        ports=[
            _make_port(
                number=0,
                port_type=PortType.READ_IN,
                rfswitch="open",
            ),
            _make_port(
                number=1,
                port_type=PortType.READ_OUT,
                rfswitch="pass",
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (0, "in", None, "loop"),
            (1, "out", None, "block"),
            ((1, 1), "out", None, "block"),
        ),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 0",
            field="rfswitch",
            before="loop",
            after="open",
            unit=None,
            is_frequency=False,
        ),
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="rfswitch",
            before="block",
            after="pass",
            unit=None,
            is_frequency=False,
        ),
        ConfigureStateChange(
            box_id="A",
            component="port (1, 1)",
            field="rfswitch",
            before="block",
            after="pass",
            unit=None,
            is_frequency=False,
        ),
    )


def test_preview_configure_ignores_unreported_rfswitch_state() -> None:
    """Given a dump without RF switches, preview should not infer their changes."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1_A,
        ports=[
            _make_port(
                number=0,
                port_type=PortType.READ_IN,
                rfswitch="open",
            ),
            _make_port(
                number=1,
                port_type=PortType.READ_OUT,
                rfswitch="pass",
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (0, "in", None, None),
            (1, "out", None, None),
            include_rfswitch=False,
        ),
    )

    assert preview.changes == ()
