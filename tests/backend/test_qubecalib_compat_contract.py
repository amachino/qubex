"""Compatibility contract tests for QuEL driver APIs consumed by qubex."""

from __future__ import annotations

import importlib

try:
    _compat = importlib.import_module("qxdriver_quel.compat")
    _sampling_period = _compat.DEFAULT_SAMPLING_PERIOD
    QubeCalib = _compat.QubeCalib
    Sequencer = _compat.Sequencer
    QuBEMasterClient = _compat.QuBEMasterClient
    SequencerClient = _compat.SequencerClient
    Quel1System = _compat.Quel1System
    Action = _compat.Action
    SingleAction = _compat.SingleAction
    MultiAction = _compat.MultiAction
    AwgId = _compat.AwgId
    AwgSetting = _compat.AwgSetting
    NamedBox = _compat.NamedBox
    RunitId = _compat.RunitId
    RunitSetting = _compat.RunitSetting
    SingleAwgId = _compat.SingleAwgId
    SingleAwgSetting = _compat.SingleAwgSetting
    SingleRunitId = _compat.SingleRunitId
    SingleRunitSetting = _compat.SingleRunitSetting
    SingleTriggerSetting = _compat.SingleTriggerSetting
    TriggerSetting = _compat.TriggerSetting
    single = importlib.import_module(_compat.SingleAction.__module__)
    multi = importlib.import_module(_compat.MultiAction.__module__)
    Skew = _compat.Skew
    BoxPool = _compat.BoxPool
    Converter = _compat.Converter
    CaptureParamTools = _compat.CaptureParamTools
    WaveSequenceTools = _compat.WaveSequenceTools
except ModuleNotFoundError:
    _root = importlib.import_module("qubecalib")
    try:
        _clockmaster = importlib.import_module("qubecalib.clockmaster_compat")
    except ModuleNotFoundError:
        _clockmaster = importlib.import_module("qubecalib.qubecalib")
    _quel1 = importlib.import_module("qubecalib.instrument.quel.quel1")
    _driver = importlib.import_module("qubecalib.instrument.quel.quel1.driver")
    _tool = importlib.import_module("qubecalib.instrument.quel.quel1.tool")
    _qcalib = importlib.import_module("qubecalib.qubecalib")
    _sampling_period = _root.neopulse.DEFAULT_SAMPLING_PERIOD
    QubeCalib = _root.QubeCalib
    Sequencer = _root.Sequencer
    QuBEMasterClient = _clockmaster.QuBEMasterClient
    SequencerClient = _clockmaster.SequencerClient
    Quel1System = _quel1.Quel1System
    Action = _driver.Action
    SingleAction = _driver.single.Action
    MultiAction = _driver.multi.Action
    AwgId = _driver.AwgId
    AwgSetting = _driver.AwgSetting
    NamedBox = _driver.NamedBox
    RunitId = _driver.RunitId
    RunitSetting = _driver.RunitSetting
    SingleAwgId = _driver.single.AwgId
    SingleAwgSetting = _driver.single.AwgSetting
    SingleRunitId = _driver.single.RunitId
    SingleRunitSetting = _driver.single.RunitSetting
    SingleTriggerSetting = _driver.single.TriggerSetting
    TriggerSetting = _driver.TriggerSetting
    single = _driver.single
    multi = _driver.multi
    Skew = _tool.Skew
    BoxPool = _qcalib.BoxPool
    Converter = _qcalib.Converter
    CaptureParamTools = _qcalib.CaptureParamTools
    WaveSequenceTools = _qcalib.WaveSequenceTools


def test_qubecalib_import_paths_required_by_qubex_are_available() -> None:
    """Given qubex dependencies, when importing qubecalib symbols, then all required paths resolve."""
    assert QubeCalib.__name__ == "QubeCalib"
    assert Sequencer.__name__ == "Sequencer"
    assert _sampling_period is not None

    assert QuBEMasterClient.__name__ == "QuBEMasterClient"
    assert SequencerClient.__name__ == "SequencerClient"
    assert Quel1System.__name__ == "Quel1System"
    assert Skew.__name__ == "Skew"

    assert Action.__name__ == "Action"
    assert AwgId.__name__ == "AwgId"
    assert AwgSetting.__name__ == "AwgSetting"
    assert NamedBox.__name__ == "NamedBox"
    assert RunitId.__name__ == "RunitId"
    assert RunitSetting.__name__ == "RunitSetting"
    assert SingleAwgId.__name__ == "AwgId"
    assert SingleAwgSetting.__name__ == "AwgSetting"
    assert SingleRunitId.__name__ == "RunitId"
    assert SingleRunitSetting.__name__ == "RunitSetting"
    assert SingleTriggerSetting.__name__ == "TriggerSetting"
    assert TriggerSetting.__name__ == "TriggerSetting"
    assert hasattr(single, "Action")
    assert hasattr(multi, "Action")
    assert SingleAction is single.Action
    assert MultiAction is multi.Action

    assert Action is not SingleAction
    assert Action is not MultiAction
    assert SingleAction is not MultiAction
    assert AwgId is not SingleAwgId
    assert AwgSetting is not SingleAwgSetting
    assert RunitId is not SingleRunitId
    assert RunitSetting is not SingleRunitSetting
    assert TriggerSetting is not SingleTriggerSetting

    assert BoxPool.__name__ == "BoxPool"
    assert Converter.__name__ == "Converter"
    assert CaptureParamTools.__name__ == "CaptureParamTools"
    assert WaveSequenceTools.__name__ == "WaveSequenceTools"


def test_qubecalib_instance_exposes_methods_required_by_qubex() -> None:
    """Given a QubeCalib instance, when checking core operations, then qubex-required methods are callable."""
    qc = QubeCalib()

    assert callable(qc.define_clockmaster)
    assert callable(qc.define_box)
    assert callable(qc.define_port)
    assert callable(qc.define_channel)
    assert callable(qc.define_target)
    assert callable(qc.modify_target_frequency)
    assert callable(qc.show_command_queue)
    assert callable(qc.clear_command_queue)
    assert callable(qc.step_execute)
    assert hasattr(qc, "_executor")
    assert hasattr(qc._executor, "add_command")  # noqa: SLF001


def test_system_config_database_and_sysdb_contract_required_by_qubex() -> None:
    """Given qubex DB access patterns, system DB exposes the legacy private contract."""
    qc = QubeCalib()
    db = qc.system_config_database

    assert qc.sysdb is db

    assert callable(db.asdict)
    assert callable(db.asjson)
    assert callable(db.get_channels_by_target)
    assert callable(db.get_channel)
    assert callable(db.create_box)
    assert callable(db.load_skew_yaml)

    assert hasattr(db, "_clockmaster_setting")
    assert hasattr(db, "_box_settings")
    assert hasattr(db, "_port_settings")
    assert hasattr(db, "_target_settings")
    assert hasattr(db, "_relation_channel_target")


def test_boxpool_contract_required_by_qubex() -> None:
    """Given qubex runtime access patterns, BoxPool exposes required private fields."""
    boxpool = BoxPool()

    assert hasattr(boxpool, "_boxes")
    assert hasattr(boxpool, "_linkstatus")
    assert hasattr(boxpool, "_box_config_cache")
