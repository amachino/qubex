"""Compatibility contract tests for QuEL driver APIs consumed by qubex."""

from __future__ import annotations

import importlib

try:
    _root = importlib.import_module("qxdriver_quel")
    _clockmaster = importlib.import_module("qxdriver_quel.clockmaster_compat")
    _quel1 = importlib.import_module("qxdriver_quel.instrument.quel.quel1")
    _driver = importlib.import_module("qxdriver_quel.instrument.quel.quel1.driver")
    _tool = importlib.import_module("qxdriver_quel.instrument.quel.quel1.tool")
    _qcalib = importlib.import_module("qxdriver_quel.qubecalib")
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

QubeCalib = _root.QubeCalib
Sequencer = _root.Sequencer
neopulse = _root.neopulse
QuBEMasterClient = _clockmaster.QuBEMasterClient
SequencerClient = _clockmaster.SequencerClient
Quel1System = _quel1.Quel1System
Action = _driver.Action
AwgId = _driver.AwgId
AwgSetting = _driver.AwgSetting
NamedBox = _driver.NamedBox
RunitId = _driver.RunitId
RunitSetting = _driver.RunitSetting
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
    assert hasattr(neopulse, "DEFAULT_SAMPLING_PERIOD")

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
    assert TriggerSetting.__name__ == "TriggerSetting"
    assert hasattr(single, "Action")
    assert hasattr(multi, "Action")

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
    assert hasattr(qc, "executor") or hasattr(qc, "_executor")


def test_system_config_database_and_sysdb_contract_required_by_qubex() -> None:
    """Given qubex DB access patterns, when inspecting sysdb, then public and fallback attributes exist."""
    qc = QubeCalib()
    db = qc.system_config_database

    assert qc.sysdb is db

    assert callable(db.asdict)
    assert callable(db.asjson)
    assert callable(db.get_channels_by_target)
    assert callable(db.get_channel)
    assert callable(db.create_box)
    assert callable(db.load_skew_yaml)
    assert callable(db.assign_target_to_channel)

    assert hasattr(db, "clockmaster_setting") or hasattr(db, "_clockmaster_setting")
    assert hasattr(db, "box_settings") or hasattr(db, "_box_settings")
    assert hasattr(db, "port_settings") or hasattr(db, "_port_settings")
    assert hasattr(db, "target_settings") or hasattr(db, "_target_settings")
    assert hasattr(db, "relation_channel_target") or hasattr(
        db, "_relation_channel_target"
    )
