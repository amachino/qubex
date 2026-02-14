"""Compatibility contract tests for QuEL driver APIs consumed by qubex."""

from __future__ import annotations

from qxdriver_quel import QubeCalib, Sequencer, neopulse
from qxdriver_quel.clockmaster_compat import QuBEMasterClient, SequencerClient
from qxdriver_quel.instrument.quel.quel1 import Quel1System
from qxdriver_quel.instrument.quel.quel1.driver import (
    Action,
    AwgId,
    AwgSetting,
    NamedBox,
    RunitId,
    RunitSetting,
    TriggerSetting,
    multi,
    single,
)
from qxdriver_quel.instrument.quel.quel1.tool import Skew
from qxdriver_quel.qubecalib import (
    BoxPool,
    CaptureParamTools,
    Converter,
    WaveSequenceTools,
)


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
    assert hasattr(qc, "executor")
    assert hasattr(qc, "_executor")


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

    assert hasattr(db, "clockmaster_setting")
    assert hasattr(db, "box_settings")
    assert hasattr(db, "port_settings")
    assert hasattr(db, "target_settings")
    assert hasattr(db, "relation_channel_target")

    assert hasattr(db, "_clockmaster_setting")
    assert hasattr(db, "_box_settings")
    assert hasattr(db, "_port_settings")
    assert hasattr(db, "_target_settings")
    assert hasattr(db, "_relation_channel_target")
