"""Tests for QuEL driver package loader and fallback behavior."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel1 import driver_loader


def _fake_class(name: str, module: str) -> type:
    """Create a synthetic class bound to a target module path."""
    return type(name, (), {"__module__": module})


def _build_fake_driver_modules(
    package_name: str,
    *,
    include_facade: bool = False,
    include_compat: bool = False,
) -> dict[str, ModuleType]:
    """Create fake modules for one driver package namespace."""
    root = cast(Any, ModuleType(package_name))
    root.QubeCalib = _fake_class("QubeCalib", package_name)
    root.Sequencer = _fake_class("Sequencer", package_name)

    clockmaster = cast(Any, ModuleType(f"{package_name}.clockmaster_compat"))
    clockmaster.QuBEMasterClient = _fake_class(
        "QuBEMasterClient", f"{package_name}.clockmaster_compat"
    )
    clockmaster.SequencerClient = _fake_class(
        "SequencerClient", f"{package_name}.clockmaster_compat"
    )

    quel1 = cast(Any, ModuleType(f"{package_name}.instrument.quel.quel1"))
    quel1.Quel1System = _fake_class(
        "Quel1System", f"{package_name}.instrument.quel.quel1"
    )

    direct = cast(Any, ModuleType(f"{package_name}.instrument.quel.quel1.driver"))
    direct.Action = _fake_class(
        "Action", f"{package_name}.instrument.quel.quel1.driver"
    )
    direct.AwgId = _fake_class("AwgId", f"{package_name}.instrument.quel.quel1.driver")
    direct.AwgSetting = _fake_class(
        "AwgSetting", f"{package_name}.instrument.quel.quel1.driver"
    )
    direct.NamedBox = _fake_class(
        "NamedBox", f"{package_name}.instrument.quel.quel1.driver"
    )
    direct.RunitId = _fake_class(
        "RunitId", f"{package_name}.instrument.quel.quel1.driver"
    )
    direct.RunitSetting = _fake_class(
        "RunitSetting", f"{package_name}.instrument.quel.quel1.driver"
    )
    direct.TriggerSetting = _fake_class(
        "TriggerSetting", f"{package_name}.instrument.quel.quel1.driver"
    )

    tool = cast(Any, ModuleType(f"{package_name}.instrument.quel.quel1.tool"))
    tool.Skew = _fake_class("Skew", f"{package_name}.instrument.quel.quel1.tool")

    neopulse = cast(Any, ModuleType(f"{package_name}.neopulse"))
    neopulse.DEFAULT_SAMPLING_PERIOD = 2
    neopulse.CapSampledSequence = _fake_class(
        "CapSampledSequence", f"{package_name}.neopulse"
    )
    neopulse.CapSampledSubSequence = _fake_class(
        "CapSampledSubSequence", f"{package_name}.neopulse"
    )
    neopulse.CaptureSlots = _fake_class("CaptureSlots", f"{package_name}.neopulse")
    neopulse.GenSampledSequence = _fake_class(
        "GenSampledSequence", f"{package_name}.neopulse"
    )
    neopulse.GenSampledSubSequence = _fake_class(
        "GenSampledSubSequence", f"{package_name}.neopulse"
    )

    legacy = cast(Any, ModuleType(f"{package_name}.qubecalib"))
    legacy.BoxPool = _fake_class("BoxPool", f"{package_name}.qubecalib")
    legacy.CaptureParamTools = _fake_class(
        "CaptureParamTools", f"{package_name}.qubecalib"
    )
    legacy.Converter = _fake_class("Converter", f"{package_name}.qubecalib")
    legacy.WaveSequenceTools = _fake_class(
        "WaveSequenceTools", f"{package_name}.qubecalib"
    )
    legacy.QuBEMasterClient = clockmaster.QuBEMasterClient
    legacy.SequencerClient = clockmaster.SequencerClient
    legacy.Quel1Box = _fake_class("Quel1Box", f"{package_name}.qubecalib")
    legacy.Quel1ConfigOption = _fake_class(
        "Quel1ConfigOption", f"{package_name}.qubecalib"
    )

    multi = cast(Any, ModuleType(f"{package_name}.instrument.quel.quel1.driver.multi"))
    multi.Action = _fake_class(
        "Action", f"{package_name}.instrument.quel.quel1.driver.multi"
    )

    single = cast(
        Any, ModuleType(f"{package_name}.instrument.quel.quel1.driver.single")
    )
    single.Action = _fake_class(
        "Action", f"{package_name}.instrument.quel.quel1.driver.single"
    )

    mapping = {
        package_name: root,
        f"{package_name}.clockmaster_compat": clockmaster,
        f"{package_name}.instrument.quel.quel1": quel1,
        f"{package_name}.instrument.quel.quel1.driver": direct,
        f"{package_name}.instrument.quel.quel1.tool": tool,
        f"{package_name}.neopulse": neopulse,
        f"{package_name}.qubecalib": legacy,
        f"{package_name}.instrument.quel.quel1.driver.multi": multi,
        f"{package_name}.instrument.quel.quel1.driver.single": single,
    }

    if include_facade:
        facade = cast(Any, ModuleType(f"{package_name}.facade"))
        facade.QuBEMasterClient = clockmaster.QuBEMasterClient
        facade.SequencerClient = clockmaster.SequencerClient
        facade.BoxPool = legacy.BoxPool
        facade.CaptureParamTools = legacy.CaptureParamTools
        facade.Converter = legacy.Converter
        facade.WaveSequenceTools = legacy.WaveSequenceTools
        facade.Quel1Box = legacy.Quel1Box
        facade.Quel1ConfigOption = legacy.Quel1ConfigOption
        facade.direct = SimpleNamespace(
            Action=direct.Action,
            AwgId=direct.AwgId,
            AwgSetting=direct.AwgSetting,
            NamedBox=direct.NamedBox,
            RunitId=direct.RunitId,
            RunitSetting=direct.RunitSetting,
            TriggerSetting=direct.TriggerSetting,
            Quel1System=quel1.Quel1System,
            multi=SimpleNamespace(Action=multi.Action),
            single=SimpleNamespace(Action=single.Action),
        )
        mapping[f"{package_name}.facade"] = facade

    if include_compat:
        compat = cast(Any, ModuleType(f"{package_name}.compat"))
        compat.QubeCalib = root.QubeCalib
        compat.Sequencer = root.Sequencer
        compat.QuBEMasterClient = clockmaster.QuBEMasterClient
        compat.SequencerClient = clockmaster.SequencerClient
        compat.Quel1System = quel1.Quel1System
        compat.Action = direct.Action
        compat.AwgId = direct.AwgId
        compat.AwgSetting = direct.AwgSetting
        compat.NamedBox = direct.NamedBox
        compat.RunitId = direct.RunitId
        compat.RunitSetting = direct.RunitSetting
        compat.TriggerSetting = direct.TriggerSetting
        compat.Skew = tool.Skew
        compat.DEFAULT_SAMPLING_PERIOD = neopulse.DEFAULT_SAMPLING_PERIOD
        compat.CapSampledSequence = neopulse.CapSampledSequence
        compat.CapSampledSubSequence = neopulse.CapSampledSubSequence
        compat.CaptureSlots = neopulse.CaptureSlots
        compat.GenSampledSequence = neopulse.GenSampledSequence
        compat.GenSampledSubSequence = neopulse.GenSampledSubSequence
        compat.BoxPool = legacy.BoxPool
        compat.CaptureParamTools = legacy.CaptureParamTools
        compat.Converter = legacy.Converter
        compat.WaveSequenceTools = legacy.WaveSequenceTools
        compat.Quel1Box = legacy.Quel1Box
        compat.Quel1ConfigOption = legacy.Quel1ConfigOption
        compat.DirectMultiAction = multi.Action
        compat.DirectSingleAction = single.Action
        mapping[f"{package_name}.compat"] = compat

    return mapping


def test_load_quel_driver_rejects_invalid_preference() -> None:
    """Given invalid preference, when loading driver, then ValueError is raised."""
    driver_loader.clear_quel_driver_cache()

    with pytest.raises(ValueError, match="Invalid QUBEX_QUEL_DRIVER"):
        driver_loader.load_quel_driver("invalid")


def test_load_quel_driver_respects_explicit_preference(monkeypatch) -> None:
    """Given explicit preference, when loading driver, then selected package is returned."""
    mapping = _build_fake_driver_modules("qxdriver_quel")

    def _fake_import(name: str) -> ModuleType:
        if name in mapping:
            return mapping[name]
        raise ModuleNotFoundError(name)

    driver_loader.clear_quel_driver_cache()
    monkeypatch.setattr(driver_loader.importlib, "import_module", _fake_import)

    modules = driver_loader.load_quel_driver("qxdriver_quel")

    assert modules.package_name == "qxdriver_quel"
    assert modules.QubeCalib.__name__ == "QubeCalib"
    assert modules.Action.__name__ == "Action"


def test_load_quel_driver_auto_falls_back_to_qubecalib(monkeypatch) -> None:
    """Given qxdriver_quel import failure, when loading auto mode, then qubecalib is selected."""
    mapping = _build_fake_driver_modules("qubecalib")

    def _fake_import(name: str) -> ModuleType:
        if name.startswith("qxdriver_quel"):
            raise ModuleNotFoundError(name)
        if name in mapping:
            return mapping[name]
        raise ModuleNotFoundError(name)

    driver_loader.clear_quel_driver_cache()
    monkeypatch.setattr(driver_loader.importlib, "import_module", _fake_import)

    modules = driver_loader.load_quel_driver("auto")

    assert modules.package_name == "qubecalib"


def test_load_quel_driver_uses_env_preference(monkeypatch) -> None:
    """Given env preference, when no explicit preference is passed, then env value is used."""
    mapping = _build_fake_driver_modules("qubecalib")

    def _fake_import(name: str) -> ModuleType:
        if name in mapping:
            return mapping[name]
        raise ModuleNotFoundError(name)

    driver_loader.clear_quel_driver_cache()
    monkeypatch.setenv("QUBEX_QUEL_DRIVER", "qubecalib")
    monkeypatch.setattr(driver_loader.importlib, "import_module", _fake_import)

    modules = driver_loader.load_quel_driver()

    assert modules.package_name == "qubecalib"


def test_load_quel_driver_qubecalib_resolves_clockmaster_from_qubecalib_module(
    monkeypatch,
) -> None:
    """Given missing clockmaster_compat, loader resolves clock symbols from qubecalib package paths."""
    mapping = _build_fake_driver_modules("qubecalib")
    del mapping["qubecalib.clockmaster_compat"]

    def _fake_import(name: str) -> ModuleType:
        if name in mapping:
            return mapping[name]
        raise ModuleNotFoundError(name)

    driver_loader.clear_quel_driver_cache()
    monkeypatch.setattr(driver_loader.importlib, "import_module", _fake_import)

    modules = driver_loader.load_quel_driver("qubecalib")

    assert modules.package_name == "qubecalib"
    assert modules.QuBEMasterClient.__name__ == "QuBEMasterClient"
    assert modules.SequencerClient.__name__ == "SequencerClient"


def test_load_quel_driver_does_not_fall_back_to_quel_clock_master(monkeypatch) -> None:
    """Given no package clock symbols, direct quel_clock_master fallback is not used."""
    mapping = _build_fake_driver_modules("qubecalib")
    del mapping["qubecalib.clockmaster_compat"]
    legacy = cast(Any, mapping["qubecalib.qubecalib"])
    del legacy.QuBEMasterClient
    del legacy.SequencerClient

    quel_clock_master = cast(Any, ModuleType("quel_clock_master"))
    quel_clock_master.QuBEMasterClient = _fake_class(
        "QuBEMasterClient", "quel_clock_master"
    )
    quel_clock_master.SequencerClient = _fake_class(
        "SequencerClient", "quel_clock_master"
    )
    mapping["quel_clock_master"] = quel_clock_master

    def _fake_import(name: str) -> ModuleType:
        if name in mapping:
            return mapping[name]
        raise ModuleNotFoundError(name)

    driver_loader.clear_quel_driver_cache()
    monkeypatch.setattr(driver_loader.importlib, "import_module", _fake_import)

    with pytest.raises(
        ModuleNotFoundError,
        match=r"Could not resolve symbol 'QuBEMasterClient' for package 'qubecalib'\.",
    ):
        driver_loader.load_quel_driver("qubecalib")


def test_load_quel_driver_resolves_symbols_from_facade_fallback(monkeypatch) -> None:
    """Given alternate class exports, when loading driver, then symbol mapping does not require legacy modules."""
    mapping = _build_fake_driver_modules("qxdriver_quel", include_facade=True)
    del mapping["qxdriver_quel.clockmaster_compat"]
    del mapping["qxdriver_quel.qubecalib"]

    def _fake_import(name: str) -> ModuleType:
        if name in mapping:
            return mapping[name]
        raise ModuleNotFoundError(name)

    driver_loader.clear_quel_driver_cache()
    monkeypatch.setattr(driver_loader.importlib, "import_module", _fake_import)

    modules = driver_loader.load_quel_driver("qxdriver_quel")

    assert modules.package_name == "qxdriver_quel"
    assert modules.QuBEMasterClient.__name__ == "QuBEMasterClient"
    assert modules.BoxPool.__name__ == "BoxPool"


def test_load_quel_driver_resolves_symbols_from_compat_layer(monkeypatch) -> None:
    """Given compat exports, when loading driver, then required symbols resolve without legacy module paths."""
    mapping = _build_fake_driver_modules("qxdriver_quel", include_compat=True)
    del mapping["qxdriver_quel.clockmaster_compat"]
    del mapping["qxdriver_quel.instrument.quel.quel1.driver"]
    del mapping["qxdriver_quel.instrument.quel.quel1.driver.multi"]
    del mapping["qxdriver_quel.instrument.quel.quel1.driver.single"]
    del mapping["qxdriver_quel.instrument.quel.quel1.tool"]
    del mapping["qxdriver_quel.qubecalib"]

    def _fake_import(name: str) -> ModuleType:
        if name in mapping:
            return mapping[name]
        raise ModuleNotFoundError(name)

    driver_loader.clear_quel_driver_cache()
    monkeypatch.setattr(driver_loader.importlib, "import_module", _fake_import)

    modules = driver_loader.load_quel_driver("qxdriver_quel")

    assert modules.package_name == "qxdriver_quel"
    assert modules.Action.__name__ == "Action"
    assert modules.Skew.__name__ == "Skew"
    assert modules.BoxPool.__name__ == "BoxPool"
    assert hasattr(modules.neopulse_module, "GenSampledSubSequence")
    assert hasattr(modules.neopulse_module, "CapSampledSubSequence")
    assert hasattr(modules.neopulse_module, "CaptureSlots")
