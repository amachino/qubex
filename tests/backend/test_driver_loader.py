"""Tests for QuEL driver package loader and fallback behavior."""

from __future__ import annotations

from types import ModuleType
from typing import Any, cast

import pytest

from qubex.backend.quel1 import driver_loader


def _build_fake_driver_modules(package_name: str) -> dict[str, ModuleType]:
    """Create fake modules for one driver package namespace."""
    root = cast(Any, ModuleType(package_name))
    root.QubeCalib = type("QubeCalib", (), {})
    root.Sequencer = type("Sequencer", (), {})

    clockmaster = cast(Any, ModuleType(f"{package_name}.clockmaster_compat"))
    clockmaster.QuBEMasterClient = type("QuBEMasterClient", (), {})
    clockmaster.SequencerClient = type("SequencerClient", (), {})

    quel1 = cast(Any, ModuleType(f"{package_name}.instrument.quel.quel1"))
    quel1.Quel1System = type("Quel1System", (), {})

    direct = cast(Any, ModuleType(f"{package_name}.instrument.quel.quel1.driver"))
    direct.Action = type("Action", (), {})
    direct.AwgId = type("AwgId", (), {})
    direct.AwgSetting = type("AwgSetting", (), {})
    direct.NamedBox = type("NamedBox", (), {})
    direct.RunitId = type("RunitId", (), {})
    direct.RunitSetting = type("RunitSetting", (), {})
    direct.TriggerSetting = type("TriggerSetting", (), {})

    tool = cast(Any, ModuleType(f"{package_name}.instrument.quel.quel1.tool"))
    tool.Skew = type("Skew", (), {})

    neopulse = cast(Any, ModuleType(f"{package_name}.neopulse"))
    neopulse.DEFAULT_SAMPLING_PERIOD = 2
    neopulse.CapSampledSequence = type("CapSampledSequence", (), {})
    neopulse.GenSampledSequence = type("GenSampledSequence", (), {})

    legacy = cast(Any, ModuleType(f"{package_name}.qubecalib"))
    legacy.BoxPool = type("BoxPool", (), {})
    legacy.CaptureParamTools = type("CaptureParamTools", (), {})
    legacy.Converter = type("Converter", (), {})
    legacy.WaveSequenceTools = type("WaveSequenceTools", (), {})

    multi = cast(Any, ModuleType(f"{package_name}.instrument.quel.quel1.driver.multi"))
    multi.Action = type("Action", (), {})

    single = cast(
        Any, ModuleType(f"{package_name}.instrument.quel.quel1.driver.single")
    )
    single.Action = type("Action", (), {})

    return {
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


def test_load_quel_driver_qubecalib_falls_back_to_legacy_clockmaster(
    monkeypatch,
) -> None:
    """Given old qubecalib layout, when clockmaster_compat is missing, then legacy module is used."""
    mapping = _build_fake_driver_modules("qubecalib")
    legacy_clockmaster = cast(Any, ModuleType("quel_clock_master"))
    legacy_clockmaster.QuBEMasterClient = type("QuBEMasterClient", (), {})
    legacy_clockmaster.SequencerClient = type("SequencerClient", (), {})
    mapping["quel_clock_master"] = legacy_clockmaster

    def _fake_import(name: str) -> ModuleType:
        if name == "qubecalib.clockmaster_compat":
            raise ModuleNotFoundError(name)
        if name in mapping:
            return mapping[name]
        raise ModuleNotFoundError(name)

    driver_loader.clear_quel_driver_cache()
    monkeypatch.setattr(driver_loader.importlib, "import_module", _fake_import)

    modules = driver_loader.load_quel_driver("qubecalib")

    assert modules.package_name == "qubecalib"
    assert modules.QuBEMasterClient.__name__ == "QuBEMasterClient"
    assert modules.SequencerClient.__name__ == "SequencerClient"
