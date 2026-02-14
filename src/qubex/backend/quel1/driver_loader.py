"""Loader for selecting QuEL driver package with compatibility fallback."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any

_DRIVER_ENV_VAR = "QUBEX_QUEL_DRIVER"
_PREFERENCE_AUTO = "auto"
_PREFERENCE_QXDRIVER = "qxdriver_quel"
_PREFERENCE_QUBECALIB = "qubecalib"
_VALID_PREFERENCES = {
    _PREFERENCE_AUTO,
    _PREFERENCE_QXDRIVER,
    _PREFERENCE_QUBECALIB,
}


@dataclass(frozen=True)
class QuelDriverModules:
    """Resolved module and symbol set used by QuEL backend runtime."""

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
    QubeCalib: Any
    Sequencer: Any
    QuBEMasterClient: Any
    SequencerClient: Any
    Quel1System: Any
    Action: Any
    AwgId: Any
    AwgSetting: Any
    NamedBox: Any
    RunitId: Any
    RunitSetting: Any
    TriggerSetting: Any
    Skew: Any
    DEFAULT_SAMPLING_PERIOD: Any
    CapSampledSequence: Any
    GenSampledSequence: Any
    BoxPool: Any
    CaptureParamTools: Any
    Converter: Any
    WaveSequenceTools: Any


def _normalize_preference(preference: str | None) -> str:
    """Resolve and validate the current driver preference label."""
    resolved = preference or os.getenv(_DRIVER_ENV_VAR, _PREFERENCE_AUTO)
    if resolved not in _VALID_PREFERENCES:
        raise ValueError(
            f"Invalid {_DRIVER_ENV_VAR}='{resolved}'. "
            f"Expected one of {sorted(_VALID_PREFERENCES)}."
        )
    return resolved


def _import_driver_package(package_name: str) -> QuelDriverModules:
    """Import one driver package and return required backend-facing symbols."""
    root_module = importlib.import_module(package_name)
    clockmaster_module = importlib.import_module(f"{package_name}.clockmaster_compat")
    quel1_module = importlib.import_module(f"{package_name}.instrument.quel.quel1")
    driver_module = importlib.import_module(
        f"{package_name}.instrument.quel.quel1.driver"
    )
    tool_module = importlib.import_module(f"{package_name}.instrument.quel.quel1.tool")
    neopulse_module = importlib.import_module(f"{package_name}.neopulse")
    qubecalib_module = importlib.import_module(f"{package_name}.qubecalib")
    direct_multi_module = importlib.import_module(
        f"{package_name}.instrument.quel.quel1.driver.multi"
    )
    direct_single_module = importlib.import_module(
        f"{package_name}.instrument.quel.quel1.driver.single"
    )

    return QuelDriverModules(
        package_name=package_name,
        root_module=root_module,
        clockmaster_module=clockmaster_module,
        quel1_module=quel1_module,
        driver_module=driver_module,
        tool_module=tool_module,
        neopulse_module=neopulse_module,
        qubecalib_module=qubecalib_module,
        direct_multi_module=direct_multi_module,
        direct_single_module=direct_single_module,
        QubeCalib=root_module.QubeCalib,
        Sequencer=root_module.Sequencer,
        QuBEMasterClient=clockmaster_module.QuBEMasterClient,
        SequencerClient=clockmaster_module.SequencerClient,
        Quel1System=quel1_module.Quel1System,
        Action=driver_module.Action,
        AwgId=driver_module.AwgId,
        AwgSetting=driver_module.AwgSetting,
        NamedBox=driver_module.NamedBox,
        RunitId=driver_module.RunitId,
        RunitSetting=driver_module.RunitSetting,
        TriggerSetting=driver_module.TriggerSetting,
        Skew=tool_module.Skew,
        DEFAULT_SAMPLING_PERIOD=neopulse_module.DEFAULT_SAMPLING_PERIOD,
        CapSampledSequence=neopulse_module.CapSampledSequence,
        GenSampledSequence=neopulse_module.GenSampledSequence,
        BoxPool=qubecalib_module.BoxPool,
        CaptureParamTools=qubecalib_module.CaptureParamTools,
        Converter=qubecalib_module.Converter,
        WaveSequenceTools=qubecalib_module.WaveSequenceTools,
    )


@lru_cache(maxsize=4)
def load_quel_driver(preference: str | None = None) -> QuelDriverModules:
    """Load the preferred driver package with compatibility fallback."""
    resolved = _normalize_preference(preference)

    if resolved == _PREFERENCE_QXDRIVER:
        return _import_driver_package(_PREFERENCE_QXDRIVER)
    if resolved == _PREFERENCE_QUBECALIB:
        return _import_driver_package(_PREFERENCE_QUBECALIB)

    try:
        return _import_driver_package(_PREFERENCE_QXDRIVER)
    except ModuleNotFoundError:
        return _import_driver_package(_PREFERENCE_QUBECALIB)


def clear_quel_driver_cache() -> None:
    """Clear cached driver resolution results."""
    load_quel_driver.cache_clear()
