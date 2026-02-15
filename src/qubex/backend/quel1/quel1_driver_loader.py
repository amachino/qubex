"""Loader for selecting QuEL driver package with compatibility fallback."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .quel1_driver_protocols import (
        ActionProtocol,
        AwgIdProtocol,
        AwgSettingProtocol,
        BoxPoolProtocol,
        CaptureParamToolsProtocol,
        ConverterProtocol,
        DirectMultiActionProtocol,
        DirectSingleActionProtocol,
        NamedBoxProtocol,
        QubeCalibProtocol,
        QuBEMasterClientProtocol,
        Quel1BoxProtocol,
        Quel1ConfigOptionProtocol,
        Quel1SystemProtocol,
        RunitIdProtocol,
        RunitSettingProtocol,
        SequencerClientProtocol,
        SequencerProtocol,
        SkewProtocol,
        TriggerSettingProtocol,
        WaveSequenceToolsProtocol,
    )

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
class QuelDriverClasses:
    """Resolved class symbols used by QuEL backend runtime."""

    package_name: str
    QubeCalib: type[QubeCalibProtocol]
    Sequencer: type[SequencerProtocol]
    QuBEMasterClient: type[QuBEMasterClientProtocol]
    SequencerClient: type[SequencerClientProtocol]
    Quel1System: type[Quel1SystemProtocol]
    Action: type[ActionProtocol]
    DirectMultiAction: type[DirectMultiActionProtocol]
    DirectSingleAction: type[DirectSingleActionProtocol]
    AwgId: type[AwgIdProtocol]
    AwgSetting: type[AwgSettingProtocol]
    NamedBox: type[NamedBoxProtocol]
    RunitId: type[RunitIdProtocol]
    RunitSetting: type[RunitSettingProtocol]
    TriggerSetting: type[TriggerSettingProtocol]
    Skew: type[SkewProtocol]
    DEFAULT_SAMPLING_PERIOD: float | int
    CapSampledSequence: type[Any]
    CapSampledSubSequence: type[Any]
    CaptureSlots: type[Any]
    GenSampledSequence: type[Any]
    GenSampledSubSequence: type[Any]
    BoxPool: type[BoxPoolProtocol]
    CaptureParamTools: type[CaptureParamToolsProtocol]
    Converter: type[ConverterProtocol]
    WaveSequenceTools: type[WaveSequenceToolsProtocol]
    Quel1Box: type[Quel1BoxProtocol]
    Quel1ConfigOption: type[Quel1ConfigOptionProtocol]


def _normalize_preference(preference: str | None) -> str:
    """Resolve and validate the current driver preference label."""
    resolved = preference or os.getenv(_DRIVER_ENV_VAR, _PREFERENCE_AUTO)
    if resolved not in _VALID_PREFERENCES:
        raise ValueError(
            f"Invalid {_DRIVER_ENV_VAR}='{resolved}'. "
            f"Expected one of {sorted(_VALID_PREFERENCES)}."
        )
    return resolved


_SymbolCandidate = tuple[str, str]

_BASE_SYMBOL_CANDIDATES: dict[str, tuple[_SymbolCandidate, ...]] = {
    "QubeCalib": (
        ("{package}.compat", "QubeCalib"),
        ("{package}", "QubeCalib"),
        ("{package}.qubecalib", "QubeCalib"),
        ("{package}.facade", "QubeCalib"),
    ),
    "Sequencer": (
        ("{package}.compat", "Sequencer"),
        ("{package}", "Sequencer"),
        ("{package}.qubecalib", "Sequencer"),
        ("{package}.facade", "Sequencer"),
    ),
    "QuBEMasterClient": (
        ("{package}.compat", "QuBEMasterClient"),
        ("{package}.clockmaster_compat", "QuBEMasterClient"),
        ("{package}.qubecalib", "QuBEMasterClient"),
        ("{package}.facade", "QuBEMasterClient"),
    ),
    "SequencerClient": (
        ("{package}.compat", "SequencerClient"),
        ("{package}.clockmaster_compat", "SequencerClient"),
        ("{package}.qubecalib", "SequencerClient"),
        ("{package}.facade", "SequencerClient"),
    ),
    "Quel1System": (
        ("{package}.compat", "Quel1System"),
        ("{package}.instrument.quel.quel1", "Quel1System"),
        ("{package}.instrument.quel.quel1.driver", "Quel1System"),
        ("{package}.facade", "direct.Quel1System"),
    ),
    "Action": (
        ("{package}.compat", "Action"),
        ("{package}.instrument.quel.quel1.driver", "Action"),
        ("{package}.facade", "direct.Action"),
    ),
    "AwgId": (
        ("{package}.compat", "AwgId"),
        ("{package}.instrument.quel.quel1.driver", "AwgId"),
        ("{package}.facade", "direct.AwgId"),
    ),
    "AwgSetting": (
        ("{package}.compat", "AwgSetting"),
        ("{package}.instrument.quel.quel1.driver", "AwgSetting"),
        ("{package}.facade", "direct.AwgSetting"),
    ),
    "NamedBox": (
        ("{package}.compat", "NamedBox"),
        ("{package}.instrument.quel.quel1.driver", "NamedBox"),
        ("{package}.facade", "direct.NamedBox"),
    ),
    "RunitId": (
        ("{package}.compat", "RunitId"),
        ("{package}.instrument.quel.quel1.driver", "RunitId"),
        ("{package}.facade", "direct.RunitId"),
    ),
    "RunitSetting": (
        ("{package}.compat", "RunitSetting"),
        ("{package}.instrument.quel.quel1.driver", "RunitSetting"),
        ("{package}.facade", "direct.RunitSetting"),
    ),
    "TriggerSetting": (
        ("{package}.compat", "TriggerSetting"),
        ("{package}.instrument.quel.quel1.driver", "TriggerSetting"),
        ("{package}.facade", "direct.TriggerSetting"),
    ),
    "Skew": (
        ("{package}.compat", "Skew"),
        ("{package}.instrument.quel.quel1.tool", "Skew"),
    ),
    "DEFAULT_SAMPLING_PERIOD": (
        ("{package}.compat", "DEFAULT_SAMPLING_PERIOD"),
        ("{package}.neopulse", "DEFAULT_SAMPLING_PERIOD"),
        ("{package}", "neopulse.DEFAULT_SAMPLING_PERIOD"),
    ),
    "CapSampledSequence": (
        ("{package}.compat", "CapSampledSequence"),
        ("{package}.neopulse", "CapSampledSequence"),
        ("{package}", "neopulse.CapSampledSequence"),
    ),
    "CapSampledSubSequence": (
        ("{package}.compat", "CapSampledSubSequence"),
        ("{package}.neopulse", "CapSampledSubSequence"),
        ("{package}", "neopulse.CapSampledSubSequence"),
    ),
    "CaptureSlots": (
        ("{package}.compat", "CaptureSlots"),
        ("{package}.neopulse", "CaptureSlots"),
        ("{package}", "neopulse.CaptureSlots"),
    ),
    "GenSampledSequence": (
        ("{package}.compat", "GenSampledSequence"),
        ("{package}.neopulse", "GenSampledSequence"),
        ("{package}", "neopulse.GenSampledSequence"),
    ),
    "GenSampledSubSequence": (
        ("{package}.compat", "GenSampledSubSequence"),
        ("{package}.neopulse", "GenSampledSubSequence"),
        ("{package}", "neopulse.GenSampledSubSequence"),
    ),
    "BoxPool": (
        ("{package}.compat", "BoxPool"),
        ("{package}.qubecalib", "BoxPool"),
        ("{package}.facade", "BoxPool"),
    ),
    "CaptureParamTools": (
        ("{package}.compat", "CaptureParamTools"),
        ("{package}.qubecalib", "CaptureParamTools"),
        ("{package}.facade", "CaptureParamTools"),
    ),
    "Converter": (
        ("{package}.compat", "Converter"),
        ("{package}.qubecalib", "Converter"),
        ("{package}.facade", "Converter"),
    ),
    "WaveSequenceTools": (
        ("{package}.compat", "WaveSequenceTools"),
        ("{package}.qubecalib", "WaveSequenceTools"),
        ("{package}.facade", "WaveSequenceTools"),
    ),
    "DirectMultiAction": (
        ("{package}.compat", "DirectMultiAction"),
        ("{package}.instrument.quel.quel1.driver.multi", "Action"),
        ("{package}.facade", "direct.multi.Action"),
    ),
    "DirectSingleAction": (
        ("{package}.compat", "DirectSingleAction"),
        ("{package}.instrument.quel.quel1.driver.single", "Action"),
        ("{package}.facade", "direct.single.Action"),
    ),
    "Quel1Box": (
        ("{package}.compat", "Quel1Box"),
        ("{package}.facade", "Quel1Box"),
        ("{package}.qubecalib", "Quel1Box"),
        ("{package}.qubecalib", "Quel1BoxWithRawWss"),
    ),
    "Quel1ConfigOption": (
        ("{package}.compat", "Quel1ConfigOption"),
        ("{package}.facade", "Quel1ConfigOption"),
        ("{package}.qubecalib", "Quel1ConfigOption"),
    ),
}


def _is_missing_target_module(error: ModuleNotFoundError, module_name: str) -> bool:
    """Return whether ModuleNotFoundError indicates the target module itself is missing."""
    missing_name = error.name
    if not missing_name:
        text = str(error)
        if text.startswith("No module named "):
            missing_name = text.split("No module named ", maxsplit=1)[1].strip("'")
        else:
            missing_name = text
    return missing_name == module_name or str(missing_name).startswith(
        f"{module_name}."
    )


def _resolve_attr_from_module(module: ModuleType, attr_path: str) -> Any:
    """Resolve dotted attributes from a module object."""
    value: Any = module
    for attr in attr_path.split("."):
        value = getattr(value, attr)
    return value


def _resolve_symbol(
    *,
    package_name: str,
    symbol_name: str,
) -> Any:
    """Resolve one required symbol from package-specific candidate paths."""
    candidates = list(_BASE_SYMBOL_CANDIDATES[symbol_name])
    for module_template, attr_path in candidates:
        module_name = module_template.format(package=package_name)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if _is_missing_target_module(error, module_name):
                continue
            raise
        try:
            return _resolve_attr_from_module(module, attr_path)
        except AttributeError:
            continue

    raise ModuleNotFoundError(
        f"Could not resolve symbol '{symbol_name}' for package '{package_name}'."
    )


def _apply_runtime_patches(package_name: str) -> None:
    """Apply optional runtime patches provided by the resolved driver package."""
    candidate_modules = (
        f"{package_name}.runtime_patches",
        f"{package_name}.compat",
        f"{package_name}.facade",
        f"{package_name}.qubecalib",
    )
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if _is_missing_target_module(error, module_name):
                continue
            raise
        apply_fn = getattr(module, "apply_runtime_patches", None)
        if callable(apply_fn):
            apply_fn()
            return


def _import_driver_package(package_name: str) -> QuelDriverClasses:
    """Import one driver package and map required backend symbols by class-level paths."""
    importlib.import_module(package_name)
    _apply_runtime_patches(package_name)

    resolved_symbols: dict[str, Any] = {}
    for symbol_name in _BASE_SYMBOL_CANDIDATES:
        resolved_symbols[symbol_name] = _resolve_symbol(
            package_name=package_name,
            symbol_name=symbol_name,
        )

    return QuelDriverClasses(
        package_name=package_name,
        QubeCalib=resolved_symbols["QubeCalib"],
        Sequencer=resolved_symbols["Sequencer"],
        QuBEMasterClient=resolved_symbols["QuBEMasterClient"],
        SequencerClient=resolved_symbols["SequencerClient"],
        Quel1System=resolved_symbols["Quel1System"],
        Action=resolved_symbols["Action"],
        DirectMultiAction=resolved_symbols["DirectMultiAction"],
        DirectSingleAction=resolved_symbols["DirectSingleAction"],
        AwgId=resolved_symbols["AwgId"],
        AwgSetting=resolved_symbols["AwgSetting"],
        NamedBox=resolved_symbols["NamedBox"],
        RunitId=resolved_symbols["RunitId"],
        RunitSetting=resolved_symbols["RunitSetting"],
        TriggerSetting=resolved_symbols["TriggerSetting"],
        Skew=resolved_symbols["Skew"],
        DEFAULT_SAMPLING_PERIOD=resolved_symbols["DEFAULT_SAMPLING_PERIOD"],
        CapSampledSequence=resolved_symbols["CapSampledSequence"],
        CapSampledSubSequence=resolved_symbols["CapSampledSubSequence"],
        CaptureSlots=resolved_symbols["CaptureSlots"],
        GenSampledSequence=resolved_symbols["GenSampledSequence"],
        GenSampledSubSequence=resolved_symbols["GenSampledSubSequence"],
        BoxPool=resolved_symbols["BoxPool"],
        CaptureParamTools=resolved_symbols["CaptureParamTools"],
        Converter=resolved_symbols["Converter"],
        WaveSequenceTools=resolved_symbols["WaveSequenceTools"],
        Quel1Box=resolved_symbols["Quel1Box"],
        Quel1ConfigOption=resolved_symbols["Quel1ConfigOption"],
    )


@lru_cache(maxsize=4)
def load_quel1_driver(preference: str | None = None) -> QuelDriverClasses:
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


def clear_quel1_driver_cache() -> None:
    """Clear cached driver resolution results."""
    load_quel1_driver.cache_clear()
