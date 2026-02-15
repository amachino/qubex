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
    Quel1Box: Any
    Quel1ConfigOption: Any


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
    "GenSampledSequence": (
        ("{package}.compat", "GenSampledSequence"),
        ("{package}.neopulse", "GenSampledSequence"),
        ("{package}", "neopulse.GenSampledSequence"),
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
) -> tuple[Any, ModuleType, str]:
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
            return _resolve_attr_from_module(module, attr_path), module, attr_path
        except AttributeError:
            continue

    raise ModuleNotFoundError(
        f"Could not resolve symbol '{symbol_name}' for package '{package_name}'."
    )


def _runtime_module_for_symbol(
    *,
    source_module: ModuleType,
    source_attr_path: str,
    runtime_attr_name: str,
    symbol: Any,
    symbol_name: str,
) -> ModuleType:
    """Resolve runtime module for a symbol, with shim fallback for re-export-only paths."""
    if hasattr(source_module, runtime_attr_name):
        if getattr(source_module, runtime_attr_name) is symbol:
            return source_module

    module_name = getattr(symbol, "__module__", None)
    if isinstance(module_name, str):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if not _is_missing_target_module(error, module_name):
                raise
        else:
            if hasattr(module, runtime_attr_name):
                if getattr(module, runtime_attr_name) is symbol:
                    return module

    shim_name = f"{source_module.__name__}.__shim__.{symbol_name}.{source_attr_path}"
    shim_module = ModuleType(shim_name)
    setattr(shim_module, runtime_attr_name, symbol)
    return shim_module


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


def _import_driver_package(package_name: str) -> QuelDriverModules:
    """Import one driver package and map required backend symbols by class-level paths."""
    root_module = importlib.import_module(package_name)
    _apply_runtime_patches(package_name)

    resolved_symbols: dict[str, Any] = {}
    symbol_sources: dict[str, tuple[ModuleType, str]] = {}
    for symbol_name in _BASE_SYMBOL_CANDIDATES:
        symbol, source_module, source_attr = _resolve_symbol(
            package_name=package_name,
            symbol_name=symbol_name,
        )
        resolved_symbols[symbol_name] = symbol
        symbol_sources[symbol_name] = (source_module, source_attr)

    runtime_module_specs: dict[str, tuple[str, str]] = {
        "clockmaster_module": ("QuBEMasterClient", "QuBEMasterClient"),
        "quel1_module": ("Quel1System", "Quel1System"),
        "driver_module": ("Action", "Action"),
        "tool_module": ("Skew", "Skew"),
        "neopulse_module": ("CapSampledSequence", "CapSampledSequence"),
        "qubecalib_module": ("BoxPool", "BoxPool"),
        "direct_multi_module": ("DirectMultiAction", "Action"),
        "direct_single_module": ("DirectSingleAction", "Action"),
    }
    runtime_modules: dict[str, ModuleType] = {}
    for field_name, (
        source_symbol_name,
        runtime_attr_name,
    ) in runtime_module_specs.items():
        source_module, source_attr = symbol_sources[source_symbol_name]
        symbol = resolved_symbols[source_symbol_name]
        runtime_modules[field_name] = _runtime_module_for_symbol(
            source_module=source_module,
            source_attr_path=source_attr,
            runtime_attr_name=runtime_attr_name,
            symbol=symbol,
            symbol_name=source_symbol_name,
        )

    return QuelDriverModules(
        package_name=package_name,
        root_module=root_module,
        clockmaster_module=runtime_modules["clockmaster_module"],
        quel1_module=runtime_modules["quel1_module"],
        driver_module=runtime_modules["driver_module"],
        tool_module=runtime_modules["tool_module"],
        neopulse_module=runtime_modules["neopulse_module"],
        qubecalib_module=runtime_modules["qubecalib_module"],
        direct_multi_module=runtime_modules["direct_multi_module"],
        direct_single_module=runtime_modules["direct_single_module"],
        QubeCalib=resolved_symbols["QubeCalib"],
        Sequencer=resolved_symbols["Sequencer"],
        QuBEMasterClient=resolved_symbols["QuBEMasterClient"],
        SequencerClient=resolved_symbols["SequencerClient"],
        Quel1System=resolved_symbols["Quel1System"],
        Action=resolved_symbols["Action"],
        AwgId=resolved_symbols["AwgId"],
        AwgSetting=resolved_symbols["AwgSetting"],
        NamedBox=resolved_symbols["NamedBox"],
        RunitId=resolved_symbols["RunitId"],
        RunitSetting=resolved_symbols["RunitSetting"],
        TriggerSetting=resolved_symbols["TriggerSetting"],
        Skew=resolved_symbols["Skew"],
        DEFAULT_SAMPLING_PERIOD=resolved_symbols["DEFAULT_SAMPLING_PERIOD"],
        CapSampledSequence=resolved_symbols["CapSampledSequence"],
        GenSampledSequence=resolved_symbols["GenSampledSequence"],
        BoxPool=resolved_symbols["BoxPool"],
        CaptureParamTools=resolved_symbols["CaptureParamTools"],
        Converter=resolved_symbols["Converter"],
        WaveSequenceTools=resolved_symbols["WaveSequenceTools"],
        Quel1Box=resolved_symbols["Quel1Box"],
        Quel1ConfigOption=resolved_symbols["Quel1ConfigOption"],
    )


@lru_cache(maxsize=4)
def load_quel1_driver(preference: str | None = None) -> QuelDriverModules:
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
