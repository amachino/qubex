"""Loader for selecting QuEL driver package with compatibility fallback."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from qubex.patches.quel_ic_config import apply_quelware_runtime_patches

if TYPE_CHECKING:
    from .quel1_qubealib_protocols import (
        ActionProtocol,
        AwgIdProtocol,
        AwgSettingProtocol,
        BoxPoolProtocol,
        CaptureParamToolsProtocol,
        ConverterProtocol,
        MultiActionProtocol,
        NamedBoxProtocol,
        QubeCalibProtocol,
        QuBEMasterClientProtocol,
        Quel1BoxCommonProtocol,
        Quel1ConfigOptionProtocol,
        Quel1SystemProtocol,
        RunitIdProtocol,
        RunitSettingProtocol,
        SequencerClientProtocol,
        SequencerProtocol,
        SingleActionProtocol,
        SingleAwgIdProtocol,
        SingleAwgSettingProtocol,
        SingleRunitIdProtocol,
        SingleRunitSettingProtocol,
        SingleTriggerSettingProtocol,
        SkewProtocol,
        TriggerSettingProtocol,
        WaveSequenceToolsProtocol,
    )

DriverPackageName: TypeAlias = Literal["qxdriver_quel", "qubecalib"]

# Canonical import paths in qubecalib namespace.
_SYMBOL_IMPORT_PATHS: dict[str, str] = {
    "DEFAULT_SAMPLING_PERIOD": "qubecalib.neopulse.DEFAULT_SAMPLING_PERIOD",
    "Action": "qubecalib.instrument.quel.quel1.driver.Action",
    "AwgId": "qubecalib.instrument.quel.quel1.driver.AwgId",
    "AwgSetting": "qubecalib.instrument.quel.quel1.driver.AwgSetting",
    "BoxPool": "qubecalib.qubecalib.BoxPool",
    "CapSampledSequence": "qubecalib.neopulse.CapSampledSequence",
    "CapSampledSubSequence": "qubecalib.neopulse.CapSampledSubSequence",
    "CaptureParamTools": "qubecalib.qubecalib.CaptureParamTools",
    "CaptureSlots": "qubecalib.neopulse.CaptureSlots",
    "Converter": "qubecalib.qubecalib.Converter",
    "GenSampledSequence": "qubecalib.neopulse.GenSampledSequence",
    "GenSampledSubSequence": "qubecalib.neopulse.GenSampledSubSequence",
    "MultiAction": "qubecalib.instrument.quel.quel1.driver.multi.Action",
    "NamedBox": "qubecalib.instrument.quel.quel1.driver.NamedBox",
    "QuBEMasterClient": "qubecalib.qubecalib.QuBEMasterClient",
    "QubeCalib": "qubecalib.QubeCalib",
    "Quel1Box": "qubecalib.qubecalib.Quel1BoxWithRawWss",
    "Quel1ConfigOption": "qubecalib.qubecalib.Quel1ConfigOption",
    "Quel1System": "qubecalib.instrument.quel.quel1.Quel1System",
    "RunitId": "qubecalib.instrument.quel.quel1.driver.RunitId",
    "RunitSetting": "qubecalib.instrument.quel.quel1.driver.RunitSetting",
    "Sequencer": "qubecalib.Sequencer",
    "SequencerClient": "qubecalib.qubecalib.SequencerClient",
    "SingleAction": "qubecalib.instrument.quel.quel1.driver.single.Action",
    "SingleAwgId": "qubecalib.instrument.quel.quel1.driver.single.AwgId",
    "SingleAwgSetting": "qubecalib.instrument.quel.quel1.driver.single.AwgSetting",
    "SingleRunitId": "qubecalib.instrument.quel.quel1.driver.single.RunitId",
    "SingleRunitSetting": "qubecalib.instrument.quel.quel1.driver.single.RunitSetting",
    "SingleTriggerSetting": "qubecalib.instrument.quel.quel1.driver.single.TriggerSetting",
    "Skew": "qubecalib.instrument.quel.quel1.tool.Skew",
    "TriggerSetting": "qubecalib.instrument.quel.quel1.driver.TriggerSetting",
    "WaveSequenceTools": "qubecalib.qubecalib.WaveSequenceTools",
}


@dataclass(frozen=True)
class Que1lDriver:
    """Resolved class symbols used by QuEL backend runtime."""

    package_name: DriverPackageName
    DEFAULT_SAMPLING_PERIOD: float | int
    Action: type[ActionProtocol]
    AwgId: type[AwgIdProtocol]
    AwgSetting: type[AwgSettingProtocol]
    BoxPool: type[BoxPoolProtocol]
    CapSampledSequence: type[Any]
    CapSampledSubSequence: type[Any]
    CaptureParamTools: type[CaptureParamToolsProtocol]
    CaptureSlots: type[Any]
    Converter: type[ConverterProtocol]
    GenSampledSequence: type[Any]
    GenSampledSubSequence: type[Any]
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


def _is_quelware_0_8_x() -> bool:
    """Return whether installed quelware major/minor version is 0.8."""
    try:
        installed_version = version("quel_ic_config")
    except PackageNotFoundError:
        return False

    matched = re.match(r"^(\d+)\.(\d+)", installed_version)
    if not matched:
        return False

    major, minor = int(matched.group(1)), int(matched.group(2))
    return major == 0 and minor == 8


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


def _resolve_symbol(*, package_name: DriverPackageName, symbol_name: str) -> Any:
    """Resolve one required symbol from package-specific import rules."""
    symbol_path = _SYMBOL_IMPORT_PATHS[symbol_name]
    module_path, resolved_attr_name = symbol_path.rsplit(".", maxsplit=1)

    if package_name == "qxdriver_quel":
        module_name = f"{package_name}.compat"
        attr_name = symbol_name
    else:
        module_name = module_path
        attr_name = resolved_attr_name

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if _is_missing_target_module(error, module_name):
            raise ModuleNotFoundError(
                f"Could not resolve symbol '{symbol_name}' for package '{package_name}'."
            ) from error
        raise

    if hasattr(module, attr_name):
        return getattr(module, attr_name)

    raise ModuleNotFoundError(
        f"Could not resolve symbol '{symbol_name}' for package '{package_name}'."
    )


def _import_driver_symbols(package_name: DriverPackageName) -> Que1lDriver:
    """Import one driver package and map required backend symbols by class-level paths."""
    importlib.import_module(package_name)
    apply_quelware_runtime_patches()

    resolved_symbols = {
        symbol_name: _resolve_symbol(
            package_name=package_name,
            symbol_name=symbol_name,
        )
        for symbol_name in _SYMBOL_IMPORT_PATHS
    }

    return Que1lDriver(
        package_name=package_name,
        **resolved_symbols,
    )


@lru_cache(maxsize=1)
def load_quel1_driver() -> Que1lDriver:
    """
    Load one driver package selected by installed quelware version.

    Notes
    -----
    This function is cached by `functools.lru_cache(maxsize=1)`.
    Calling it from multiple places is safe: module import and symbol
    resolution happen only on the first call in a process unless the
    cache is explicitly cleared via `clear_quel1_driver_cache`.
    """
    package_name: DriverPackageName = (
        "qubecalib" if _is_quelware_0_8_x() else "qxdriver_quel"
    )
    return _import_driver_symbols(package_name)


def clear_quel1_driver_cache() -> None:
    """Clear cached driver resolution results."""
    load_quel1_driver.cache_clear()
