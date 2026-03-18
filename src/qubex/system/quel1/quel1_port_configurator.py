"""QuEL-1-specific port configuration helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Final, Literal

import numpy as np
from typing_extensions import TypedDict

from qubex.system.control_system import Box, BoxType
from qubex.system.quantum_system import Mux, Qubit
from qubex.system.quel1.quel1_system_constants import (
    AWG_MAX_HZ,
    CNCO_CENTER_CTRL_HZ,
    CNCO_CENTER_READ_HZ,
    FNCO_MAX_HZ,
    LO_STEP_HZ,
    NCO_STEP_HZ,
)
from qubex.typing import ConfigurationMode

logger = logging.getLogger(__name__)


class ReadoutMixingConfig(TypedDict):
    """Readout mixing configuration for one mux."""

    lo: int | None
    cnco: int
    fnco: int


class ControlChannelConfig(TypedDict):
    """Per-channel control mixing configuration."""

    fnco: int
    targets: list[str]


class ControlMixingConfig(TypedDict):
    """Control mixing configuration for one qubit."""

    lo: int | None
    cnco: int
    channels: dict[int, ControlChannelConfig]


class CrTargetConfig(TypedDict):
    """CR target label and frequency pair."""

    label: str
    frequency: float


class MixingUtil:
    """Utility helpers for LO/NCO mixing calculations."""

    @staticmethod
    def calc_lo_cnco(
        f: float,
        cnco_center: int | None,
        ssb: Literal["U", "L"] | None,
        lo_step: int = LO_STEP_HZ,
        nco_step: int = NCO_STEP_HZ,
    ) -> tuple[int | None, int, int]:
        """Calculate LO/CNCO settings for a target frequency."""
        if ssb is None:
            lo = None
            cnco = round(f / nco_step) * nco_step
            f_mix = cnco
        else:
            if cnco_center is None:
                raise ValueError("CNCO center is required when SSB is not None.")
            if ssb == "U":
                lo = round((f - cnco_center) / lo_step) * lo_step
                cnco = round((f - lo) / nco_step) * nco_step
            elif ssb == "L":
                lo = round((f + cnco_center) / lo_step) * lo_step
                cnco = round((lo - f) / nco_step) * nco_step
            else:
                raise ValueError("Invalid SSB")
            f_mix = lo + cnco if ssb == "U" else lo - cnco
        return lo, cnco, f_mix

    @staticmethod
    def calc_fnco(
        f: float,
        ssb: Literal["U", "L"] | None,
        lo: int | None,
        cnco: int,
        nco_step: int = NCO_STEP_HZ,
    ) -> tuple[int, int]:
        """Calculate FNCO settings for a target frequency."""
        if ssb is None and lo is None:
            fnco = round((f - cnco) / nco_step) * nco_step
            f_mix = cnco + fnco
        elif lo is None:
            raise ValueError("LO frequency is required when SSB is not None.")
        elif ssb is None:
            raise ValueError("SSB is required when LO frequency is not None.")
        else:
            if ssb == "U":
                fnco = round((f - (lo + cnco)) / nco_step) * nco_step
            elif ssb == "L":
                fnco = round(((lo - cnco) - f) / nco_step) * nco_step
            else:
                raise ValueError("Invalid SSB")
            f_mix = lo + cnco + fnco if ssb == "U" else lo - cnco - fnco
        return fnco, f_mix


QUEL1_BOX_TYPES: Final[frozenset[BoxType]] = frozenset(
    {
        BoxType.QUEL1_A,
        BoxType.QUEL1_B,
        BoxType.QUBE_RIKEN_A,
        BoxType.QUBE_RIKEN_B,
        BoxType.QUBE_OU_A,
        BoxType.QUBE_OU_B,
        BoxType.QUEL1SE_A,
        BoxType.QUEL1SE_B,
        BoxType.QUEL1SE_R8,
    }
)


def get_boxes_to_configure(boxes: Sequence[Box]) -> list[Box]:
    """Return boxes that require QuEL-1 port initialization."""
    return [box for box in boxes if box.type in QUEL1_BOX_TYPES]


def create_readout_configuration(
    mux: Mux,
    *,
    excluded_targets: Sequence[str],
    ssb: Literal["U", "L"] | None = "U",
    cnco_center: int | None = CNCO_CENTER_READ_HZ,
) -> ReadoutMixingConfig:
    """Build readout mixing settings for one mux."""
    resonators = [
        resonator
        for resonator in mux.resonators
        if resonator.is_valid and resonator.label not in excluded_targets
    ]
    freqs = [resonator.frequency * 1e9 for resonator in resonators]
    f_target = (max(freqs) + min(freqs)) / 2
    lo, cnco, _ = MixingUtil.calc_lo_cnco(
        f=f_target,
        ssb=ssb,
        cnco_center=cnco_center,
    )
    fnco, _ = MixingUtil.calc_fnco(
        f=f_target,
        ssb=ssb,
        lo=lo,
        cnco=cnco,
    )
    return {"lo": lo, "cnco": cnco, "fnco": fnco}


def create_control_configuration(
    *,
    mode: ConfigurationMode,
    qubit: Qubit,
    n_channels: int,
    get_spectator_qubits: Callable[[str], list[Qubit]],
    excluded_targets: Sequence[str],
    ssb: Literal["U", "L"] | None = "L",
    cnco_center: int = CNCO_CENTER_CTRL_HZ,
    min_frequency: float = 6.5e9,
) -> ControlMixingConfig:
    """Build control mixing settings for one qubit."""
    layout = _resolve_control_layout(mode=mode, n_channels=n_channels)

    if layout == "ge":
        f_target = qubit.frequency * 1e9
        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            f=f_target,
            ssb=ssb,
            cnco_center=cnco_center,
        )
        fnco, _ = MixingUtil.calc_fnco(f=f_target, ssb=ssb, lo=lo, cnco=cnco)
        return {
            "lo": lo,
            "cnco": cnco,
            "channels": {0: {"fnco": fnco, "targets": [qubit.label]}},
        }

    if layout == "ge-ef":
        f_ge = qubit.frequency * 1e9
        f_ef = qubit.control_frequency_ef * 1e9
        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            f=(f_ge + f_ef) / 2,
            ssb=ssb,
            cnco_center=cnco_center,
        )
        fnco_ge, _ = MixingUtil.calc_fnco(f=f_ge, ssb=ssb, lo=lo, cnco=cnco)
        fnco_ef, _ = MixingUtil.calc_fnco(f=f_ef, ssb=ssb, lo=lo, cnco=cnco)
        return {
            "lo": lo,
            "cnco": cnco,
            "channels": {
                0: {"fnco": fnco_ge, "targets": [qubit.label]},
                1: {"fnco": fnco_ef, "targets": [f"{qubit.label}-ef"]},
            },
        }

    if layout == "ge-cr":
        f_ge = qubit.frequency * 1e9
        f_ef = qubit.control_frequency_ef * 1e9
        cr_targets = _collect_cr_targets(
            qubit=qubit,
            get_spectator_qubits=get_spectator_qubits,
            excluded_targets=excluded_targets,
        )
        f_crs = [target["frequency"] for target in cr_targets]
        if not f_crs:
            f_crs = [f_ge]
        f_cr_max = max(f_crs)
        if f_cr_max > f_ge:
            if f_ef < min_frequency:
                f_ef = f_ge
            lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                f=f_ef + FNCO_MAX_HZ, ssb=ssb, cnco_center=cnco_center
            )
            f_crs_valid = [f for f in f_crs if f < f_coarse + FNCO_MAX_HZ + AWG_MAX_HZ]
        else:
            lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                f=f_ge - FNCO_MAX_HZ, ssb=ssb, cnco_center=cnco_center
            )
            f_crs_valid = [f for f in f_crs if f > f_coarse - FNCO_MAX_HZ - AWG_MAX_HZ]
        f_cr = _find_center_freq_for_cr(f_coarse=f_coarse, f_crs=f_crs_valid)
        fnco_ge, _ = MixingUtil.calc_fnco(
            f=(f_ge + f_ef) * 0.5, ssb=ssb, lo=lo, cnco=cnco
        )
        fnco_cr, _ = MixingUtil.calc_fnco(f=f_cr, ssb=ssb, lo=lo, cnco=cnco)
        return {
            "lo": lo,
            "cnco": cnco,
            "channels": {
                0: {"fnco": fnco_ge, "targets": [qubit.label]},
                1: {
                    "fnco": fnco_cr,
                    "targets": [f"{qubit.label}-CR", *_target_labels(cr_targets)],
                },
            },
        }

    if layout == "ge-ef-cr":
        f_ge = qubit.frequency * 1e9
        f_ef = qubit.control_frequency_ef * 1e9
        cr_targets = _collect_cr_targets(
            qubit=qubit,
            get_spectator_qubits=get_spectator_qubits,
            excluded_targets=excluded_targets,
        )
        f_crs = [target["frequency"] for target in cr_targets]
        if not f_crs:
            f_crs = [f_ge]
        f_cr_max = max(f_crs)
        if f_cr_max > f_ge:
            if f_ef < min_frequency:
                f_ef = f_ge
            lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                f=f_ef + FNCO_MAX_HZ, ssb=ssb, cnco_center=cnco_center
            )
            f_crs_valid = [f for f in f_crs if f < f_coarse + FNCO_MAX_HZ + AWG_MAX_HZ]
        else:
            lo, cnco, f_coarse = MixingUtil.calc_lo_cnco(
                f=f_ge - FNCO_MAX_HZ, ssb=ssb, cnco_center=cnco_center
            )
            f_crs_valid = [f for f in f_crs if f > f_coarse - FNCO_MAX_HZ - AWG_MAX_HZ]
        f_cr = _find_center_freq_for_cr(f_coarse=f_coarse, f_crs=f_crs_valid)
        fnco_ge, _ = MixingUtil.calc_fnco(f=f_ge, ssb=ssb, lo=lo, cnco=cnco)
        fnco_ef, _ = MixingUtil.calc_fnco(f=f_ef, ssb=ssb, lo=lo, cnco=cnco)
        fnco_cr, _ = MixingUtil.calc_fnco(f=f_cr, ssb=ssb, lo=lo, cnco=cnco)
        return {
            "lo": lo,
            "cnco": cnco,
            "channels": {
                0: {"fnco": fnco_ge, "targets": [qubit.label]},
                1: {"fnco": fnco_ef, "targets": [f"{qubit.label}-ef"]},
                2: {
                    "fnco": fnco_cr,
                    "targets": [f"{qubit.label}-CR", *_target_labels(cr_targets)],
                },
            },
        }

    if layout == "ge-cr-cr":
        f_ge = qubit.frequency * 1e9
        cr_targets = _collect_cr_targets(
            qubit=qubit,
            get_spectator_qubits=get_spectator_qubits,
            excluded_targets=excluded_targets,
        )
        if not cr_targets:
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f=f_ge,
                ssb=ssb,
                cnco_center=cnco_center,
            )
            fnco_ge, _ = MixingUtil.calc_fnco(f=f_ge, ssb=ssb, lo=lo, cnco=cnco)
            fnco_cr, _ = MixingUtil.calc_fnco(f=f_ge, ssb=ssb, lo=lo, cnco=cnco)
            return {
                "lo": lo,
                "cnco": cnco,
                "channels": {
                    0: {"fnco": fnco_ge, "targets": [qubit.label]},
                    1: {"fnco": fnco_cr, "targets": [f"{qubit.label}-CR"]},
                    2: {"fnco": fnco_cr, "targets": []},
                },
            }

        group1, group2 = _split_cr_target_group(cr_targets)
        f_cr_1 = _mean_target_frequency(group1, fallback=f_ge)
        f_cr_2 = _mean_target_frequency(group2, fallback=f_cr_1)
        f_min = min(f_ge, f_cr_1, f_cr_2)
        f_max = max(f_ge, f_cr_1, f_cr_2)
        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            f=(f_min + f_max) / 2, ssb=ssb, cnco_center=cnco_center
        )
        fnco_ge, _ = MixingUtil.calc_fnco(f=f_ge, ssb=ssb, lo=lo, cnco=cnco)
        fnco_cr_1, _ = MixingUtil.calc_fnco(f=f_cr_1, ssb=ssb, lo=lo, cnco=cnco)
        fnco_cr_2, _ = MixingUtil.calc_fnco(f=f_cr_2, ssb=ssb, lo=lo, cnco=cnco)
        return {
            "lo": lo,
            "cnco": cnco,
            "channels": {
                0: {"fnco": fnco_ge, "targets": [qubit.label]},
                1: {
                    "fnco": fnco_cr_1,
                    "targets": [f"{qubit.label}-CR", *_target_labels(group1)],
                },
                2: {
                    "fnco": fnco_cr_2,
                    "targets": _target_labels(group2),
                },
            },
        }
    raise ValueError("Invalid mode.")


def _resolve_control_layout(*, mode: ConfigurationMode, n_channels: int) -> str:
    """Resolve the channel-role prefix implied by one configuration mode."""
    if n_channels <= 0:
        raise ValueError("Control ports must expose at least one channel.")
    if n_channels == 1:
        return "ge"
    if n_channels == 2:
        return "ge-ef" if mode == "ge-ef-cr" else "ge-cr"
    if n_channels == 3:
        return mode
    raise ValueError(f"Unsupported control channel count: {n_channels}.")


def _collect_cr_targets(
    *,
    qubit: Qubit,
    get_spectator_qubits: Callable[[str], list[Qubit]],
    excluded_targets: Sequence[str],
) -> list[CrTargetConfig]:
    """Collect valid pair-CR targets for one control qubit."""
    return [
        {
            "label": f"{qubit.label}-{spectator.label}",
            "frequency": spectator.frequency * 1e9,
        }
        for spectator in get_spectator_qubits(qubit.label)
        if spectator.frequency > 0
        and spectator.label not in excluded_targets
        and f"{qubit.label}-{spectator.label}" not in excluded_targets
    ]


def _target_labels(targets: Sequence[CrTargetConfig]) -> list[str]:
    """Return only the labels from one CR target group."""
    return [target["label"] for target in targets]


def _mean_target_frequency(
    targets: Sequence[CrTargetConfig],
    *,
    fallback: float,
) -> float:
    """Return the mean target frequency or a fallback when the group is empty."""
    if not targets:
        return fallback
    return np.mean([target["frequency"] for target in targets]).astype(float)


def _split_cr_target_group(
    group: list[CrTargetConfig],
) -> tuple[list[CrTargetConfig], list[CrTargetConfig]]:
    group = sorted(group, key=lambda x: x["frequency"])
    if len(group) == 0:
        raise ValueError("No CR target found.")
    if len(group) == 1:
        return [group[0]], []
    if len(group) == 2:
        return [group[0]], [group[1]]
    if len(group) == 3:
        split_options = [
            ([group[0], group[1]], [group[2]]),
            ([group[0]], [group[1], group[2]]),
        ]
    elif len(group) == 4:
        split_options = [
            ([group[0], group[1]], [group[2], group[3]]),
            ([group[0], group[1], group[2]], [group[3]]),
            ([group[0]], [group[1], group[2], group[3]]),
        ]
    else:
        raise ValueError("Too many CR targets.")
    best_split = None
    best_max_bandwidth = float("inf")
    for group1, group2 in split_options:
        f_min1 = min(target["frequency"] for target in group1)
        f_max1 = max(target["frequency"] for target in group1)
        f_min2 = min(target["frequency"] for target in group2)
        f_max2 = max(target["frequency"] for target in group2)
        bandwidth1 = f_max1 - f_min1 if len(group1) > 1 else 0
        bandwidth2 = f_max2 - f_min2 if len(group2) > 1 else 0
        max_band = max(bandwidth1, bandwidth2)
        if max_band < best_max_bandwidth:
            best_max_bandwidth = max_band
            best_split = (group1, group2)
    if best_split is None:
        raise ValueError("No split found.")
    return best_split


def _find_center_freq_for_cr(
    *,
    f_coarse: int,
    f_crs: list[float],
) -> int:
    if not f_crs:
        return f_coarse
    min_center_freq = f_coarse - FNCO_MAX_HZ
    max_center_freq = f_coarse + FNCO_MAX_HZ
    search_range = np.arange(
        max(min(f_crs), min_center_freq),
        min(max(f_crs), max_center_freq) + 1,
        NCO_STEP_HZ,
    )
    center_freqs_by_count = []
    for frequency in search_range:
        valid_f_crs = [
            f_cr
            for f_cr in f_crs
            if frequency - AWG_MAX_HZ <= f_cr <= frequency + AWG_MAX_HZ
        ]
        if not valid_f_crs:
            continue
        center = (min(valid_f_crs) + max(valid_f_crs)) / 2
        center_freqs_by_count.append((len(valid_f_crs), center))
    if not center_freqs_by_count:
        return f_coarse
    center_freqs_by_count.sort(key=lambda x: (x[0], x[1]), reverse=True)
    center_freq = int(center_freqs_by_count[0][1])
    center_freq = round(center_freq / NCO_STEP_HZ) * NCO_STEP_HZ
    center_freq = max(min_center_freq, min(center_freq, max_center_freq))
    return center_freq
