"""QuEL-1 oscillator mixing arithmetic, independent of system models."""

from typing import Literal

LO_STEP_HZ = 500_000_000
NCO_STEP_HZ = 23_437_500


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
