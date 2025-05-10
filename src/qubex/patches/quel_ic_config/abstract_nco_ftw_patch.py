from fractions import Fraction
from typing import Union

from quel_ic_config.ad9082_nco import AbstractNcoFtw


def patched_from_frequency(
    cls,
    nco_freq_hz_: Union[Fraction, float],
    converter_freq_hz_: Union[int, float],
    epsilon: float = 0.0,
):
    converter_freq_hz: int = cls._converter_frequency_as_interger(converter_freq_hz_)

    hcf: int = converter_freq_hz // 2
    if not (-hcf <= nco_freq_hz_ < hcf):
        raise ValueError(
            f"the given nco_frequency (= {nco_freq_hz_:f}Hz) is out of range"
        )

    if not isinstance(converter_freq_hz, int):
        raise TypeError("converter_freq_hz must be integer")

    if isinstance(nco_freq_hz_, Fraction):
        nco_freq_hz: Fraction = nco_freq_hz_
    else:
        nco_freq_hz = Fraction(nco_freq_hz_)

    negative: bool = nco_freq_hz < 0
    if negative:
        t: Fraction = -nco_freq_hz * (1 << 48) / converter_freq_hz
    else:
        t = nco_freq_hz * (1 << 48) / converter_freq_hz
    t = t.limit_denominator(0xFFFF_FFFF_FFFF)

    x: int = int(t)
    if negative and t.denominator > 1:
        x += 1
    delta_b: int = t.numerator - t.denominator * x
    modulus_a: int = t.denominator
    if negative:
        x = -x
        delta_b = -delta_b  # Notes: delta_b becomes non-negative, finally.
    obj = cls(ftw=x, delta_b=delta_b, modulus_a=modulus_a)

    conv_error = abs(obj.to_frequency(converter_freq_hz) - nco_freq_hz_)
    if conv_error > epsilon:
        raise ValueError(
            f"large conversion error (= {conv_error}Hz) of the given nco frequency (= {nco_freq_hz:f}Hz)"
        )

    return obj


AbstractNcoFtw.from_frequency = classmethod(patched_from_frequency)  # type: ignore
