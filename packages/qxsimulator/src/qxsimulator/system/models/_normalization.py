"""Normalize system-model inputs to canonical scalar units."""

from __future__ import annotations

from qxcore import Frequency, Time


def normalize_frequency_to_ghz(value: float | Frequency) -> float:
    """
    Normalize a cyclic frequency to a float in GHz.

    Parameters
    ----------
    value : float | Frequency
        Frequency quantity, or a bare float already expressed in GHz.

    Returns
    -------
    float
        Cyclic frequency in GHz.
    """
    if isinstance(value, Frequency):
        return float(value.value_in_base_units() * 1e-9)
    return float(value)


def normalize_time_to_ns(value: float | Time | None) -> float | None:
    """
    Normalize a duration to a float in ns.

    Parameters
    ----------
    value : float | Time | None
        Time quantity, a bare float already expressed in ns, or `None`.

    Returns
    -------
    float | None
        Duration in ns, or `None` when no duration was supplied.
    """
    if value is None:
        return None
    if isinstance(value, Time):
        return float(value.value_in_base_units() * 1e9)
    return float(value)


def normalize_rate_to_inverse_ns(value: float | Frequency) -> float:
    """
    Normalize an exponential decay rate to a float in inverse ns.

    Parameters
    ----------
    value : float | Frequency
        Frequency-dimensional rate, or a bare float already expressed in
        inverse ns.

    Returns
    -------
    float
        Decay rate in inverse ns.

    Notes
    -----
    Frequency-dimensional `tunits` values are converted by unit scaling only;
    no `2 * pi` angular-frequency conversion is applied.
    """
    if isinstance(value, Frequency):
        return float(value.value_in_base_units() * 1e-9)
    return float(value)


def resolve_decoherence_rates(
    *,
    t1: float | Time | None,
    t2: float | Time | None,
    relaxation_rate: float | Frequency | None,
    dephasing_rate: float | Frequency | None,
) -> tuple[float, float]:
    """
    Resolve time- or rate-based decoherence inputs into physical rates.

    Parameters
    ----------
    t1 : float | Time | None
        Energy-relaxation time in ns. Bare floats are interpreted as ns.
    t2 : float | Time | None
        Total transverse-coherence time in ns. Bare floats are interpreted as
        ns.
    relaxation_rate : float | Frequency | None
        Energy-relaxation rate in inverse ns. Bare floats are interpreted as
        inverse ns.
    dephasing_rate : float | Frequency | None
        Pure-dephasing rate in inverse ns. Bare floats are interpreted as
        inverse ns.

    Returns
    -------
    tuple[float, float]
        Energy-relaxation and pure-dephasing rates, respectively, in inverse
        ns. Unspecified rates are zero.

    Raises
    ------
    ValueError
        If times and rates are both supplied, a time is nonpositive, a rate is
        negative, or `t2` would imply a negative pure-dephasing rate.

    Notes
    -----
    The physical rates satisfy
    `1 / t2 = relaxation_rate / 2 + dephasing_rate`. Rate conversion never
    introduces a `2 * pi` factor.
    """
    uses_coherence_times = t1 is not None or t2 is not None
    uses_rates = relaxation_rate is not None or dephasing_rate is not None
    if uses_coherence_times and uses_rates:
        raise ValueError(
            "Specify either coherence times or rates, not both. "
            "Coherence times are t1 and t2; rates are relaxation_rate "
            "and dephasing_rate."
        )

    normalized_t1 = normalize_time_to_ns(t1)
    normalized_t2 = normalize_time_to_ns(t2)
    normalized_relaxation_rate = (
        normalize_rate_to_inverse_ns(relaxation_rate)
        if relaxation_rate is not None
        else None
    )
    normalized_dephasing_rate = (
        normalize_rate_to_inverse_ns(dephasing_rate)
        if dephasing_rate is not None
        else None
    )

    for name, value in (("t1", normalized_t1), ("t2", normalized_t2)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
    for name, value in (
        ("relaxation_rate", normalized_relaxation_rate),
        ("dephasing_rate", normalized_dephasing_rate),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative.")

    resolved_relaxation_rate = (
        1 / normalized_t1
        if normalized_t1 is not None
        else normalized_relaxation_rate or 0.0
    )
    if normalized_t2 is not None:
        resolved_dephasing_rate = 1 / normalized_t2 - resolved_relaxation_rate / 2
        if resolved_dephasing_rate < 0:
            raise ValueError("t2 must not exceed 2 × t1.")
    else:
        resolved_dephasing_rate = normalized_dephasing_rate or 0.0

    return resolved_relaxation_rate, resolved_dephasing_rate
