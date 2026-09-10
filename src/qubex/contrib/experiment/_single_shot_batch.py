"""Acquire ordered single-shot schedules for contrib experiments."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge
from qubex.experiment import Experiment
from qubex.pulse import PulseSchedule


def measure_single_shot_batch(
    exp: Experiment,
    schedules: Sequence[PulseSchedule],
    *,
    n_shots: int,
    shot_interval: float,
) -> list[dict[str, NDArray[np.complex128]]]:
    """
    Acquire a single-shot sweep and validate its ordered IQ payloads.

    Parameters
    ----------
    exp
        Experiment providing readout timing and the async measurement service.
    schedules
        Ordered preparation/analyzer schedules without readout. Each callback
        returns a copy so readout construction cannot mutate the inputs.
        An empty sequence returns an empty list without accessing hardware.
    n_shots
        Validated shot count, at least two for the contrib estimators.
    shot_interval
        Validated positive interval in ns.

    Returns
    -------
    list[dict[str, NDArray[np.complex128]]]
        One mapping per input schedule, containing integrated IQ arrays of
        shape `(n_shots,)` keyed by canonical measured target.

    Raises
    ------
    ValueError
        If point counts, capture counts, or finite single-shot shapes differ
        from the acquisition contract. Callers validate required target labels.

    Notes
    -----
    One API call can produce multiple hardware executions depending on backend
    support and schedule packing. The shared async bridge supports synchronous
    calls from notebooks; its wait budget includes estimated acquisition time
    and an overhead allowance, not a strict hardware duration limit.
    """
    if not schedules:
        return []
    schedules = tuple(schedules)
    readout_duration = (
        exp.pulse.readout_duration
        + exp.pulse.readout_pre_margin
        + exp.pulse.readout_post_margin
    )
    # Allow the complete acquisition plus a bridge overhead budget in notebooks.
    acquisition_seconds = (
        n_shots
        * sum(
            schedule.duration + shot_interval + readout_duration
            for schedule in schedules
        )
        * 1e-9
    )
    timeout = DEFAULT_TIMEOUT_SECONDS + 2 * acquisition_seconds
    result = get_shared_async_bridge(key="experiment").run(
        lambda: exp.measurement_service.run_sweep_measurement(
            lambda index: schedules[int(index)].copy(),
            sweep_values=np.arange(len(schedules)),
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=False,
            time_integration=True,
            state_classification=False,
            final_measurement=True,
            plot=False,
            enable_tqdm=False,
        ),
        timeout=timeout,
    )
    if len(result.results) != len(schedules):
        raise ValueError("Single-shot batch returned an unexpected number of points.")
    shots_by_point: list[dict[str, NDArray[np.complex128]]] = []
    for index, point in enumerate(result.results):
        shots_by_target: dict[str, NDArray[np.complex128]] = {}
        for target, captures in point.data.items():
            if len(captures) != 1:
                raise ValueError(
                    f"Batch point {index}, {target}: expected one capture."
                )
            iq = np.asarray(captures[0].data, dtype=np.complex128)
            if iq.shape != (n_shots,) or not np.all(np.isfinite(iq)):
                raise ValueError(
                    f"Batch point {index}, {target}: expected {n_shots} finite IQ shots."
                )
            shots_by_target[target] = iq
        shots_by_point.append(shots_by_target)
    return shots_by_point
