"""Acquire ordered single-shot schedules for contrib experiments."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from qubex.backend.quel1 import Quel1BackendController
from qubex.core.async_bridge import get_shared_async_bridge
from qubex.experiment import Experiment
from qubex.measurement.models import MeasurementSchedule
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.pulse import PulseSchedule


def measure_single_shot_batch(
    exp: Experiment,
    schedules: Sequence[PulseSchedule],
    *,
    n_shots: int,
    shot_interval: float,
    enable_tqdm: bool = False,
    capture_index: int | None = None,
    final_capture_only: bool = False,
) -> list[dict[str, NDArray[np.complex128]]]:
    """
    Acquire a single-shot sweep and validate its ordered IQ payloads.

    Parameters
    ----------
    exp
        Experiment providing readout timing and the async measurement service.
    schedules
        Ordered schedules before the automatically appended final readout. They
        may already contain readout pulses and captures. Each callback returns a
        copy so final-readout construction cannot mutate the inputs. An empty
        sequence returns an empty list without accessing hardware.
    n_shots
        Validated shot count, at least two for the contrib estimators.
    shot_interval
        Validated positive interval in ns.
    enable_tqdm
        Whether the underlying sweep displays its progress bar.
    capture_index
        Capture selected independently for each target and sweep point. Negative
        indexing is supported. By default, require exactly one capture and
        select it. Indices refer to the captures retained for acquisition.
    final_capture_only
        Whether to acquire only the final capture for each target while keeping
        the complete pulse schedule. QuEL-1 always acquires all capture windows.

    Returns
    -------
    list[dict[str, NDArray[np.complex128]]]
        One mapping per input schedule, containing integrated IQ arrays of
        shape `(n_shots,)` keyed by canonical measured target.

    Raises
    ------
    ValueError
        If point counts, the requested capture, or finite single-shot shapes
        differ from the acquisition contract. Callers validate required target
        labels.
    TypeError
        If `capture_index` is not an integer or `None`.

    Notes
    -----
    One API call can produce multiple hardware executions depending on backend
    support and schedule packing. The shared async bridge supports synchronous
    calls from notebooks without imposing a bridge wait timeout.
    """
    if capture_index is not None and (
        isinstance(capture_index, bool) or not isinstance(capture_index, Integral)
    ):
        raise TypeError("capture_index must be an integer or None.")
    resolved_capture_index = None if capture_index is None else int(capture_index)
    if not schedules:
        return []
    schedules = tuple(schedules)

    def build_schedule(index: int) -> PulseSchedule | MeasurementSchedule:
        pulse_schedule = schedules[index].copy()
        if not final_capture_only or isinstance(
            exp.backend_controller, Quel1BackendController
        ):
            return pulse_schedule
        measurement = exp.build_measurement_schedule(
            pulse_schedule, final_measurement=True
        )
        # Replace the model to avoid retaining the cached `channels` mapping.
        return MeasurementSchedule(
            pulse_schedule=measurement.pulse_schedule,
            capture_schedule=CaptureSchedule(
                captures=[
                    windows[-1]
                    for windows in measurement.capture_schedule.channels.values()
                    if windows
                ]
            ),
        )

    result = get_shared_async_bridge(key="experiment").run_without_timeout(
        lambda: exp.measurement_service.run_sweep_measurement(
            lambda index: build_schedule(int(index)),
            sweep_values=np.arange(len(schedules)),
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=False,
            time_integration=True,
            state_classification=False,
            final_measurement=True,
            plot=False,
            enable_tqdm=enable_tqdm,
        ),
    )
    if len(result.results) != len(schedules):
        raise ValueError("Single-shot batch returned an unexpected number of points.")
    shots_by_point: list[dict[str, NDArray[np.complex128]]] = []
    for index, point in enumerate(result.results):
        shots_by_target: dict[str, NDArray[np.complex128]] = {}
        for target, captures in point.data.items():
            capture_count = len(captures)
            if resolved_capture_index is None:
                if capture_count != 1:
                    raise ValueError(
                        f"Batch point {index}, {target}: expected one capture."
                    )
                capture = captures[0]
            else:
                try:
                    capture = captures[resolved_capture_index]
                except IndexError as error:
                    raise ValueError(
                        f"Batch point {index}, {target}: capture_index "
                        f"{resolved_capture_index} is out of range for "
                        f"{capture_count} "
                        f"{'capture' if capture_count == 1 else 'captures'}."
                    ) from error
            iq = np.asarray(capture.data, dtype=np.complex128)
            if iq.shape != (n_shots,) or not np.all(np.isfinite(iq)):
                raise ValueError(
                    f"Batch point {index}, {target}: expected {n_shots} finite IQ shots."
                )
            shots_by_target[target] = iq
        shots_by_point.append(shots_by_target)
    return shots_by_point
