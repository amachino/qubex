"""
Count-preserving measurements for the manual bSWAP campaign.

The notebook owns the Experiment connection. This module never connects,
configures, changes reservations, or writes shared calibration parameters.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

from .irb import unitary_key
from .irb_analysis import analyze_irb
from .pulses import compile_campaign, ideal_circuit_unitary, xeb_circuit
from .tomography import BASES, density_from_counts, fit_local_phases, state_vector

if TYPE_CHECKING:
    from qubex.experiment.experiment import Experiment

    from .irb import NativeBSWAPCache


def _json_default(item: Any) -> Any:
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, Path):
        return str(item)
    raise TypeError(type(item).__name__)


def save_json(path: str | Path, value: Any) -> None:
    """Atomically save a JSON result with NumPy values and paths converted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=_json_default))
    temp.replace(path)


def zero_phase_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicit uncorrected phase probe, never a missing-data fallback."""
    record = deepcopy(dict(recipe))
    record["phase_calibration"] = dict(
        pre_active_rad=0.0, post_active_rad=0.0, post_passive_rad=0.0
    )
    record["phase_status"] = "uncorrected calibration probe, not a qualified gate"
    return record


class CampaignMeasurements:
    """
    Keep a caller-owned Experiment session and save every shot before analysis.

    Parameters
    ----------
    exp : Experiment
        Already connected experiment with current production DRAG and classifiers.
        This class does not connect, configure, or grant hardware authorization.
    run : str or Path
        Run location retained for provenance; no metadata file is required.
    recipes : Mapping[str, Mapping[str, Any]]
        Independent full/root recipes, copied when constructing the session.
    qubits : Sequence[str]
        Exactly two distinct labels in active/passive order.
    drive_label, cancel_label : str
        Registered main and cancellation target labels.
    shots : int, optional
        Default number of shots per acquisition, by default 512.
    shot_interval_ns : float, optional
        Repetition interval in ns, by default 1,000,000.
    deadline : datetime or None, optional
        A timezone-aware execution boundary; acquisition stops one minute early.

    Notes
    -----
    Target frequencies and production pulse waveforms are captured at construction.
    The public measurement constraint profile supplies any backend-only prefix
    for carrier phase compensation, without changing the calibrated logical origin.
    The caller must keep the associated RF, classifier and calibration state valid.
    Prepared-state normalization is combined SPAM, not an independent detector model.
    """

    def __init__(
        self,
        exp: Experiment,
        run: str | Path,
        recipes: Mapping[str, Mapping[str, Any]],
        *,
        qubits: Sequence[str],
        drive_label: str,
        cancel_label: str,
        shots: int = 512,
        shot_interval_ns: float = 1_000_000.0,
        deadline: datetime | None = None,
    ) -> None:
        self.exp, self.run = exp, Path(run)
        self.recipes = {
            kind: deepcopy(dict(recipe)) for kind, recipe in recipes.items()
        }
        self.qubits = tuple(qubits)
        if len(self.qubits) != 2 or len(set(self.qubits)) != 2:
            raise ValueError(
                "qubits must specify distinct active/passive labels in order"
            )
        self.drive_label = drive_label
        self.cancel_label = cancel_label
        self.shots, self.shot_interval_ns = int(shots), float(shot_interval_ns)
        self.classifiers = exp.classifiers
        self.x90 = {q: exp.pulse.get_drag_hpi_pulse(q) for q in self.qubits}
        self.xpi = {q: exp.pulse.get_drag_pi_pulse(q) for q in self.qubits}
        self.rabi_scale = exp.pulse.rabi_params[
            self.qubits[0]
        ].frequency / exp.params.get_control_amplitude(self.qubits[0])
        self.references = {q: float(exp.targets[q].frequency) for q in self.qubits}
        self.targets = {
            label: float(exp.targets[label].frequency)
            for label in (self.drive_label, self.cancel_label)
        }
        profile = exp.ctx.measurement.constraint_profile
        self.backend_preamble_ns = (
            profile.extra_capture_duration_ns
            if profile.require_workaround_capture
            else 0.0
        )
        self.deadline = deadline
        if deadline is not None and deadline.tzinfo is None:
            raise ValueError("deadline must include a timezone")
        self.serial = 0
        self.session_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.assignment: NDArray[np.float64] | None = None
        self.assignment_source: str | None = None

    def acquire(
        self,
        gates: Sequence[Any],
        directory: str | Path,
        label: str,
        *,
        prepared: Sequence[str] = ("0", "0"),
        basis: str = "ZZ",
        delay_ns: float = 0.0,
        shots: int | None = None,
        recipes: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Compile one circuit and retain IQ before validating software classification.

        Parameters
        ----------
        gates : Sequence
            Logical gate records accepted by the bSWAP compiler.
        directory : str or Path
            Destination for per-shot NPZ and classified JSON records.
        label : str
            Acquisition label appended to a session-unique filename.
        prepared : Sequence[str], optional
            Active/passive preparation labels, by default ground states.
        basis : str, optional
            Two-qubit Pauli analysis basis, by default ZZ.
        delay_ns : float, optional
            Global delay before preparation, in ns.
        shots : int or None, optional
            Shot override; None uses the session default.
        recipes : Mapping or None, optional
            Explicit recipe snapshot; None uses the session recipes.

        Returns
        -------
        dict[str, Any]
            Raw counts, IQ provenance and optional unclipped SPAM normalization.

        Notes
        -----
        Fixed custom-target references are supplied to both compiler and measure.
        No persistent calibration or hardware configuration is changed here.
        Acquisition start/end timestamps bracket the measure call; its elapsed
        duration uses a monotonic clock. They are not individual-shot hardware
        timestamps. The existing timestamp still records the later classified
        row creation time. Timing and IQ survive a subsequent classifier failure.
        """
        if self.deadline is not None and datetime.now(
            self.deadline.tzinfo
        ) >= self.deadline - timedelta(seconds=60):
            raise TimeoutError(
                "Reservation is ending; stop before another hardware request"
            )
        shots = self.shots if shots is None else int(shots)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        selected = self.recipes if recipes is None else recipes
        sequence, compiled = compile_campaign(
            gates,
            recipes=selected,
            qubits=self.qubits,
            drive_label=self.drive_label,
            cancel_label=self.cancel_label,
            target_frequencies_ghz=self.targets,
            reference_frequencies_ghz=self.references,
            rabi_ghz_per_amplitude=self.rabi_scale,
            x90=self.x90,
            xpi=self.xpi,
            prepared=prepared,
            basis=basis,
            delay_ns=delay_ns,
            backend_preamble_ns=self.backend_preamble_ns,
        )
        # All tones were materialized relative to these fixed target frequencies.
        acquisition_started_at = datetime.now().astimezone().isoformat()
        acquisition_start_clock = perf_counter()
        result = self.exp.measure(
            sequence,
            frequencies=self.targets,
            mode="single",
            n_shots=shots,
            shot_interval=self.shot_interval_ns,
            enable_dsp_classification=False,
            plot=False,
        )
        acquisition_duration_seconds = perf_counter() - acquisition_start_clock
        timing = dict(
            acquisition_started_at=acquisition_started_at,
            acquisition_ended_at=datetime.now().astimezone().isoformat(),
            acquisition_duration_seconds=acquisition_duration_seconds,
        )
        iq = np.stack(
            [np.asarray(result.data[q].kerneled).reshape(-1) for q in self.qubits]
        )
        self.serial += 1
        path = directory / f"{self.session_id}_{self.serial:06d}_{label}.npz"
        np.savez_compressed(
            path, iq=iq, qubits=self.qubits, prepared=prepared, basis=basis, **timing
        )
        labels = np.stack(
            [self.classifiers[q].predict(iq[i]) for i, q in enumerate(self.qubits)]
        )
        if (
            iq.shape != (2, shots)
            or labels.shape != (2, shots)
            or not np.isin(labels, [0, 1]).all()
        ):
            raise ValueError(
                "Unexpected shot count/classifier alphabet; raw IQ retained"
            )
        labels = labels.astype(np.int64)
        counts = np.bincount(2 * labels[0] + labels[1], minlength=4)
        row: dict[str, Any] = dict(
            label=label,
            iq_file=str(path),
            counts=counts.tolist(),
            shots=shots,
            gates=gates,
            prepared=prepared,
            basis=basis,
            delay_ns=delay_ns,
            compiled=compiled,
            timestamp=datetime.now().astimezone().isoformat(),
            **timing,
        )
        row["raw_probabilities"] = (counts / shots).tolist()
        if self.assignment is not None:
            corrected = np.linalg.solve(self.assignment, counts / shots)
            row.update(
                mitigated_probabilities_unclipped=corrected.tolist(),
                mitigation_source=self.assignment_source,
                mitigation_scope="prepared-state response inverse; includes thermal/preparation error; not detector-only",
            )
        save_json(path.with_suffix(".json"), row)
        print(f"MEASURE {label} counts={counts.tolist()}", flush=True)
        return row

    def calibrate_assignment(
        self, directory: str | Path, *, shots: int = 4096
    ) -> dict[str, Any]:
        """
        Measure the prepared-state response with reported outcomes as matrix rows.

        Parameters
        ----------
        directory : str or Path
            Destination for four-state counts and the response record.
        shots : int, optional
            Shots per prepared label, by default 4096.

        Returns
        -------
        dict[str, Any]
            C[reported, prepared], conditioning and raw acquisition provenance.

        Notes
        -----
        Columns are prepared labels, not independently established pure states.
        The inverse is combined-SPAM normalization; negative or above-one results
        remain unclipped and cannot certify physical transfer or leakage.
        """
        directory = Path(directory)
        self.assignment = None
        self.assignment_source = None
        states = [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")]
        rows = [
            self.acquire(
                [],
                directory,
                "assignment_" + "".join(state),
                prepared=state,
                shots=shots,
            )
            for state in states
        ]
        matrix = np.column_stack([np.asarray(row["counts"]) / shots for row in rows])
        condition = float(np.linalg.cond(matrix))
        report = dict(
            matrix=matrix.tolist(),
            convention="C[reported, prepared]",
            states=["".join(s) for s in states],
            shots=shots,
            condition_number=condition,
            scope="combined preparation/readout normalization; thermal excitation is not removed from the physical state",
            clipping=False,
            rows=rows,
        )
        source = directory / "prepared_state_response.json"
        save_json(source, report)
        if not np.isfinite(condition) or condition > 100:
            raise ValueError(
                "Prepared-state response is too ill-conditioned for mitigation; raw data retained"
            )
        self.assignment, self.assignment_source = matrix, str(source)
        return report

    def tomography(
        self,
        gates: Sequence[Any],
        state: Sequence[str],
        directory: str | Path,
        name: str,
        *,
        recipes: Mapping[str, Mapping[str, Any]] | None = None,
        shots: int | None = None,
        delay_ns: float = 0.0,
    ) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
        """
        Acquire all nine Pauli bases and save raw and optional normalized states.

        Returns
        -------
        tuple[NDArray, NDArray]
            Raw linear density matrix and the nine four-outcome count rows.
            Optional normalized output is saved separately without PSD projection.
        """
        counts: list[list[int]] = []
        mitigated: list[list[float]] = []
        for basis in BASES:
            row = self.acquire(
                gates,
                directory,
                f"{name}_{basis}",
                prepared=state,
                basis=basis,
                recipes=recipes,
                shots=shots,
                delay_ns=delay_ns,
            )
            counts.append(row["counts"])
            if "mitigated_probabilities_unclipped" in row:
                mitigated.append(row["mitigated_probabilities_unclipped"])
        rho = density_from_counts(counts)
        extra: dict[str, Any] = {}
        if mitigated:
            extra = dict(
                mitigated_probabilities_unclipped=mitigated,
                mitigated_rho_unclipped=density_from_counts(mitigated),
            )
        np.savez_compressed(
            Path(directory) / f"{name}_tomography.npz",
            counts=counts,
            rho=rho,
            bases=BASES,
            **extra,
        )
        return rho, np.asarray(counts, dtype=np.int64)

    def phase_calibrate(
        self,
        kind: str,
        recipe: Mapping[str, Any],
        directory: str | Path,
        *,
        shots: int | None = None,
    ) -> tuple[dict[str, Any], NDArray[np.int64]]:
        """
        Fit primitive local phases and residual ZZ from four independent inputs.

        Returns
        -------
        tuple[dict[str, Any], NDArray]
            Recipe with measured phases and all input/basis count rows.
            The record remains pending independent held-out validation.

        Notes
        -----
        This calibrates the supplied physical pulse. It does not refit a target
        on benchmark data or report a gate fidelity.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        selected = deepcopy(self.recipes)
        selected[kind] = zero_phase_recipe(recipe)
        gate_name = "BSWAP" if kind == "bswap" else "RAW_SQRT_BSWAP"
        states = [("0", "+"), ("1", "+"), ("+", "0"), ("+", "1")]
        rhos, all_counts = [], []
        for i, state in enumerate(states):
            rho, counts = self.tomography(
                [gate_name],
                state,
                directory,
                f"phase_{i}",
                recipes=selected,
                shots=shots,
            )
            rhos.append(rho)
            all_counts.append(counts)
        fit = fit_local_phases(kind, states, rhos)
        record = deepcopy(dict(recipe))
        record.update(
            phase_calibration=fit,
            pre_vz_rad=fit["pre_vz_rad"],
            post_vz_rad=fit["post_vz_rad"],
            zz_phase_rad=fit["zz_phase_rad"],
            gate_start_ns=max(
                p.duration for p in [*self.x90.values(), *self.xpi.values()]
            ),
            phase_data_directory=str(directory),
            phase_status="measured; held-out validation pending",
        )
        save_json(directory / "phase_calibration.json", record)
        return record, np.asarray(all_counts, dtype=np.int64)

    def gate_checks(
        self, names: Sequence[str], directory: str | Path, *, shots: int = 512
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Acquire phase-sensitive, placement and repetition diagnostics for each gate.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Populations and input-state overlaps against separate zero-ZZ ideal
            and frozen raw-ZZ models, never measured gate fidelity.
        """
        directory = Path(directory)
        results: dict[str, list[dict[str, Any]]] = {}
        states = [
            ("0", "0"),
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
            ("+", "0"),
            ("0", "+"),
            ("+", "+"),
        ]
        for name in names:
            rows: list[dict[str, Any]] = []
            zz = {
                kind: float(recipe["zz_phase_rad"])
                for kind, recipe in self.recipes.items()
            }
            for n in (1, 2, 4):
                gates = [name] * n
                u = ideal_circuit_unitary(gates)
                raw_u = ideal_circuit_unitary(gates, zz_phases=zz)
                for index, state in enumerate(states):
                    # Four basis inputs have population checks; equators have tomography.
                    if index < 4:
                        row = self.acquire(
                            gates,
                            directory / name,
                            f"n{n}_state{index}",
                            prepared=state,
                            shots=shots,
                        )
                        entry: dict[str, Any] = dict(
                            repeats=n,
                            state=state,
                            populations=(np.asarray(row["counts"]) / shots).tolist(),
                        )
                        if "mitigated_probabilities_unclipped" in row:
                            entry["mitigated_probabilities_unclipped"] = row[
                                "mitigated_probabilities_unclipped"
                            ]
                        rows.append(entry)
                    else:
                        rho, _ = self.tomography(
                            gates,
                            state,
                            directory / name,
                            f"n{n}_state{index}",
                            shots=shots,
                        )
                        vector = u @ state_vector(state)
                        raw_vector = raw_u @ state_vector(state)
                        rows.append(
                            dict(
                                repeats=n,
                                state=state,
                                ideal_state_overlap=float(
                                    np.vdot(vector, rho @ vector).real
                                ),
                                raw_model_state_overlap=float(
                                    np.vdot(raw_vector, rho @ raw_vector).real
                                ),
                            )
                        )
            for delay in (2.0, 16.0, 64.0, 256.0):
                rho, _ = self.tomography(
                    [name],
                    ("0", "+"),
                    directory / name,
                    f"delay{int(delay)}",
                    shots=shots,
                    delay_ns=delay,
                )
                rows.append(
                    dict(
                        delay_ns=delay,
                        state=("0", "+"),
                        rho_real=rho.real.tolist(),
                        rho_imag=rho.imag.tolist(),
                    )
                )
            results[name] = rows
            save_json(directory / f"{name}_checks.json", rows)
        return results


def make_irb_circuit(
    cache: NativeBSWAPCache, depth: int, seed: int, gate_name: str | None = None
) -> list[str]:
    """Build a paired native-full circuit with an explicit ideal Clifford inverse."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(cache.sequences), size=int(depth))
    gates, u = [], np.eye(4, dtype=complex)
    for index in indices:
        gates.extend(cache.sequences[index])
        u = cache.unitaries[index] @ u
        if gate_name is not None:
            gates.append(gate_name)
            u = ideal_circuit_unitary([gate_name]) @ u
    gates.extend(cache.lookup[unitary_key(u.conj().T)])
    error = 1 - abs(np.trace(ideal_circuit_unitary(gates))) ** 2 / 16
    if abs(error) > 1e-9:
        raise ValueError("Ideal Clifford inverse does not close")
    return gates


def acquire_irb(
    measurements: CampaignMeasurements,
    cache: NativeBSWAPCache,
    gate_name: str,
    directory: str | Path,
    *,
    depths: Sequence[int] | np.ndarray,
    seeds: Sequence[int] | np.ndarray,
    shots: int = 512,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    """
    Acquire matched reference/interleaved circuits and preserve partial arrays.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Existing caller-owned session with current preparation and classifier.
    cache : NativeBSWAPCache
        Saved native-full Clifford table; this function does not regenerate it.
    gate_name : str
        Ideal Clifford block such as BSWAP, ROOT_PAIR or XX90, not a single root.
    directory : str or Path
        Destination for per-shot records, partial arrays and analysis.
    depths, seeds : Sequence[int]
        Matched random-Clifford depths and seed identities.
    shots : int, optional
        Shots per circuit, by default 512.

    Returns
    -------
    tuple[dict[str, Any], dict[str, NDArray]]
        Raw IRB analysis and raw reference/interleaved survival arrays.
        Unclipped prepared-state normalization is a separate diagnostic only.

    Notes
    -----
    Randomize (seed, depth) pair units, acquiring REF/IRB adjacently with
    randomized, globally balanced first conditions. Circuit seeds and shot
    counts are unchanged. A saved manifest and raw rows retain pair identity,
    acquisition index, block identity and measure-call timing. Here a block
    is one adjacent pair, not an independent complete temporal replicate.
    The existing bootstrap still resamples paired circuit seeds across depths;
    its conditional statistical interval is not a drift-robust time-block CI.
    """
    if np.asarray(depths).ndim != 1 or np.asarray(seeds).ndim != 1:
        raise ValueError("depths and seeds must be one-dimensional")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        mode: np.full((len(seeds), len(depths)), np.nan)
        for mode in ("reference", "interleaved")
    }
    mitigated = {mode: np.full((len(seeds), len(depths)), np.nan) for mode in arrays}
    pairs = [(s, d) for s in range(len(seeds)) for d in range(len(depths))]
    order_seed = 82391
    rng = np.random.default_rng(order_seed)
    pair_order = rng.permutation(len(pairs))
    first_modes = (np.arange(len(pairs)) + rng.integers(2)) % 2
    rng.shuffle(first_modes)
    modes = ("reference", "interleaved")
    requests: list[dict[str, Any]] = []
    for block_id, (pair_index, first) in enumerate(
        zip(pair_order, first_modes, strict=True)
    ):
        s, d = pairs[int(pair_index)]
        for mode in (modes[int(first)], modes[1 - int(first)]):
            requests.append(
                dict(
                    pair_id=f"seed_index_{s}_depth_index_{d}",
                    block_id=block_id,
                    acquisition_index=len(requests),
                    seed_index=s,
                    depth_index=d,
                    seed=int(seeds[s]),
                    depth=int(depths[d]),
                    mode=mode,
                    status="planned",
                )
            )
    manifest_file = directory / "irb_acquisition_manifest.json"
    manifest = dict(
        order_seed=order_seed,
        pair_count=len(pairs),
        circuit_count=len(requests),
        shots=shots,
        depths=list(depths),
        seeds=list(seeds),
        gate_name=gate_name,
        order="shuffled seed/depth pairs; adjacent REF/IRB with balanced randomized first condition",
        block_definition="one adjacent REF/IRB pair, not a complete temporal replicate",
        bootstrap_unit="paired circuit seed across depths and conditions, not time blocks",
        drift_robust_interval=False,
        requests=requests,
    )
    save_json(manifest_file, manifest)
    for request in requests:
        s, d, mode = request["seed_index"], request["depth_index"], request["mode"]
        gates = make_irb_circuit(
            cache,
            int(depths[d]),
            int(seeds[s]),
            gate_name if mode == "interleaved" else None,
        )
        request["status"] = "acquiring"
        save_json(manifest_file, manifest)
        row = measurements.acquire(
            gates, directory, f"{mode}_s{seeds[s]}_d{depths[d]}", shots=shots
        )
        request.update(
            status="completed",
            **{
                key: row[key]
                for key in (
                    "iq_file",
                    "timestamp",
                    "acquisition_started_at",
                    "acquisition_ended_at",
                    "acquisition_duration_seconds",
                )
            },
        )
        row.update(request, acquisition_manifest=str(manifest_file))
        save_json(Path(row["iq_file"]).with_suffix(".json"), row)
        save_json(manifest_file, manifest)
        arrays[mode][s, d] = row["counts"][0] / shots
        if "mitigated_probabilities_unclipped" in row:
            mitigated[mode][s, d] = row["mitigated_probabilities_unclipped"][0]
        np.savez_compressed(
            directory / "irb_partial.npz",
            **arrays,
            depths=depths,
            seeds=seeds,
            mitigated_reference_unclipped=mitigated["reference"],
            mitigated_interleaved_unclipped=mitigated["interleaved"],
        )
    summary = analyze_irb(depths, arrays["reference"], arrays["interleaved"])
    summary["target"] = f"ideal {gate_name}; residual ZZ counts as error"
    summary["interleaved_block"] = gate_name
    summary["acquisition_manifest"] = str(manifest_file)
    summary["bootstrap_unit"] = manifest["bootstrap_unit"]
    summary["drift_robust_interval"] = False
    if all(np.isfinite(values).all() for values in mitigated.values()):
        fits = {
            mode: fit_unclipped_decay(depths, values)
            for mode, values in mitigated.items()
        }
        summary["mitigated_unclipped"] = dict(
            fits=fits,
            scope="combined-SPAM diagnostic; not a pure-readout corrected gate-fidelity claim",
        )
    save_json(directory / "irb_analysis.json", summary)
    return summary, arrays


def fit_unclipped_decay(depths: ArrayLike, values: ArrayLike) -> dict[str, Any]:
    """
    Fit a diagnostic decay without clipping normalized negative or above-one data.

    Returns
    -------
    dict[str, Any]
        Input range, means, uncertainties and numerical fit or failure detail.
        This diagnostic is not used as a raw IRB gate-fidelity estimate.
    """
    x, values = np.asarray(depths), np.asarray(values)
    mean = values.mean(axis=0)
    sem = np.maximum(values.std(axis=0, ddof=1) / np.sqrt(len(values)), 0.002)
    result: dict[str, Any] = dict(
        depths=x.tolist(),
        means=mean.tolist(),
        sem=sem.tolist(),
        input_min=float(values.min()),
        input_max=float(values.max()),
        clipping=False,
    )
    used = x > 0
    try:
        pars, cov = curve_fit(
            lambda t, a, p, b: a * p**t + b,
            x[used],
            mean[used],
            p0=(0.8, 0.95, 0.25),
            sigma=sem[used],
            absolute_sigma=True,
            bounds=([-3, 0.00001, -3], [3, 1, 3]),
            maxfev=10000,
        )
        result.update(
            amplitude=float(pars[0]),
            p=float(pars[1]),
            floor=float(pars[2]),
            covariance=cov.tolist(),
        )
    except (RuntimeError, ValueError) as error:
        result["error"] = str(error)
    return result


def summarize_xeb(
    rows: Sequence[Mapping[str, Any]],
    depths: Sequence[int] | np.ndarray,
    *,
    mitigated: bool = False,
    probability_key: str = "ideal_probabilities",
) -> dict[str, Any]:
    """
    Regress finite-ensemble XEB observables against an explicitly selected target.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Circuit records with target probabilities and counts or normalized values.
    depths : Sequence[int]
        Requested cycle depths; no low-contrast depth is silently removed.
    mitigated : bool, optional
        Use unclipped combined-SPAM probabilities instead of raw counts.
    probability_key : str, optional
        Row field containing the frozen target probabilities. The compatibility
        field `ideal_probabilities` holds the primary target, which may retain ZZ.

    Returns
    -------
    dict[str, Any]
        Circuit/cycle XEB scores, seed-bootstrap errors and decay diagnostics.
        No unconditional single-gate fidelity conversion is performed.
    """
    scores, errors = [], []
    for depth in depths:
        group = [r for r in rows if r["depth"] == depth]
        p = np.asarray([r[probability_key] for r in group])
        if mitigated:
            q = np.asarray(
                [r["mitigated_probabilities_unclipped"] for r in group], dtype=float
            )
        else:
            q = np.asarray([r["counts"] for r in group], dtype=float)
            q /= q.sum(axis=1)[:, None]
        d = np.sum((p - 0.25) ** 2, axis=1)
        n = np.sum((p - 0.25) * (q - 0.25), axis=1)
        if np.sum(d * d) < 1e-10:
            raise ValueError("XEB ideal distribution has no discriminating contrast")
        score = float(np.sum(d * n) / np.sum(d * d))
        draws = []
        rng = np.random.default_rng(9841 + int(depth))
        for _ in range(300):
            selected = rng.integers(len(d), size=len(d))
            draws.append(np.sum(d[selected] * n[selected]) / np.sum(d[selected] ** 2))
        scores.append(score)
        errors.append(float(np.std(draws, ddof=1)))
    result: dict[str, Any] = dict(
        depths=list(depths),
        scores=scores,
        errors=errors,
        ensemble="8 XY DRAG-X90 axes x 8 VZ angles independently on both qubits",
        estimator="sum(d*n)/sum(d*d), circuit-observable regression",
        claim="circuit/cycle XEB; no automatic single-gate fidelity conversion",
        probability_key=probability_key,
    )
    result["mitigation"] = "combined-SPAM inverse, unclipped" if mitigated else "none"
    try:
        model = lambda x, a, p, b: a * p**x + b
        pars, cov = curve_fit(
            model,
            depths,
            scores,
            sigma=np.maximum(errors, 0.002),
            p0=(1.0, 0.97, 0.0),
            bounds=([0.0, 0.00001, -0.3], [1.5, 1.0, 0.3]),
            absolute_sigma=True,
            maxfev=10000,
        )
        residual = np.asarray(scores) - model(np.asarray(depths), *pars)
        chi = float(
            np.sum((residual / np.maximum(errors, 0.002)) ** 2)
            / max(1, len(depths) - 3)
        )
        usable = bool(
            np.isfinite(cov).all()
            and np.sqrt(max(cov[1, 1], 0)) < 0.03
            and chi < 5
            and pars[1] ** max(depths) < 0.1
        )
        result.update(
            amplitude=float(pars[0]),
            cycle_p=float(pars[1]),
            floor=float(pars[2]),
            covariance=cov.tolist(),
            reduced_chi2=chi,
            decay_identified=usable,
        )
    except (RuntimeError, ValueError) as error:
        result.update(decay_identified=False, error=str(error))
    return result


def acquire_xeb(
    measurements: CampaignMeasurements,
    gate_name: str | None,
    directory: str | Path,
    *,
    depths: Sequence[int] | np.ndarray,
    seeds: Sequence[int] | np.ndarray,
    shots: int = 512,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Acquire 64-pattern XEB with frozen raw-ZZ and zero-ZZ targets from the same shots.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Existing session. Recipe and local-phase/ZZ calibration snapshots are
        taken before any acquisition and used unchanged for every circuit.
    gate_name : str or None
        BSWAP, RAW_SQRT_BSWAP, ROOT_PAIR, XX90, or a local-only reference.
    directory : str or Path
        Destination for frozen targets, raw IQ/counts and both target analyses.
    depths, seeds : Sequence[int]
        Positive cycle depths and seeds; the generator supplies XY/VZ pattern indices.
        Depth zero is unidentifiable with the terminal equatorial local layer.
    shots : int, optional
        Shots per circuit, by default 512.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        Primary raw-ZZ model circuit/cycle XEB and all acquisition rows.
        `zero_zz_diagnostic` compares the same counts to ideal zero-ZZ gates.
        Each target also has a separate unclipped SPAM diagnostic when available.

    Notes
    -----
    Targets are independently calibrated models, not fitted to these XEB shots.
    The compatibility row field `ideal_probabilities` stores the primary raw-model
    probabilities; both explicitly named probability arrays are also retained.
    XEB scores and decay parameters are not unconditional hardware gate fidelities.
    Existing XEB outputs in the destination are never overwritten; use a new
    directory for another acquisition attempt.
    """
    if np.asarray(depths).ndim != 1 or np.asarray(seeds).ndim != 1:
        raise ValueError("depths and seeds must be one-dimensional")
    if np.any(np.asarray(depths) == 0):
        raise ValueError(
            "XEB depth 0 has no target contrast with the terminal local layer"
        )
    directory = Path(directory)
    if any(
        (directory / filename).exists()
        for filename in (
            "xeb_frozen_targets.json",
            "xeb_partial.json",
            "xeb_summary.json",
        )
    ):
        raise FileExistsError(
            "XEB outputs already exist; use a new acquisition directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    frozen_recipes = deepcopy(measurements.recipes)
    frozen_zz = {
        kind: float(recipe["zz_phase_rad"])
        for kind, recipe in frozen_recipes.items()
        if "zz_phase_rad" in recipe
    }
    required_kind = (
        "bswap"
        if gate_name == "BSWAP"
        else "sqrt_bswap"
        if gate_name is not None
        else None
    )
    if required_kind is not None and required_kind not in frozen_zz:
        raise ValueError("Raw-model XEB requires an independently calibrated ZZ phase")
    if not all(np.isfinite(value) for value in frozen_zz.values()):
        raise ValueError("Frozen ZZ phases must be finite")
    target_name = gate_name or "local-only"
    raw_target = f"frozen independently calibrated raw-ZZ model {target_name}"
    zero_target = f"frozen zero-ZZ ideal {target_name}"
    target_record = dict(
        frozen_at=datetime.now().astimezone().isoformat(),
        primary_target=raw_target,
        comparison_target=zero_target,
        recipes=frozen_recipes,
        zz_phases_rad=frozen_zz,
        phase_calibration_provenance={
            kind: dict(
                phase_calibration=recipe.get("phase_calibration"),
                phase_data_directory=recipe.get("phase_data_directory"),
                calibration_data_directory=recipe.get("calibration_data_directory"),
                gate_start_ns=recipe.get("gate_start_ns"),
            )
            for kind, recipe in frozen_recipes.items()
        },
        target_refitted_on_benchmark=False,
        claim="circuit/cycle XEB against declared targets, not unconditional gate fidelity",
    )
    target_file = directory / "xeb_frozen_targets.json"
    save_json(target_file, target_record)
    target_fingerprint = sha256(target_file.read_bytes()).hexdigest()
    requests = [(depth, seed) for seed in seeds for depth in depths]
    rows: list[dict[str, Any]] = []
    for index in np.random.default_rng(721).permutation(len(requests)):
        depth, seed = requests[index]
        plan = xeb_circuit(int(depth), int(seed), gate_name)
        gates = plan["gates"]
        raw_model = np.abs(ideal_circuit_unitary(gates, zz_phases=frozen_zz)[:, 0]) ** 2
        zero_zz = np.abs(ideal_circuit_unitary(gates)[:, 0]) ** 2
        row = measurements.acquire(
            gates,
            directory,
            f"xeb_s{seed}_d{depth}",
            shots=shots,
            recipes=deepcopy(frozen_recipes),
        )
        row.update(
            depth=int(depth),
            seed=int(seed),
            ideal_probabilities=raw_model.tolist(),
            raw_model_probabilities=raw_model.tolist(),
            zero_zz_ideal_probabilities=zero_zz.tolist(),
            local_indices=plan["local_indices"],
            gate_name=gate_name,
            target=raw_target,
            zero_zz_target=zero_target,
            target_refitted_on_benchmark=False,
            frozen_target_file=str(target_file),
            frozen_target_sha256=target_fingerprint,
        )
        save_json(Path(row["iq_file"]).with_suffix(".json"), row)
        rows.append(row)
        save_json(directory / "xeb_partial.json", rows)
    summary = summarize_xeb(rows, depths)
    comparison = summarize_xeb(
        rows, depths, probability_key="zero_zz_ideal_probabilities"
    )
    summary["target"] = raw_target
    comparison["target"] = zero_target
    if all("mitigated_probabilities_unclipped" in row for row in rows):
        summary["mitigated_unclipped"] = summarize_xeb(rows, depths, mitigated=True)
        summary["mitigated_unclipped"]["target"] = raw_target
        comparison["mitigated_unclipped"] = summarize_xeb(
            rows, depths, mitigated=True, probability_key="zero_zz_ideal_probabilities"
        )
        comparison["mitigated_unclipped"]["target"] = zero_target
    summary["zero_zz_diagnostic"] = comparison
    summary["frozen_target_file"] = str(target_file)
    summary["frozen_target_sha256"] = target_fingerprint
    summary["target_refitted_on_benchmark"] = False
    save_json(directory / "xeb_summary.json", summary)
    return summary, rows


def plot_decay(summary: Mapping[str, Any], title: str) -> go.Figure:
    """Plot circuit/cycle XEB scores with the explicitly declared target label."""
    fig = go.Figure()
    fig.add_scatter(
        x=summary["depths"],
        y=summary["scores"],
        error_y=dict(type="data", array=summary["errors"]),
        mode="markers",
        name=summary.get("target", "XEB"),
    )
    fig.update_layout(
        title=f"{title}<br><sup>{summary.get('target', 'Target not specified')}</sup>",
        xaxis_title="Cycle depth",
        yaxis_title="Circuit/cycle XEB score (unclipped)",
    )
    return fig
