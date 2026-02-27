# QuEL-3 Demo Readiness (2026-03-19)

## Goal

Prepare executable demo paths for:

1. qubit frequency identification
2. 1Q Rabi oscillation
3. 1Q gate calibration and benchmarking
4. 2Q gate calibration and benchmarking
5. Bell-state generation/measurement
6. multi-qubit experiment demo

## Current source-based status (as of 2026-02-27)

| Demo item | Status | Evidence in current code | Required work before demo |
| --- | --- | --- | --- |
| qubit frequency identification | BLOCKED | `CharacterizationService.scan_qubit_frequencies()` / `scan_resonator_frequencies()` / `measure_electrical_delay()` call `SystemManager.modified_backend_settings(...)`; QuEL-3 path does not support that capability today. | Define QuEL-3 sweep semantics and implement QuEL-3-safe path. |
| 1Q Rabi oscillation | BLOCKED | `MeasurementService.rabi_experiment()` uses `sweep_parameter()`, and `sweep_parameter()` unconditionally calls `reset_awg_and_capunits()`. QuEL-3 backend currently lacks this reset capability. | Add QuEL-3 reset strategy or remove hard dependency in sweep paths. |
| 1Q gate calibration + benchmarking | BLOCKED | Calibration paths depend on the same sweep/reset chain; `BenchmarkingService.rb_experiment_1q()` defaults `reset_awg_and_capunits=True`. | Same as above + verify 1Q RB on QuEL-3 hardware. |
| 2Q gate calibration + benchmarking | BLOCKED | 2Q paths still rely on reset-dependent measurement flows. Multi-alias execution restriction was removed in QuEL-3 execution path on `2026-02-27`, but hardware validation evidence is not attached yet. | Complete reset-safe execution path and attach cross-unit synchronized-trigger hardware evidence. |
| Bell-state demo | BLOCKED | `MeasurementService.measure_bell_state()` defaults `reset_awg_and_capunits=True`; tomography calls it without overriding reset behavior. | Make Bell path QuEL-3-capability aware and validate 2Q execution. |
| multi-qubit demo | BLOCKED | Several high-level `Experiment` APIs are moved/deprecated with `NotImplementedError`; remaining paths still inherit unresolved QuEL-3 blockers above. | Select concrete contrib flow and clear all upstream blockers first. |

## Deep dive: qubit frequency identification gap

### What is QuEL-1-specific today

- Frequency scans are implemented with subrange coarse retuning:
  - compute `LO/CNCO` from `MixingUtil.calc_lo_cnco(...)`
  - apply retune by `SystemManager.modified_backend_settings(...)`
  - continue per-point sweep with `modified_frequencies(...)`
- This design assumes availability of:
  - backend settings mutation APIs (`config_port`, `config_channel`, `config_runit`)
  - backend settings cache sync/pull
  - explicit AWG/CAP reset path

### Why QuEL-3 is unclear

- QuEL-3 synchronizer is currently a no-op and has no `dump_box`-equivalent pull path.
- QuEL-3 backend controller does not expose QuEL-1-style backend setting mutation/reset APIs.
- Therefore the current `CharacterizationService` implementation cannot be used as-is for QuEL-3 frequency identification.

### Minimum contract required for v1.5.0 beta

1. Decide QuEL-3 coarse/fine tuning contract for spectroscopy.
2. If coarse retune API exists in quelware, adopt it explicitly in QuEL-3 path.
3. If coarse retune is unavailable, define fixed-coarse scan mode and its valid frequency window.
4. On out-of-range requests, fail fast with a deterministic error and suggested valid range.
5. Add contract tests for both positive and out-of-range paths.

## P0 implementation tasks for 3/19 demo gate

1. Add a backend capability profile for frequency scan/reset/introspection features.
2. Refactor `CharacterizationService` to use capability-gated strategy, not direct QuEL-1-only APIs.
3. Implement QuEL-3 frequency sweep strategy for qubit/resonator scans and electrical delay measurement.
4. Remove or gate unconditional reset dependencies in sweep-heavy flows (`rabi`, `rb`, `bell` paths).
5. DONE (`2026-02-27`): Completed `InstrumentResolver` migration and removed legacy `InstrumentMapper` dependency in QuEL-3 execution manager.
6. IN_PROGRESS: Removed single-alias execution restriction in software (`2026-02-27`); cross-unit synchronized-trigger hardware validation is still required.
7. IN_PROGRESS: Bound capture-mode directives for `avg`/`single` (`AVERAGED_VALUE` and `VALUES_PER_ITER` with legacy `VALUES_PER_LOOP` fallback); waveform-inspection (`AVERAGED_WAVEFORM`) contract validation is still pending.

## Questions to confirm with quelware team

1. Is there an official runtime API for coarse frequency retune (LO/CNCO equivalent) in QuEL-3?
2. If yes, what is the expected call sequence and safety constraints during rapid spectroscopy scans?
3. If no, what is the guaranteed fine-sweep window per transceiver/port around a fixed coarse setting?
4. Can Qubex query supported sweep ranges programmatically per resource?
5. Is there a `dump_box`-equivalent API for runtime frequency/introspection snapshots?
6. Are cross-unit synchronous triggers guaranteed in one `Session.trigger(instrument_ids=...)` call, and what jitter budget is specified?
7. For `CaptureMode.AVERAGED_WAVEFORM`, what exact result shape/key contract should Qubex assume?
