# QuEL-3 Demo Readiness (2026-03-19)

## Goal

Prepare executable demo paths for:

1. qubit frequency identification
2. 1Q Rabi oscillation
3. 1Q gate calibration and benchmarking
4. 2Q gate calibration and benchmarking
5. Bell-state generation/measurement
6. multi-qubit experiment demo

## Current source-based status (as of 2026-03-15)

| Demo item | Status | Evidence in current code | Required work before demo |
| --- | --- | --- | --- |
| qubit frequency identification | BLOCKED | `CharacterizationService.scan_qubit_frequencies()` / `scan_resonator_frequencies()` / `measure_electrical_delay()` still enter `SystemManager.modified_backend_settings(...)`, but the QuEL-3 path now treats that request as a compatibility no-op. The effective sweep window is therefore still undefined when coarse retune would be required. | Define QuEL-3 sweep semantics and implement a QuEL-3-safe contract for out-of-range requests. |
| 1Q Rabi oscillation | PARTIAL | `MeasurementService.rabi_experiment()` still flows through `sweep_parameter()`, but QuEL-3 now treats `reset_awg_and_capunits()` as a compatibility no-op instead of raising. The software path no longer hard-fails on reset capability. | Validate 1Q Rabi behavior on hardware and confirm that no explicit reset is required between sweep points. |
| 1Q gate calibration + benchmarking | PARTIAL | Calibration/RB paths still request reset by default, but those requests no longer raise on QuEL-3. The remaining risk is behavioral validation, not missing software capability. | Validate representative 1Q calibration and RB flows on QuEL-3 hardware. |
| 2Q gate calibration + benchmarking | BLOCKED | Reset requests no longer hard-fail, but 2Q paths still depend on cross-unit synchronized execution and lack attached hardware evidence. | Attach cross-unit synchronized-trigger hardware evidence and validate representative 2Q flows. |
| Bell-state demo | BLOCKED | `MeasurementService.measure_bell_state()` no longer hard-fails on reset capability, but the demo still depends on validated 2Q execution/tomography behavior. | Validate Bell/tomography path on QuEL-3 after 2Q execution evidence is available. |
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
- The current QuEL-3 path now degrades reset and backend-settings override requests to compatibility no-op instead of raising.
- Therefore the current `CharacterizationService` implementation can proceed on QuEL-3, but its valid frequency window is still undefined whenever a true coarse retune would have been required.

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
4. DONE PARTIALLY (`2026-03-15`): reset-dependent sweep-heavy flows (`rabi`, `rb`, `bell` paths) no longer hard-fail on QuEL-3 because unsupported reset requests are treated as compatibility no-op; hardware behavior still needs validation.
5. DONE (`2026-02-27`): Completed `InstrumentResolver` migration and removed legacy `InstrumentMapper` dependency in QuEL-3 execution manager.
6. IN_PROGRESS: Removed single-alias execution restriction in software (`2026-02-27`); cross-unit synchronized-trigger hardware validation is still required.
7. IN_PROGRESS: Bound capture-mode directives for `avg`/`single` (`AVERAGED_VALUE` and `VALUES_PER_ITER`); waveform-inspection (`AVERAGED_WAVEFORM`) contract validation is still pending.

## Questions to confirm with quelware team

1. Is there an official runtime API for coarse frequency retune (LO/CNCO equivalent) in QuEL-3?
2. If yes, what is the expected call sequence and safety constraints during rapid spectroscopy scans?
3. If no, what is the guaranteed fine-sweep window per transceiver/port around a fixed coarse setting?
4. Can Qubex query supported sweep ranges programmatically per resource?
5. Is there a `dump_box`-equivalent API for runtime frequency/introspection snapshots?
6. Are cross-unit synchronous triggers guaranteed in one `Session.trigger(instrument_ids=...)` call, and what jitter budget is specified?
7. For `CaptureMode.AVERAGED_WAVEFORM`, what exact result shape/key contract should Qubex assume?
