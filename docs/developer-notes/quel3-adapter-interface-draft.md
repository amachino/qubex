# QuEL-3 Adapter Interface Draft

## Purpose

Define a concrete integration draft for QuEL-3 support using `quelware-client` while keeping the `Measurement` compatibility contract.

## Naming convention

- Product name: `QuEL` (for example, `QuEL-3`)
- Class/type prefix in code: `Quel` (for example, `Quel3MeasurementBackendAdapter`)
- Middleware/client name: `quelware` (always lowercase, for example, `quelware-client`)

## Source snapshot (reference)

- `packages/quelware-client/quelware-client/src/quelware_client/client/helpers/sequencer/__init__.py`
- `packages/quelware-client/quelware-client/src/quelware_client/core/_client.py`
- `packages/quelware-client/quelware-client/src/quelware_client/core/_session.py`
- `packages/quelware-client/quelware-client/src/quelware_client/core/instrument_driver/__init__.py`
- `packages/quelware-client/quelware-client/examples/generate_readout_pulse.py`

## Integration boundary in Qubex

- Compatibility surface remains `Measurement`.
- Backend-specific behavior diverges below `Measurement`.
- One experiment session uses a single backend family (`quel1` or `quel3`), not mixed operation.
- `MeasurementBackendAdapter` handles both boundaries:
  - request conversion: schedule/config to backend execution request
  - result conversion: backend result payload to canonical `MeasurementResult`

## Current implementation scaffold

- Added `Quel3MeasurementBackendAdapter` in `src/qubex/measurement/adapters/quel3_backend_adapter.py`.
- Added `Quel3ExecutionPayload`/timeline dataclasses in `src/qubex/backend/quel3/quel3_execution_payload.py`.
- Adapter builds backend payload models; backend controller/execution manager consume them.
- `Quel3ExecutionPayload.fixed_timelines` is keyed by logical target, and runtime resolution merges/rewrites timelines by resolved instrument alias before sequencer export.
- `Quel3ExecutionPayload` now includes runtime binding metadata (`instrument_bindings`, `capture_port_bindings`) for resolver-stage mapping.
- Target-to-alias mapping policy for beta is runtime auto-resolution from wiring/port consistency checks, using current `quelware-client` `InstrumentResolver` flow.
- `Quel3ExecutionPayload` is limited to fields currently exercised by `quelware-client-internal` flow; QuEL-1 DSP/classifier options are intentionally excluded.
- Adapter no longer rejects multiple logical targets mapped to one alias; this is required for transceiver-style (`trx`) readout convergence.
- Added `Quel3BackendController` scaffold in `src/qubex/backend/quel3/quel3_backend_controller.py`.
- `Quel3BackendController.execute(...)` includes a quelware invocation path and returns backend-level `Quel3BackendExecutionResult`.
- If quelware dependencies are missing, execution fails fast with an explicit runtime error message.
- `SystemManager.load(..., backend_kind=...)` and `Measurement.load(..., backend_kind=...)` now select backend family at session scope.
- Added default adapter-selection hint: `MEASUREMENT_BACKEND_KIND="quel3"` in `MeasurementScheduleRunner`.
- For `MEASUREMENT_BACKEND_KIND="quel3"`, backend executes through `execute(request=...)`.
- `MeasurementScheduleRunner.execute()` calls `BackendController.execute(request=...)` and routes backend-result conversion through adapter-level `build_measurement_result(...)`.
- QuEL-3 adapter `build_measurement_result(...)` converts backend alias labels to Qubex target labels.
- Added adapter tests in `tests/measurement/test_quel3_measurement_backend_adapter.py`.
- Beta scope includes direct quelware invocation for synchronized multi-instrument execution and result conversion.

## QuEL-3 mapping draft

| Qubex concept | quelware API candidate | Draft mapping |
| --- | --- | --- |
| schedule waveform samples | `Sequencer.register_waveform` | Register target waveform per logical event block. |
| bind hardware sampling constraints | `Sequencer.bind` | Bind alias to instrument `sampling_period_fs` and `timeline_step_samples` before timeline export. |
| schedule pulse placement | `Sequencer.add_event` | Use schedule start offset in ns and waveform name. |
| capture schedule | `Sequencer.add_capture_window` | Capture windows are added in ns, then exported in samples. |
| backend timeline export | `Sequencer.export_set_fixed_timeline_directive` | Export timeline for one bound alias; final length is aligned to sample-grid constraints. |
| hardware execution | `InstrumentDriver.apply` + `Session.trigger(instrument_ids=...)` | Apply fixed-timeline directive, then trigger selected instruments via session API. |
| measured data fetch | `InstrumentDriver.fetch_result` | Read `ResultContainer.iq_result` (`WaveformList` or `IqPointList`). |

## Constraint model assumptions

- QuEL-3 path is treated as relaxed constraints:
  - no explicit WORD/BLOCK alignment in Qubex adapter
  - sample-grid placement based on backend dt
  - final packing/alignment delegated to quelware side
- Canonical dt source:
  - `instrument_config.sampling_period_fs / 1e6` (ns)

## Result conversion draft

- Convert `ResultContainer.iq_result` into Qubex measurement result payload.
- Keep `sampling_period_ns` in result metadata.
- Preserve alias-to-target capture mapping metadata through backend result (`capture_targets_by_alias`) to restore logical output labels in adapter conversion.
- Preserve quelware capture-mode semantics:
  - `single` uses `CaptureMode.VALUES_PER_ITER`.
  - `avg` uses `CaptureMode.AVERAGED_VALUE`.
  - waveform inspection flows (for example `check_waveform`) use `CaptureMode.AVERAGED_WAVEFORM`.
- Remove QuEL-1 specific extra-capture assumptions from QuEL-3 result path.

## v1.5.0 beta decisions (updated on 2026-02-25)

Dependency note:

- Integration baseline follows current `quelware-client` source and examples in this workspace.
- Resolver baseline is `InstrumentResolver` (legacy `InstrumentMapper` flow is no longer the contract).

| ID | Topic | Beta decision | Why now | Status |
| --- | --- | --- | --- | --- |
| DF-01 | Target-to-alias mapping | Resolve target-to-instrument alias automatically from wiring/port consistency at runtime; unresolved or ambiguous mapping fails fast. Fallback to label guessing is prohibited. | Matches operator workflow (port-first wiring) while keeping execution deterministic. | DECIDED |
| DF-02 | Capture-window key policy | Standardize key as `{instrument_alias}:{capture_index}`. `capture_index` is per-alias 0-based and deterministic (`start_time` asc, `duration` asc, then definition order). | Keeps backend payload/result vocabulary in instrument-alias terms and avoids controller-side label conversion. | DECIDED |
| DF-03 | Trigger orchestration | Require synchronized execution with multiple instrument aliases, including cross-unit trigger in one measurement run. | Beta gate explicitly requires multi-instrument and cross-unit synchronization on hardware. | DECIDED |
| DF-04 | Result mode contract | Use quelware capture modes as canonical: `single`=`VALUES_PER_ITER`, `avg`=`AVERAGED_VALUE`; waveform inspection uses `AVERAGED_WAVEFORM`. | Avoids ambiguous local averaging semantics and follows current upstream contracts. | DECIDED |
| DF-05 | `tx/rx/trx` resource mapping | Keep logical `read_out`/`read_in` in `ExperimentSystem`, but allow both to resolve to one transceiver (`trx`) resource/alias in QuEL-3 runtime when consistent. | Matches quelware resource model while keeping experiment-level vocabulary stable. | DECIDED |
| DF-06 | Backend settings introspection | Treat QuEL-1-style `dump_box` introspection as unsupported on QuEL-3 unless explicit quelware API is provided. | Current quelware surface does not confirm LO/CNCO/FNCO-style snapshot retrieval. | DECIDED |

## Implementation notes from DF-01 to DF-06

- Alias resolution must use runtime resolver flow (`InstrumentResolver`) with wiring/port consistency validation.
- Missing, ambiguous, or inconsistent alias resolution must raise clear runtime/configuration errors (fail-fast).
- Capture lookup keys in sequencer/export/result-fetch paths must use `{instrument_alias}:{capture_index}`.
- Session execution must open all resolved instrument resource IDs and trigger them synchronously, including cross-unit combinations.
- QuEL-3 capture mode must be configured via `SetCaptureMode` according to the mode contract (`VALUES_PER_ITER` / `AVERAGED_VALUE` / `AVERAGED_WAVEFORM`).
- `window_name` is treated as metadata/display only and is not part of the contract key.
- Adapter/payload path must allow one alias to carry both waveform events and capture windows when resolved role is transceiver.
- Any utility path that assumes QuEL-1 `dump_box` snapshots must be capability-gated on QuEL-3 and fail with explicit unsupported errors.
- `CharacterizationService` spectroscopy flows that currently retune LO/CNCO through QuEL-1-specific backend-settings APIs must be split to a QuEL-3-capability path before demo/GA gates.

## Follow-up questions (post-beta candidate)

1. Decide whether manual target-to-alias override configuration is needed in addition to runtime auto-resolution.
2. Define GA-level observability requirements (resolver trace logs, trigger timing diagnostics, and failure categorization).
3. Confirm whether quelware will provide a settings-introspection API equivalent to QuEL-1 `dump_box`.
