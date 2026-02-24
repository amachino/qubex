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
- `packages/quelware-client/quelware-client/examples/use_instrument.py`

## Integration boundary in Qubex

- Compatibility surface remains `Measurement`.
- Backend-specific behavior diverges below `Measurement`.
- One experiment session uses a single backend family (`quel1` or `quel3`), not mixed operation.
- `MeasurementScheduleRunner.create_default()` now supports backend-provided hooks:
  - `create_measurement_backend_adapter(experiment_system, constraint_profile)`
  - `create_measurement_result_factory(experiment_system)`

## Current implementation scaffold

- Added `Quel3MeasurementBackendAdapter` in `src/qubex/measurement/adapters/backend_adapter.py`.
- Added `Quel3ExecutionPayload`/timeline dataclasses in `src/qubex/backend/quel3/quel3_execution_payload.py`.
- Adapter builds backend payload models; backend controller/execution manager consume them.
- Added `instrument_aliases` to `Quel3ExecutionPayload`; adapter resolves alias via `resolve_instrument_alias(target)` hook.
- Added `Quel3BackendController` scaffold in `src/qubex/backend/quel3/quel3_backend_controller.py`.
- `Quel3BackendController.execute(...)` includes a quelware invocation path and returns canonical `MeasurementResult` directly.
- If quelware dependencies are missing, execution fails fast with an explicit runtime error message.
- `SystemManager.load(..., backend_kind=...)` and `Measurement.load(..., backend_kind=...)` now select backend family at session scope.
- Added default adapter-selection hint: `MEASUREMENT_BACKEND_KIND="quel3"` in `MeasurementScheduleRunner`.
- For `MEASUREMENT_BACKEND_KIND="quel3"`, backend executes through `execute(request=...)`.
- `MeasurementScheduleRunner.execute()` calls `BackendController.execute(request=...)`, and QuEL-3 execution manager returns canonical `MeasurementResult` directly (result-factory bypass path).
- Added adapter tests in `tests/measurement/test_quel3_measurement_backend_adapter.py`.
- Scope is intentionally minimal: relaxed validation + payload construction; direct quelware invocation remains in follow-up work.

## QuEL-3 mapping draft

| Qubex concept | quelware API candidate | Draft mapping |
| --- | --- | --- |
| schedule waveform samples | `Sequencer.register_waveform` | Register target waveform per logical event block. |
| schedule pulse placement | `Sequencer.add_event` | Use schedule start offset in ns and waveform name. |
| capture schedule | `Sequencer.add_capture_window` | Capture windows are added in ns, then exported in samples. |
| backend timeline export | `Sequencer.export_set_fixed_timeline_directive` | Convert ns timeline into sample timeline using instrument `sampling_period_fs`. |
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
- Keep `avg_sample_stride` explicit. Default remains `4` for 4-way multiplexed readout demodulation semantics unless backend contract provides another value.
- Remove QuEL-1 specific extra-capture assumptions from QuEL-3 result path.

## v1.5.0 beta decisions (updated on 2026-02-24)

Dependency note:

- `quelware-client` completion is still required for full QuEL-3 execution-path validation.
- Despite the dependency status, DF-01 and DF-02 are fixed for beta scope.
- DF-03 and DF-04 remain deferred until upstream behavior is finalized.

| ID | Topic | Beta decision | Why now | Status |
| --- | --- | --- | --- | --- |
| DF-01 | Target-to-alias mapping | Require explicit instrument alias resolution for execution and prohibit fallback to target labels. | QuEL-3 execution is not valid without explicit instrument mapping. | DECIDED |
| DF-02 | Capture-window key policy | Standardize key as `{target}:{capture_index}`. `capture_index` is per-target 0-based and deterministic (`start_time` asc, `duration` asc, then definition order). | Removes coupling to `window_name` and keeps key generation stable across internal naming changes. | DECIDED |
| DF-03 | Trigger orchestration | Keep deferred until `quelware-client` orchestrator behavior is finalized. | Current dependency is incomplete; fixing policy now would be speculative. | DEFERRED |
| DF-04 | Result mode contract | Keep deferred until `quelware-client` result contracts are finalized. | Mode-level guarantees depend on unfinished upstream behavior. | DEFERRED |

## Implementation notes from DF-01 and DF-02

- `instrument_aliases` must be resolved explicitly for all execution targets.
- Missing alias resolution must raise a clear runtime/configuration error.
- Capture lookup keys in sequencer/export/result-fetch paths must use `{target}:{capture_index}`.
- `window_name` is treated as metadata/display only and is not part of the contract key.

## Follow-up questions (post-beta candidate)

1. Re-open DF-03 when `quelware-client` is ready and define synchronized orchestration policy.
2. Re-open DF-04 when `quelware-client` mode/result contracts are ready.
