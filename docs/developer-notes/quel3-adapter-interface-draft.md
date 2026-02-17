# QuEL-3 Adapter Interface Draft

## Purpose

Define a concrete integration draft for QuEL-3 support using `quelware-client` while keeping the `MeasurementClient` compatibility contract.

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

- Compatibility surface remains `MeasurementClient`.
- Backend-specific behavior diverges below `MeasurementClient`.
- One experiment session uses a single backend family (`quel1` or `quel3`), not mixed operation.
- `MeasurementScheduleExecutor.create_default()` now supports backend-provided hooks:
  - `create_measurement_backend_executor(execution_mode, clock_health_checks)`
  - `create_measurement_backend_adapter(experiment_system, constraint_profile)`
  - `create_measurement_result_factory(experiment_system)`

## Current implementation scaffold

- Added `Quel3MeasurementBackendAdapter` in `src/qubex/measurement/adapters/backend_adapter.py`.
- Added `Quel3BackendExecutor` in `src/qubex/measurement/adapters/backend_executor.py` (delegates to `execute_measurement(payload=...)` hook).
- Added `Quel3ExecutionPayload`/timeline dataclasses for schedule-to-payload conversion.
- Added `instrument_aliases` to `Quel3ExecutionPayload`; adapter resolves alias via `resolve_instrument_alias(target)` hook when available (fallback: target label itself).
- Added `Quel3BackendController` scaffold in `src/qubex/backend/quel3/quel3_backend_controller.py`.
- `Quel3BackendController.execute_measurement(...)` now includes an initial quelware invocation path and returns canonical `MeasurementResult` directly.
- If quelware dependencies are missing, execution fails fast with an explicit runtime error message.
- `SystemManager.load(..., backend_kind=...)` and `MeasurementClient.load(..., backend_kind=...)` now select backend family at session scope.
- Added default adapter-selection hint: `MEASUREMENT_BACKEND_KIND="quel3"` in `MeasurementScheduleExecutor`.
- For `MEASUREMENT_BACKEND_KIND="quel3"`, backend must provide either:
  - `create_measurement_backend_executor(...)`, or
  - `execute_measurement(payload=...)` (used by built-in `Quel3BackendExecutor`).
- `MeasurementScheduleExecutor.execute()` now accepts a backend executor returning canonical `MeasurementResult` directly (result-factory bypass path).
- Added adapter tests in `tests/measurement/test_quel3_measurement_backend_adapter.py`.
- Scope is intentionally minimal: relaxed validation + payload construction; direct quelware invocation remains in follow-up work.

## QuEL-3 mapping draft

| Qubex concept | quelware API candidate | Draft mapping |
| --- | --- | --- |
| schedule waveform samples | `Sequencer.register_waveform` | Register target waveform per logical event block. |
| schedule pulse placement | `Sequencer.add_event` | Use schedule start offset in ns and waveform name. |
| capture schedule | `Sequencer.add_capture_window` | Capture windows are added in ns, then exported in samples. |
| backend timeline export | `Sequencer.export_set_fixed_timeline_directive` | Convert ns timeline into sample timeline using instrument `sampling_period_fs`. |
| hardware execution | `InstrumentDriver.apply` + `Session.trigger` | Apply fixed-timeline directive then trigger session. |
| measured data fetch | `InstrumentDriver.fetch_result` | Read `FixedTimelineResult.iq_datas`. |

## Constraint model assumptions

- QuEL-3 path is treated as relaxed constraints:
  - no explicit WORD/BLOCK alignment in Qubex adapter
  - sample-grid placement based on backend dt
  - final packing/alignment delegated to quelware side
- Canonical dt source:
  - `instrument_config.sampling_period_fs / 1e6` (ns)

## Result conversion draft

- Convert `FixedTimelineResult.iq_datas` into Qubex measurement result payload.
- Keep `sampling_period_ns` in result metadata.
- Keep `avg_sample_stride` explicit. Default remains `4` for 4-way multiplexed readout demodulation semantics unless backend contract provides another value.
- Remove QuEL-1 specific extra-capture assumptions from QuEL-3 result path.

## Open questions before implementation

1. Final policy for target-to-alias mapping (current scaffold: `resolve_instrument_alias` hook, fallback to target label).
2. Whether one capture window name should encode target + capture index or use separate lookup table.
3. How to handle multi-unit trigger orchestration policy when multiple resources are involved.
4. Whether QuEL-3 returns already-demodulated/integrated payloads or raw sampled windows for each mode.
