# v1.5.0 Minimal Synchronized Protocol Scenario

## Purpose

Define the minimum synchronized measurement scenario required for v1.5.0 beta gate sign-off.

## Scenario ID

- `SP-BETA-001`

## Scope

- Backend family: QuEL-3 (`backend_kind="quel3"`)
- API contract surface: `MeasurementClient.execute_measurement_schedule(...)`
- Mode: `avg`
- Goal: verify one payload is applied and triggered as a synchronized flow across all involved aliases.

## Status

- `PROPOSED`
- Execution-path validation is deferred until `quelware-client` completion.

## Preconditions

- `system.yaml` / `chip.yaml` backend resolution is configured for QuEL-3.
- Target-to-alias resolution is configured (`resolve_instrument_alias` or direct label fallback).
- Capture schedule contains at least two channels/targets in one schedule.
- Test environment can run both:
  - real hardware validation path
  - `mock_mode=True` compatibility path

## Procedure

1. Build one `MeasurementSchedule` with at least two readout targets and capture windows.
2. Build one `MeasurementConfig` in `avg` mode (`shots >= 2`).
3. Execute once through `MeasurementClient.execute_measurement_schedule(schedule, config)`.
4. Confirm result uses canonical `MeasurementResult` shape by output target label.
5. Confirm no backend timing/alignment error is raised for the synchronized payload.
6. Record `dt`, backend kind, alias list, and pass/fail in hardware validation sheet.

## Pass criteria

- Execution succeeds without runtime/backend errors.
- Returned object is `MeasurementResult`.
- Result includes expected output targets and non-empty IQ arrays for each capture.
- `sampling_period_ns` is set from backend/controller contract.
- Same high-level flow remains callable with `mock_mode=True`.

## Failure handling

- Mark `HV-003` as fail in `hardware-validation-template.md`.
- Capture blocking logs and payload summary.
- File follow-up item in release plan as `P0` if compatibility or synchronized path is broken.

## Out of scope for v1.5.0 beta

- Multi-session orchestration across backend families.
- Advanced demodulation/classification contract expansion beyond current `MeasurementResult` compatibility.
