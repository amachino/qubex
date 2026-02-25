# v1.5.0 Minimal Synchronized Protocol Scenario

## Purpose

Define the minimum synchronized measurement scenario required for v1.5.0 beta gate sign-off.

## Scenario ID

- `SP-BETA-001`

## Scope

- Backend family: QuEL-3 (`backend_kind="quel3"`)
- API contract surface: `Measurement.run_measurement_schedule(...)`
- Mode contract:
  - `avg` uses `CaptureMode.AVERAGED_VALUE`
  - `single` uses `CaptureMode.VALUES_PER_LOOP`
- Goal: verify one payload is applied and triggered as a synchronized flow across all involved aliases, including cross-unit combinations.

## Status

- `REQUIRED` (beta gate)
- Must pass on real hardware before v1.5.0 beta cut.

## Preconditions

- `system.yaml` / `chip.yaml` backend resolution is configured for QuEL-3.
- Target-to-alias mapping is resolved automatically from wiring/port consistency; fallback to target-label guessing is not allowed.
- Capture schedule contains at least two channels/targets mapped to different instrument aliases and different units.
- Test environment can run both:
  - real hardware validation path
  - `mock_mode=True` compatibility path

## Procedure

1. Build one `MeasurementSchedule` with at least two readout targets and capture windows, where targets resolve to different units.
2. Build one `MeasurementConfig` in `avg` mode (`shots >= 2`) and execute through `Measurement.run_measurement_schedule(schedule, config)`.
3. Build one `MeasurementConfig` in `single` mode and execute through the same API.
4. Confirm synchronized trigger includes all resolved instrument IDs in one session trigger path across units.
5. Confirm result uses canonical `MeasurementResult` shape by output target label for both modes.
6. Confirm no backend timing/alignment error is raised for the synchronized payload.
7. Confirm unresolved/ambiguous auto-resolution case fails fast with explicit runtime/configuration error.
8. Record `dt`, backend kind, unit list, alias list, capture mode, and pass/fail in hardware validation sheet.

## Pass criteria

- Execution succeeds without runtime/backend errors.
- Returned object is `MeasurementResult`.
- Result includes expected output targets and non-empty IQ arrays for each capture.
- Multi-instrument cross-unit synchronized trigger is confirmed in execution evidence.
- Mode semantics are consistent with quelware capture-mode contract:
  - `avg` returns `AVERAGED_VALUE` path semantics
  - `single` returns `VALUES_PER_LOOP` path semantics
- `sampling_period_ns` is set from backend/controller contract.
- Same high-level flow remains callable with `mock_mode=True`.

## Failure handling

- Mark `HV-003` as fail in `hardware-validation-template.md`.
- Capture blocking logs and payload summary.
- File follow-up item in release plan as `P0` if compatibility or synchronized path is broken.

## Out of scope for v1.5.0 beta

- Multi-session orchestration across backend families.
- Advanced demodulation/classification contract expansion beyond current `MeasurementResult` compatibility.
