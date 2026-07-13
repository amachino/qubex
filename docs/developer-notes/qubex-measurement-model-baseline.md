# qubex Measurement Model Baseline (Pre-qxschema)

## Status

- State: `PROPOSED`
- Last updated: `2026-02-23`
- Scope: freeze qubex-side internal model semantics before qxschema v1 externalization

## Goal

Stabilize the qubex-internal measurement model contract first, then design `qxschema`
external models as a conversion target.

This document is the source of truth for "what qubex currently executes and guarantees."

## Out of scope

- Final `qxschema` public field names and transport schema.
- Multi-rate (per-channel mixed) sampling-period normalization across LCM boundaries.
- Runtime model unification between qubex and qxschema.

## Internal model inventory

| Concept | Current type location | Runtime role |
| --- | --- | --- |
| `MeasurementConfig` | `src/qubex/measurement/models/measurement_config.py` | Canonical execution request config for schedule execution path. |
| `MeasurementResult` | `src/qubex/measurement/models/measurement_result.py` | Canonical serializable execution result. |
| `SweepMeasurementConfig` | `qxschema.SweepMeasurementConfig` (re-exported via `qubex.schema`) | Input for `SweepMeasurementBuilder`; only a subset is execution-effective today. |
| `SweepMeasurementResult` | `qxschema.SweepMeasurementResult` (re-exported via `qubex.schema`) | Not consumed by qubex runtime logic today. |

## 1) `MeasurementConfig` baseline

### Current internal shape

`MeasurementConfig` has required fields:

- `mode: MeasurementMode` (`"single"` or `"avg"`)
- `shots: int`
- `interval: float` (ns)
- `frequencies: dict[str, float]`
- `enable_dsp_demodulation: bool`
- `enable_dsp_sum: bool`
- `enable_dsp_classification: bool`
- `line_param0: tuple[float, float, float]`
- `line_param1: tuple[float, float, float]`

### Execution-effective semantics (freeze target)

- `mode`, `shots`, `interval`, `enable_dsp_*`, `line_param0`, and `line_param1`
  are execution-effective in backend adapters.
- `frequencies` is currently not consumed in the direct
  `run_measurement_schedule(...)` path; treat as reserved/advisory until wired.

### Validation/invariants

- All fields are required in direct model construction.
- Factory defaults are provided by `MeasurementConfigFactory`.

## 2) `MeasurementResult` baseline

### Current internal shape

- `mode: MeasurementMode`
- `data: dict[str, list[np.ndarray]]`
- `device_config: dict[str, Any] = {}`
- `measurement_config: dict[str, Any] = {}`
- `sampling_period_ns: float | None = None`
- `avg_sample_stride: int | None = None`

### Runtime semantics (freeze target)

- `data` key = output target label (usually qubit-like labels such as `Q00`).
- `data` value = capture list per target; index is capture order.
- `sampling_period_ns` and `avg_sample_stride` are optional metadata used by
  legacy conversion/time-axis logic.
- `measurement_config` is a snapshot payload (currently untyped dict from
  `MeasurementConfig.to_dict()`).
- `device_config` is backend/device snapshot payload.

### Canonical payload shape contract

`CaptureData` is the owner of payload-shape normalization. Backend adapters may
receive extra singleton axes from hardware/runtime libraries, but they must be
normalized before a payload is exposed through the canonical model.

Canonical shapes by `MeasurementConfig.primary_return_item`:

- `WAVEFORM_SERIES`: `(n_shots, capture_length)`
- `IQ_SERIES`: `(n_shots,)`
- `STATE_SERIES`: `(n_shots,)`
- `AVERAGED_WAVEFORM`: `(capture_length,)`
- `AVERAGED_IQ`: `()`

Implications:

- singleton trailing axes such as `(n_shots, 1)` for integrated IQ are not part
  of the canonical contract and are removed at the `CaptureData` boundary
- sweep aggregation prepends sweep axes to the canonical per-capture shape
  rather than preserving backend-specific singleton axes

## 3) `SweepMeasurementConfig` baseline (as consumed by qubex)

### Important distinction

qubex currently reuses `qxschema.SweepMeasurementConfig` type, but runtime uses
only part of it in `SweepMeasurementBuilder`.

### Execution-effective fields in current builder

- `channel_list`
- `sequence.delta_time`
- `sequence.command_list`
- `frequency.channel_to_frequency`
- `frequency.channel_to_frequency_shift`
- `sweep_parameter.sweep_content_list`
- `sweep_parameter.sweep_axis`

### Currently accepted but not execution-effective in builder

- `data_acquisition.*` (kept for compatibility, not consumed by current builder)

In the current executor/runtime split, this should be read more precisely as:

- `data_acquisition.channel_to_averaging_time`
  - only the key set is used for fallback capture-channel consistency validation
  - the values are unused because runtime has no per-channel averaging-time
    programming path
- `data_acquisition.channel_to_averaging_window`
  - only the key set and key order are used when no active readout-target
    pulses are present in the schedule
  - the window values are unused because runtime has no per-channel
    averaging-window programming path

### Currently unsupported in executor/runtime

- `frequency.channel_to_frequency_reference`
- `frequency.keep_oscillator_relative_phase=False`

Current runtime policy is:

- require `channel_to_frequency_reference == {}`
- require `keep_oscillator_relative_phase is True`
- reject unsupported values during executor validation

Reasons:

- `frequency.channel_to_frequency_reference`
  - current runtime has no model for shared oscillator/reference groups, so a
    non-empty mapping would imply semantics it cannot preserve
- `frequency.keep_oscillator_relative_phase=False`
  - current runtime executes each sweep point as an independent schedule and has
    no phase-reset or shared-oscillator-state control corresponding to `False`

### Currently unused in executor/runtime

- `sequence.variable_list`
  - reason: variable resolution is driven from
    `sweep_parameter.sweep_content_list[*].sweep_target`
- `data_acquisition.data_acquisition_timeout`
  - reason: current `MeasurementConfig` and execution path expose no timeout
    field
- `data_acquisition.delta_time`
  - reason: runtime sampling period is taken from `sequence.delta_time`, not
    from acquisition config

### Builder validation semantics (freeze target)

- `sweep_axis` cannot contain empty axis entries.
- Every sweep key in `sweep_axis` must exist in `sweep_content_list`.
- Sweep keys grouped in one axis must have equal `value_list` length.
- Supported sweep categories:
  - `sequence_variable`
  - `frequency_shift`
- `Blank` command requires exactly one duration argument.

### Sampling-period scope note

- Mixed per-channel sampling period and LCM alignment issues are out of scope for
  this baseline phase.

## 4) `SweepMeasurementResult` baseline

- `measurement` layer runtime now consumes this model in
  `run_sweep_measurement(...)`.
- Internal result shape is intentionally decoupled from external `qxschema`
  result shape.

## Decision update (2026-02-26)

The following sweep-API contract was accepted for the minimal v1 implementation:

- Sweep execution API uses internal models and is decoupled from `qxschema` model
  shape.
- `run_sweep_measurement(...)` reuses `MeasurementConfig` directly.
- A dedicated `SweepMeasurementConfig` is not introduced in v1.
- New minimal internal result model is introduced:
  - `SweepMeasurementResult(sweep_values, config, results)`
- 1D sweep API contract is:
  - `schedule: Callable[[SweepValue], MeasurementSchedule]`
  - `sweep_values: Sequence[SweepValue]`
  - `config: MeasurementConfig | None = None`
- Execution policy is fixed to:
  - pointwise dispatch
  - fail-fast error handling

See details:

- [`run_sweep_measurement` minimal spec](run-sweep-measurement-minimal-spec.md)

Open items that remain for later phases:

1. Decide if `MeasurementResult.measurement_config` remains dict snapshot or is
   replaced by a typed internal snapshot model.
2. Keep the canonical payload-shape contract aligned with future tensor-style
   sweep/export work if new return-item kinds are introduced.
