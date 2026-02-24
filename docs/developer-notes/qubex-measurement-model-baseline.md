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

- `frequency.channel_to_frequency_reference`
- `frequency.keep_oscillator_relative_phase`
- `data_acquisition.*` (kept for compatibility, not consumed by current builder)

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

- No qubex runtime path consumes this model today.
- Current presence is compatibility export only (`qubex.schema` / `qxschema` tests).
- Freeze decision: do not build internal logic on this type until qubex-side
  sweep-result requirements are defined.

## Decision update (2026-02-24)

The following sweep-API contract was accepted for the minimal v1 implementation:

- Sweep execution API uses internal models and is decoupled from `qxschema` model
  shape.
- `run_sweep_measurement(...)` reuses `MeasurementConfig` directly.
- A dedicated `SweepMeasurementConfig` is not introduced in v1.
- New minimal internal result model is introduced:
  - `SweepPointResult(index, point, result)`
  - `SweepMeasurementResult(results: list[SweepPointResult])`
- Execution policy is fixed to:
  - pointwise dispatch
  - fail-fast error handling

See details:

- [`run_sweep_measurement` minimal spec](run-sweep-measurement-minimal-spec.md)

Open items that remain for later phases:

1. Decide if `MeasurementResult.measurement_config` remains dict snapshot or is
   replaced by a typed internal snapshot model.
2. Define a strict qubex internal axis order contract for future tensor-style
   sweep result aggregation.
