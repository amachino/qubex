# run_sweep_measurement minimal spec

## Status

- State: `ACCEPTED`
- Last updated: `2026-02-24`
- Scope: internal `measurement` layer sweep API (minimal, no hardware batch sweep)

## Goal

Define the minimal internal contract for `Measurement.run_sweep_measurement` in v1.

This spec intentionally keeps the sweep API simple:

- pointwise execution only
- fail-fast only
- no sweep-specific execution config model

## Design policy

- `qxschema` sweep models are for external/common schema.
- qubex internal runtime models do not need to match `qxschema` one-to-one.
- Conversion between internal models and `qxschema` is handled at the boundary.

## API contract

`Measurement.run_sweep_measurement` and `MeasurementExecutionService.run_sweep_measurement`
use the same request/response contract.

```python
async def run_sweep_measurement(
    self,
    *,
    schedule: Callable[[SweepPoint], MeasurementSchedule],
    sweep_points: Sequence[SweepPoint],
    config: MeasurementConfig,
) -> SweepMeasurementResult:
    ...
```

### Parameter semantics

- `schedule`: callback that builds one `MeasurementSchedule` for one `SweepPoint`.
- `sweep_points`: ordered sweep point list to execute.
- `config`: shared `MeasurementConfig` used for all points.

## Model contract

### MeasurementConfig reuse

- `SweepMeasurementConfig` is not introduced in v1.
- Sweep execution reuses existing `MeasurementConfig`.

`MeasurementConfig` canonical fields:

- `mode: MeasurementMode`
- `shots: int`
- `interval: float`
- `enable_dsp_demodulation: bool`
- `enable_dsp_sum: bool`
- `enable_dsp_classification: bool`
- `line_param0: tuple[float, float, float]`
- `line_param1: tuple[float, float, float]`

### SweepPoint

```python
class SweepPoint(Model):
    """One sweep point."""

    parameters: dict[str, Value | int | float | str]
```

Notes:

- parameter values are not restricted to `float`.
- value interpretation is delegated to the `schedule` callback.

### SweepMeasurementResult

```python
class SweepPointResult(DataModel):
    """Result for one sweep point."""

    index: int
    point: SweepPoint
    result: MeasurementResult


class SweepMeasurementResult(DataModel):
    """Sweep measurement result (minimal)."""

    results: list[SweepPointResult]
```

Result ordering rules:

- `results[i]` corresponds to `sweep_points[i]`.
- `SweepPointResult.index` equals the original point index.

## Execution behavior

### Dispatch mode

- fixed to pointwise execution.
- each point is processed independently in input order.

### Failure policy

- fixed to fail-fast.
- if any point fails, execution stops immediately and raises the exception.
- partial `SweepMeasurementResult` is not returned on failure.

### Out of scope for v1

- hardware-native sweep execution
- batch dispatch/chunking options
- configurable failure policy
- sweep-specific timeout/save flags in API contract

## Compatibility and future extension

- future batch/hardware sweep support must be added in a backward-compatible way.
- candidate extension points (not part of v1 contract):
  - optional dispatch strategy field
  - optional failure policy field
  - optional timeout/chunk controls
