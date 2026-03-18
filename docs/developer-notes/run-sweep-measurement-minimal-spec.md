# run_sweep_measurement minimal spec

## Status

- State: `ACCEPTED`
- Last updated: `2026-02-26`
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
    schedule: Callable[[SweepValue], MeasurementSchedule],
    *,
    sweep_values: Sequence[SweepValue],
    config: MeasurementConfig | None = None,
) -> SweepMeasurementResult:
    ...
```

### Parameter semantics

- `schedule`: callback that builds one `MeasurementSchedule` for one sweep value.
- `sweep_values`: ordered sweep value list to execute.
- `config`: shared `MeasurementConfig` used for all points.
- if `config is None`, runtime resolves default configuration using standard measurement-config creation path.

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

### SweepValue

```python
SweepValue = Value | int | float | str
```

Notes:

- sweep values are not restricted to `float`.
- value interpretation is delegated to the `schedule` callback.

### SweepMeasurementResult

```python
class SweepMeasurementResult(DataModel):
    """Sweep measurement result (minimal)."""

    sweep_values: list[SweepValue]
    config: MeasurementConfig
    results: list[MeasurementResult]
```

Result ordering rules:

- `results[i]` corresponds to `sweep_values[i]`.

Derived aggregation contract:

- `result.data[target][capture_index]` prepends the sweep axis to the canonical
  per-point capture payload shape
- shape = `(len(sweep_values), *capture_shape)`
- canonical `capture_shape` is determined by `MeasurementConfig.primary_return_item`:
  - `WAVEFORM_SERIES`: `(n_shots, capture_length)`
  - `IQ_SERIES`: `(n_shots,)`
  - `STATE_SERIES`: `(n_shots,)`
  - `AVERAGED_WAVEFORM`: `(capture_length,)`
  - `AVERAGED_IQ`: `()`

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
