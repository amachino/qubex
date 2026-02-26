# Sweep / NDSweep Measurement API Spec

## Status

- State: `DRAFT`
- Last updated: `2026-02-26`
- Scope: `measurement` layer async sweep APIs (`run_sweep_measurement`, `run_ndsweep_measurement`)

## Goal

Define a clear and lightweight API contract for 1D sweep and N-dimensional Cartesian-product sweep.

Design intent:

- keep pointwise callback (`schedule`) model
- keep fail-fast behavior
- keep result payload JSON-friendly
- avoid per-point duplicated config payload in sweep results

## Type aliases

```python
SweepKey = str
SweepValue = Value | int | float | str
SweepPoint = dict[SweepKey, SweepValue]
SweepAxes = tuple[SweepKey, ...]
```

## API contract

```python
async def run_sweep_measurement(
    self,
    schedule: Callable[[SweepPoint], MeasurementSchedule],
    *,
    sweep_points: Sequence[SweepPoint],
    config: MeasurementConfig | None = None,
) -> SweepMeasurementResult: ...


async def run_ndsweep_measurement(
    self,
    schedule: Callable[[SweepPoint], MeasurementSchedule],
    *,
    sweep_points: dict[SweepKey, Sequence[SweepValue]],
    sweep_axes: SweepAxes | None = None,
    config: MeasurementConfig | None = None,
) -> NDSweepMeasurementResult: ...
```

### Parameter semantics

- `schedule`: builds one `MeasurementSchedule` from one resolved sweep point.
- `config`: shared execution config for all points.
- if `config is None`, runtime resolves default via standard measurement-config creation path.

For ND sweep:

- `sweep_points`: axis-value table (`axis key -> ordered values`).
- `sweep_axes`: axis order for Cartesian product.
- if `sweep_axes is None`, axis order is resolved from `sweep_points` insertion order.
- `sweep_axes` must contain each key in `sweep_points` exactly once.

## Result models

```python
class SweepMeasurementResult(DataModel):
    sweep_points: list[SweepPoint]
    config: MeasurementConfig
    results: list[MeasurementResult]


class NDSweepMeasurementResult(DataModel):
    sweep_points: dict[SweepKey, list[SweepValue]]
    sweep_axes: SweepAxes
    shape: tuple[int, ...]
    config: MeasurementConfig
    results: list[MeasurementResult]
```

Notes:

- `shape` is derivable from `sweep_points` and `sweep_axes`, but stored explicitly for fast access and integrity checks.
- dedicated per-point wrapper models (`SweepPointResult`, `NDSweepPointResult`) are intentionally omitted in this draft.
- `measurement` layer uses strict `dict`/`tuple` contracts; future `Experiment` layer can provide a more flexible facade.

## Ordering and indexing rules

- Sweep result order:
  - `results[i]` corresponds to `sweep_points[i]`.
- ND sweep result order:
  - Cartesian product order with last axis varying fastest (C-order).
  - equivalent to `numpy.ndindex(shape)` iteration order.

## Invariants

- Sweep:
  - `len(results) == len(sweep_points)`
- ND sweep:
  - `shape == tuple(len(sweep_points[key]) for key in sweep_axes)`
  - `len(results) == prod(shape)`

## Failure policy

- fixed fail-fast:
  - first point error aborts execution and raises immediately
  - partial result object is not returned

## Config payload policy

- `MeasurementResult.measurement_config` is mandatory.
- `MeasurementResult.mode` is removed; callers should use
  `MeasurementResult.measurement_config.mode`.
- `SweepMeasurementResult.config` / `NDSweepMeasurementResult.config` is retained as
  shared sweep-level metadata.

## Helper methods

Recommended helpers for convenience and readability:

- `SweepMeasurementResult.get(index: int) -> MeasurementResult`
- `SweepMeasurementResult.get_sweep_point(index: int) -> SweepPoint`
- `NDSweepMeasurementResult.get(index: int | tuple[int, ...]) -> MeasurementResult`
- `NDSweepMeasurementResult.get_sweep_point(index: int | tuple[int, ...]) -> SweepPoint`

For ND sweep helpers:

- `index: int` is interpreted as flat index.
- `index: tuple[int, ...]` is interpreted as `ndindex`.
- `ndindex` access uses C-order flattening (`np.ravel_multi_index` equivalent).
