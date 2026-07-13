# SweepMeasurementExecutor Design

## Status

- State: `PROPOSED`
- Last updated: `2026-03-12`
- Scope: dedicated executor for externally provided `qxschema.SweepMeasurementConfig`

## Goal

Define a dedicated class that:

- accepts `qxschema.SweepMeasurementConfig`
- executes the sweep using `qubex`
- returns `qxschema.SweepMeasurementResult`

This design intentionally keeps the external protocol boundary outside
`Measurement`.

## Public API

```python
class SweepMeasurementExecutor:
    def __init__(
        self,
        *,
        measurement: Measurement,
        command_registry: Mapping[str, Any] | None = None,
    ) -> None: ...

    async def run(
        self,
        config: SweepMeasurementConfig,
    ) -> SweepMeasurementResult: ...
```

## Why a dedicated class

`Measurement` is a broad workflow facade. It already owns:

- session lifecycle
- schedule construction
- backend execution
- classification state
- high-level convenience APIs

`SweepMeasurementExecutor` should own only the external protocol boundary for
`qxschema` sweep configs:

- protocol validation
- config-to-runtime conversion
- sweep-point orchestration
- runtime-to-protocol result conversion

This keeps `Measurement` focused on runtime workflows and keeps protocol logic
isolated and testable.

## Responsibilities

`SweepMeasurementExecutor` is responsible for four things.

### 1. Protocol validation

Validate the incoming `SweepMeasurementConfig` beyond model parsing:

- supported command names
- supported sweep categories
- sweep-axis consistency
- active `Measurement` session readiness
- compatibility between external channel names and loaded `qubex` targets

### 2. Shared execution-config construction

Convert `config.data_acquisition` into internal `MeasurementConfig`.

### 3. Sweep execution orchestration

Reuse `SweepMeasurementBuilder` to:

- iterate sweep points
- resolve sweep state
- build `MeasurementSchedule` for each point

and call existing `measurement.run_measurement(...)` pointwise.

### 4. Result conversion

Convert the executed sweep into `qxschema.SweepMeasurementResult`.

## High-level execution flow

`run(config)` should perform the following steps.

1. Validate protocol-level and runtime-level prerequisites.
2. Create a `SweepMeasurementBuilder(config=config, command_registry=...)`.
3. Derive one shared internal `MeasurementConfig` from `config.data_acquisition`.
4. Iterate sweep points in C-order using the builder shape.
5. For each point:
   - build `MeasurementSchedule`
   - call existing `measurement.run_measurement(...)`
6. Aggregate per-point `MeasurementResult` objects.
7. Convert the aggregate into `qxschema.SweepMeasurementResult`.

This design deliberately reuses existing runtime execution instead of adding a
second backend path.

## Internal collaborators

The executor should remain small by delegating to focused helpers.

Recommended helper split:

- `_SweepMeasurementConfigConverter`
  - `DataAcquisitionConfig -> MeasurementConfig`
- `_SweepMeasurementResultConverter`
  - `list[MeasurementResult] + sweep metadata -> qxschema.SweepMeasurementResult`

`SweepMeasurementBuilder` itself should own:

- `resolve_sweep_state(...)`
- `iterate(...)`
- `build_measurement_schedule(indices) -> MeasurementSchedule`

If needed, pulse-only construction can remain as an internal helper inside
`SweepMeasurementBuilder`, for example `_build_pulse_schedule(...)`.

## Session prerequisites

`SweepMeasurementExecutor` does not own session lifecycle.

The provided `Measurement` instance must already be in an executable state:

- `measurement.session_service.experiment_system` is loaded
- `measurement.session_service.backend_controller` is available
- hardware is connected when using a real backend

If prerequisites are missing, `run(...)` should fail before the first sweep
point is built.

## Command model assumptions

The executor assumes the current `qxschema` command model:

```python
class ParametricSequencePulseCommand(Model):
    name: str
    channel_list: list[str]
    argument_list: list[str | float]
```

and the current sweep model:

```python
class ParameterSweepContent(Model):
    category: Literal["frequency_shift", "sequence_variable"]
    sweep_target: list[str]
    value_list: ValueArrayLike
```

So sequence variables are resolved by name and substituted into
`argument_list`.

## SweepMeasurementBuilder contract in v1

`SweepMeasurementBuilder` should be the schedule-building owner for one sweep
point.

Public responsibilities:

- `resolve_sweep_state(...)`
- `iterate(...)`
- `build_measurement_schedule(indices) -> MeasurementSchedule`

Internal responsibilities:

- build `PulseSchedule`
- derive `CaptureSchedule`
- apply schedule-effective parts of `config.data_acquisition`

Out of scope for the builder:

- `MeasurementConfig` construction
- shot count / averaging-mode interpretation for execution
- backend execution
- result conversion

This keeps the split clean:

- Builder: what to play and what to capture for one point
- Executor: how to run the sweep and how to export the result

## Supported command set in v1

The v1 `SweepMeasurementBuilder` should support the same built-in command set as
the current pulse-building implementation:

- `Barrier`
- `Blank`
- `Delay`
- `Wait`
- `Rect`
- `Gaussian`
- `FlatTop`
- `Drag`
- `RaisedCosine`
- `VirtualZ`

Rules:

- `Barrier` and blank aliases are reserved commands handled by the builder.
- Reserved commands are not looked up in `command_registry`.
- Unknown command names cause immediate validation failure.

## Command Registry Specification

`command_registry` customizes how `SweepMeasurementBuilder` converts
non-reserved commands into pulse objects. The executor accepts the registry and
passes it into the builder.

### Registry shape

Public constructor type remains:

```python
Mapping[str, Any] | None
```

but supported values are callables implementing the following protocol.

```python
class SweepCommandFactory(Protocol):
    def __call__(self, context: SweepCommandContext) -> SweepCommandValue: ...
```

Supporting types:

```python
@dataclass(frozen=True)
class SweepCommandContext:
    command: ParametricSequencePulseCommand
    resolved_argument_list: tuple[float, ...]
    sequence_variables: Mapping[str, float]
    frequency_shifts: Mapping[str, float]
    delta_time: float


SweepCommandValue = PulseSchedule | Waveform | PhaseShift
```

### Factory behavior

Factories must be pure with respect to executor state:

- same context -> same result
- no direct backend access
- no mutation of the shared `Measurement`

Returned values are interpreted as:

- `PulseSchedule`: inserted with `schedule.call(...)`
- `Waveform`: added to each channel in `command.channel_list`
- `PhaseShift`: added to each channel in `command.channel_list`

Any other return type is an execution error.

### Lookup rules

Registry lookup should be normalized as follows:

1. exact command name
2. lower-case command name

Merging policy:

- built-in factories are registered first
- custom `command_registry` entries override built-ins by normalized key
- duplicate normalized keys inside the custom registry are invalid

### Reserved commands

The following names are reserved and cannot be overridden by `command_registry`:

- `Barrier`
- `Blank`
- `Delay`
- `Wait`

If a registry tries to override a reserved command, builder or executor
construction should raise `ValueError`.

### Built-in factory adapters

Built-in `qxpulse` constructors should be wrapped internally into
`SweepCommandFactory` implementations.

Examples:

- `Rect`
- `Gaussian`
- `FlatTop`
- `Drag`
- `RaisedCosine`
- `VirtualZ`

This keeps the external contract uniform even though the underlying constructors
use positional or named parameters.

### Example

```python
def my_square(context: SweepCommandContext) -> Waveform:
    duration, amplitude = context.resolved_argument_list
    return Rect(duration=duration, amplitude=amplitude)


executor = SweepMeasurementExecutor(
    measurement=measurement,
    command_registry={"MySquare": my_square},
)
```

## Data-acquisition responsibility split

`config.data_acquisition` contains both schedule-effective and execution-config
fields.

### Builder-owned fields

These affect one point's `MeasurementSchedule` and therefore belong to
`SweepMeasurementBuilder`:

- `data_acquisition_duration`
- `data_acquisition_delay`
- capture channel selection derived from acquisition maps

### Executor-owned fields

These affect runtime execution policy and therefore belong to
`SweepMeasurementExecutor`:

- `shot_count`
- `flag_average_shots`
- `flag_average_waveform`
- future execution-policy fields

### Shared-but-non-execution-effective fields in v1

These should be validated and preserved in metadata, but not interpreted as
backend programming directives in v1:

- `data_acquisition_timeout`
- `delta_time`
- `channel_to_averaging_time`
- `channel_to_averaging_window`

## Data-acquisition conversion policy

`run(config)` has no runtime override parameters, so all execution settings must
come from `config.data_acquisition`.

The v1 conversion policy should be:

- `shot_count -> MeasurementConfig.n_shots`
- `shot_repetition_margin -> MeasurementConfig.shot_interval`
- `flag_average_shots -> MeasurementConfig.shot_averaging`
- `flag_average_waveform -> MeasurementConfig.time_integration`
- `state_classification -> False`
- `return_items -> inferred from legacy flags`

This keeps the derived `MeasurementConfig` fully deterministic.

## Capture policy

`qxschema.SweepMeasurementConfig` does not define explicit capture commands, so
the builder must define one runtime policy.

Recommended v1 policy:

- if active readout-target pulses are present (for example on `RQ00`), capture
  channels are those readout targets
- for each readout target, capture windows are derived from pulse start times on
  the same target
- if no active readout-target pulses are present, fall back to the keys of
  `data_acquisition.channel_to_averaging_window`
- each pulse range produces one capture
- capture start time:
  - `pulse_start_time + data_acquisition_delay`
- capture duration:
  - `data_acquisition_duration`

Additional rules:

- if a capture channel has no pulse range, validation fails
- if `data_acquisition_duration <= 0`, validation fails
- if the computed capture window exceeds schedule duration, validation fails

This policy makes `data_acquisition_delay` and `data_acquisition_duration`
schedule-effective without relying on `qubex`-specific readout conventions.

## Handling `channel_to_averaging_time` and `channel_to_averaging_window`

These fields do not have a direct execution landing zone in current runtime
models.

Recommended v1 behavior:

- require that every capture channel has entries in both maps
- preserve both maps in result metadata
- do not attempt backend-specific DSP/filter programming from these fields

Current interpretation detail:

- `channel_to_averaging_time`
  - only the key set is used for fallback capture-channel consistency validation
  - the time values themselves are unused because current runtime has no
    backend/DSP API that accepts per-channel averaging-time programming from
    this config
- `channel_to_averaging_window`
  - the key set and key order are used only when no active readout-target pulses
    are present
  - the window values themselves are unused because current runtime has no
    backend/DSP API that accepts per-channel averaging-window programming from
    this config

This must be documented as accepted-but-non-execution-effective in v1.

Silent dropping without documentation is not acceptable.

## Frequency semantics

The builder should apply:

- `frequency.channel_to_frequency`
- `frequency.channel_to_frequency_shift`
- sweep-time `frequency_shift`

by reusing `SweepMeasurementBuilder`.

The following field is unsupported in v1:

- `frequency.channel_to_frequency_reference`

Recommended v1 policy:

- require `frequency.channel_to_frequency_reference == {}`
- reject non-empty mappings with a fail-fast validation error

`frequency.keep_oscillator_relative_phase` is also not execution-effective in
v1.

Recommended v1 policy:

- require `frequency.keep_oscillator_relative_phase is True`
- reject `False` with a fail-fast validation error

The accepted `True` value may be preserved in result metadata for diagnostic
visibility.

## Currently unused fields and reasons

The following fields are currently unused by runtime execution and validation.

- `sequence.variable_list`
  - reason: sequence-variable resolution is driven from
    `sweep_parameter.sweep_content_list[*].sweep_target`, and command arguments
    are resolved directly against that expanded variable map
  - implication: the field is currently declarative only and does not constrain
    execution
- `data_acquisition.data_acquisition_timeout`
  - reason: current `MeasurementConfig` and `measurement.run_measurement(...)`
    path do not expose a timeout control to forward this value into runtime
  - implication: the field has no execution landing zone in v1
- `data_acquisition.delta_time`
  - reason: current schedule sampling period is taken from
    `sequence.delta_time`; the acquisition path does not implement an
    independent sampling grid
  - implication: differing acquisition `delta_time` values would have no
    defined runtime behavior in v1

## Channel-name contract

`config.channel_list` and all command `channel_list` entries must be valid
runtime labels for the active `Measurement` session.

Recommended rule:

- each external channel name must match a loaded experiment target label exactly

If channel aliasing is needed in the future, it should be added as a separate
translation layer rather than implicit behavior inside the executor.

## Sweep result conversion

The executor returns `qxschema.SweepMeasurementResult`, not the internal
`qubex` sweep result models.

### Result ordering

Execution order and result order must follow:

- `sweep_axis` order from `config`
- C-order flattening
- last axis varies fastest

### `sweep_key_list`

`qxschema.SweepMeasurementResult` only provides a flat `sweep_key_list`, while
`SweepMeasurementConfig` supports grouped keys per axis.

Recommended v1 policy:

- `sweep_key_list` is the flattened key list in axis order
- original grouped `sweep_axis` is preserved in `metadata["sweep_axis"]`

### `data_key_list`

`data_key_list` identifies each exported data plane.

Recommended order:

1. target order in first appearance order
2. capture index order within each target

Recommended key format:

- single capture: `"{target}"`
- multiple captures: `"{target}[{capture_index}]"`

### Exported payload

Each `MeasurementResult` may contain structured `CaptureData`, but
`qxschema.SweepMeasurementResult` has only one `data` ndarray.

Recommended v1 policy:

- export only `CaptureData.data`, i.e. the primary payload determined by the
  derived internal `MeasurementConfig`
- do not export auxiliary payload fields separately in v1

### Tensor layout

Recommended tensor layout:

```text
data.shape == (*sweep_shape, n_data_keys, *payload_shape)
```

where:

- `sweep_shape` comes from `config.sweep_parameter.sweep_axis`
- `n_data_keys == len(data_key_list)`
- `payload_shape` is the per-capture data shape

### Payload-shape invariant

To build one dense ndarray, all exported data keys must have the same
`payload_shape`.

If payload shapes differ across targets or capture indices, conversion should
raise an explicit error.

This is stricter than the internal runtime model, but it matches the current
`qxschema.SweepMeasurementResult` shape.

### Metadata contents

Recommended metadata keys:

- `measurement_config`
- `sampling_period`
- `sweep_axis`
- `channel_to_averaging_time`
- `channel_to_averaging_window`
- `keep_oscillator_relative_phase`
- `data_axis`:
  - `"after_sweep_axes"`

This metadata should make protocol loss visible until the result schema is
strengthened.

## Error policy

`run(config)` should be fail-fast.

Failure categories:

- protocol validation error
- unsupported semantics error
- command-factory error
- schedule-construction error
- runtime execution error
- result-conversion error

No partial `SweepMeasurementResult` should be returned on failure.

## Non-goals for v1

The first version of `SweepMeasurementExecutor` should not attempt to solve:

- backend-native batch sweep execution
- hardware-specific DSP/filter programming from averaging windows
- channel alias translation
- protocol-version negotiation
- multiple exported payload tensors in one result
- lossy auto-fallback when payload shapes differ

## Implementation order

1. Extend `SweepMeasurementBuilder` with `build_measurement_schedule(...)`
2. Keep pulse-only construction as an internal helper if needed
3. Add `src/qubex/measurement/sweep_measurement_executor.py`
4. Add protocol and registry helper types
5. Add acquisition-config conversion
6. Add qxschema result conversion
7. Add executor unit tests
8. Add end-to-end tests against the existing measurement runtime

## Test plan

Minimum required tests:

- successful 1D sweep execution
- successful ND sweep execution
- reserved command override rejection
- unknown command rejection
- capture-channel-without-pulse rejection
- invalid channel-name rejection
- deterministic `data_key_list` ordering
- qxschema result tensor shape correctness
- failure when payload shapes differ

## Bottom line

`SweepMeasurementExecutor` should be the dedicated protocol adapter between:

- external `qxschema.SweepMeasurementConfig`
- internal `qubex` measurement runtime
- external `qxschema.SweepMeasurementResult`

The critical design points are:

1. keep it outside `Measurement`
2. make `command_registry` a real contract, not an untyped escape hatch
3. define a concrete capture policy
4. define a concrete result-tensor layout
5. reject unsupported semantics explicitly
