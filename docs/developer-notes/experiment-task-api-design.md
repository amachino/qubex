# Experiment task API design proposal

## Status

- State: `PROPOSED`
- Created: `2026-08-24`
- Updated: `2026-08-24`
- Scope: task-based experiment execution below the `Experiment` facade
- Discussion:
  - [Qubex issue #357](https://github.com/amachino/qubex/issues/357)
  - [QDash pull request #1363](https://github.com/oqtopus-team/qdash/pull/1363)

This document is a design proposal for team discussion. It does not introduce
a public API or change the compatibility contract. If the proposal is
accepted, change the state to `ACCEPTED` before merging it.

## Summary

Introduce a common task-based execution layer below the human-facing
`Experiment` facade.

An experiment Task contains the fully resolved parameters and
experiment-specific execution flow. It exposes the effective parameters before
execution and runs against an `ExperimentRuntime` that provides only the
capabilities needed by that Task. It does not read or update `CalibrationNote`
during execution.

`Experiment` remains a concise entry point for interactive use. A Service
constructs a Task from the current context, runs it, and owns optional policy
such as applying calibrated values to `CalibrationNote`, creating figures, and
adapting a task-specific result to the current `Result` contract. External
workflow systems can construct the same Task with explicit parameters and
consume its result without mutating Qubex session state.

## Motivation

Some experiment and calibration paths resolve inputs from `CalibrationNote`,
configuration, defaults, or other mutable context late in the execution path.
This is convenient for consecutive interactive work, but it creates two
problems.

First, external workflow systems cannot reliably reproduce an execution from
their recorded task inputs alone. The values used by Qubex may depend on the
current session state, and restoring that state requires the caller to know
Qubex internals.

Second, interactive users cannot easily inspect the complete realized
parameter set or identify where each value came from before starting an
experiment.

QDash currently works around this by restoring selected calibration inputs
into the Qubex context before dependent experiments. That compatibility layer
duplicates parts of Qubex parameter resolution and makes snapshot-based
re-execution harder to reason about.

## Goals

- Make experiment execution reproducible from explicit, inspectable inputs.
- Resolve all execution parameters before measurement starts.
- Keep one canonical owner for each default value.
- Allow interactive callers to use context-based convenience without making
  context access part of Task execution.
- Separate experiment execution from visualization, persistence, and
  `CalibrationNote` mutation.
- Give external workflow systems a supported API below `Experiment` without
  coupling them to the complete `ExperimentContext`.
- Support the temporary hardware-control operations required within experiment
  flows through narrowly scoped runtime capabilities.

## Non-goals

- Finalize class names, package paths, or every field in this proposal.
- Add every lower-level parameter as a flat argument to every `Experiment`
  method.
- Replace or break the current `Experiment` or `Result` public contracts.
- Define the serialization format for tasks, parameters, or results.
- Implement the task layer in this documentation change.
- Require one migration of all calibration, characterization, and benchmarking
  methods at once.

## Design principles

### Resolve before execution

A Task must contain all values that affect its measurement conditions,
sweeps, search initial values, and analysis behavior before `run()` starts.
Required values that remain unresolved must produce an error before hardware
execution.

The complete effective parameter set must be inspectable through
`task.parameters`. The resolution API should also make parameter provenance
inspectable so callers can distinguish explicit overrides, calibration state,
configuration, and canonical defaults. The exact provenance model remains an
open question.

### Keep the facade concise

`Experiment` is the human-facing facade and index of available experiment
methods. Its signatures should expose the main parameters that users commonly
need to understand or control. They should not flatten every control, readout,
pulse-shape, acquisition, analysis, and backend detail into hundreds of
arguments.

This does not imply that the current facade signatures are complete. Meaningful
experiment parameters should be added when they improve interactive use. Full
control belongs to the corresponding Task API.

### Keep execution free from calibration state

Task execution must not read from or write to `CalibrationNote`. It must not
receive the complete `ExperimentContext`, because doing so would make hidden
mutable-state access possible again.

Using calibrated values as inputs, producing new calibration estimates, and
committing accepted estimates to calibration state are separate operations.

### Keep side-effect policy outside the Task

A Task produces a task-specific result. It does not create or display figures,
save images, convert to the current generic `Result`, or decide whether to
update `CalibrationNote`.

The Service owns those policies for the existing interactive workflow. An
external workflow system may instead validate and persist the task-specific
result using its own policy.

### Restrict runtime capabilities

`ExperimentRuntime` provides the minimum capabilities needed during execution.
Measurement alone is insufficient because some experiments temporarily change
system settings while running. For example, qubit spectroscopy may change an
LO or NCO frequency while scanning different frequency ranges.

The runtime may therefore expose narrowly scoped measurement and temporary
system-control capabilities. It must not expose the complete
`ExperimentContext`, `CalibrationNote`, visualization, or persistence services.

## Proposed responsibility split

```text
Experiment
  - Human-facing entry point and method index
  - Keeps concise signatures focused on the main experiment parameters
  - Delegates processing to the owning Service

Service
  - Resolves context-dependent inputs and constructs a Task
  - Runs the Task with an ExperimentRuntime
  - Optionally applies accepted results to CalibrationNote
  - Creates and saves figures according to interactive policy
  - Adapts the task-specific result to the current Result contract

Task
  - Holds fully resolved experiment parameters
  - Implements the experiment-specific execution flow
  - Produces a task-specific result
  - Does not access or update CalibrationNote
  - Does not own visualization or persistence policy

ExperimentRuntime
  - Provides only the measurement and temporary system-control capabilities
    required to run a Task
```

There are two supported construction paths:

```text
Interactive use

Experiment -> Service -> Task.from_context(...) -> Task.run(runtime)
                       -> Service-side adaptation and optional state update

System-facing use

External workflow -> Task(explicit parameters) -> Task.run(runtime)
                  -> workflow-owned validation and persistence
```

Both paths must execute the same Task implementation after construction.

## Parameter lifecycle

### Context-based construction

For interactive use, a factory such as `Task.from_context(...)` resolves all
parameters immediately. The intended precedence is:

1. Explicit overrides passed to the factory.
2. Values owned by the current experiment or calibration context.
3. Canonical defaults owned by the appropriate Qubex layer.

Resolution must finish before the Task is returned. It must not pass unresolved
`None` values through the execution chain merely so a deeper layer can consult
mutable state later.

Keeping one owner for each default remains important. Immediate resolution does
not mean duplicating the same default across the facade, Service, and Task.

### Explicit construction

For system-facing use, direct Task construction accepts every required
execution parameter explicitly. It does not implicitly fall back to
`CalibrationNote` or other session state. Missing required parameters fail
validation before execution.

Grouped parameter models may be appropriate when they preserve domain
structure and make validation clearer. Whether individual Tasks should use
grouped models, explicit fields, or a combination is not decided here.

### Execution

After construction, the Task's effective parameters are stable and available
through `task.parameters`. `run(runtime)` uses only those parameters and the
narrow capabilities supplied by the runtime.

The implementation must define cleanup semantics for temporary system changes.
An execution failure must not leave an LO, NCO, or other temporary setting in
an unintended state.

### Result handling

The task-specific result contains experiment output and newly estimated values
needed by its consumers. The effective input parameters remain available from
the Task and may also be included in a serializable execution record in a
future design.

Applying estimates to `CalibrationNote` is an explicit Service or caller
operation after validation. A failed or rejected result must not update shared
calibration state merely because the Task ran.

## Illustrative API shape

The following examples explain the responsibility boundary. They do not freeze
names, signatures, inheritance, synchronization style, or package layout.

```python
class ExperimentTask[ParametersT, ResultT]:
    @property
    def parameters(self) -> ParametersT:
        ...

    def run(self, runtime: ExperimentRuntime) -> ResultT:
        ...
```

Interactive construction resolves context-dependent values before execution:

```python
task = QubitSpectroscopyTask.from_context(
    ctx,
    target="Q01",
    frequency_range=frequency_range,
    power_range=power_range,
    readout_amplitude=readout_amplitude,
    readout_frequency=readout_frequency,
    n_shots=n_shots,
    shot_interval=shot_interval,
)

# Inspect every realized value before execution.
task.parameters

task_result = task.run(runtime)
```

An external workflow supplies the complete input set directly:

```python
task = QubitSpectroscopyTask(
    target="Q01",
    frequency_range=frequency_range,
    power_range=power_range,
    readout_amplitude=0.2,
    readout_frequency=6.1,
    control_pulse_duration=200.0,
    readout_pulse_duration=1000.0,
    n_shots=2048,
    shot_interval=1024.0,
)

task.parameters
task_result = task.run(runtime)
```

The Service preserves the existing interactive contract around the Task:

```python
class CharacterizationService:
    def qubit_spectroscopy(...) -> Result:
        task = QubitSpectroscopyTask.from_context(self.ctx, ...)
        task_result = task.run(self._runtime)

        figure = self._make_qubit_spectroscopy_figure(task_result)
        self._show_or_save_figure(figure, ...)

        return self._to_result(task_result, figure)
```

## Compatibility and migration

The task API described here is new and unreleased. The removed legacy
`ExperimentTask`, `ExperimentTaskResult`, and `Experiment.run(task)` contracts
must not be used as its compatibility foundation.

The existing `Experiment` facade and generic `Result` contract remain the
compatibility surface during incremental migration. A Service can delegate to
a new Task internally without requiring existing interactive callers to adopt
the lower-level API.

Migration should proceed one coherent experiment path at a time. Qubit
spectroscopy is a useful candidate reference slice because it exercises both
parameter resolution and temporary LO or NCO control. The first implemented
slice should establish reusable runtime and validation patterns before broader
calibration, characterization, and benchmarking migration.

Each implementation pull request must separately decide:

- whether any affected public behavior has already been released;
- which existing defaults and result semantics must remain compatible;
- which parameter and result types are ready to become supported contracts;
- what unit, integration, and hardware validation is required.

## Alternatives considered

### Add every parameter to every Experiment method

This would make all values discoverable from the facade signature, but the
signature would mix main experiment controls with lower-level control,
readout, acquisition, analysis, and backend details. It would also tightly
couple the human-facing facade to implementation changes.

The proposal instead improves facade signatures selectively and provides the
Task API for complete control.

### Pass ExperimentContext to Task.run

This is convenient because all current capabilities and state are available,
but it preserves the possibility of late reads and writes to hidden mutable
state. It also makes Task dependencies difficult to inspect and test.

The proposal passes a restricted `ExperimentRuntime` instead.

### Keep resolving parameters at the deepest owning layer

This avoids duplicating defaults, but the complete effective input set remains
unknown until late in execution. The proposal keeps one owner per default while
moving resolution to Task construction.

### Let the Task update CalibrationNote and create figures

This preserves current all-in-one interactive behavior, but it prevents
external callers from validating results before committing them and mixes
execution with presentation and persistence policy.

The proposal keeps these responsibilities in the Service or external caller.

## Open questions

- Where should the common Task, parameter, result, and runtime contracts live?
- Should `ParametersT` use one structured model, explicit Task fields, or a
  combination of domain-specific submodels?
- How should parameter source provenance be represented and inspected?
- Which result data and execution metadata must be serializable for replay and
  audit?
- Should Task execution be synchronous, asynchronous, or support both through
  one canonical implementation?
- Which temporary system-control capabilities belong in
  `ExperimentRuntime`, and what cleanup contract should each capability have?
- What API should validate and atomically apply accepted task results to
  `CalibrationNote`?
- Which Task types and fields are stable public contracts for external systems,
  and which remain internal during the first migration?

## Acceptance criteria

This proposal is ready to move to `ACCEPTED` when the team agrees that:

- `Experiment` remains the concise human-facing facade;
- complete parameter resolution occurs before Task execution;
- direct system-facing Task construction has no implicit calibration-state
  fallback;
- Task execution cannot access or mutate `CalibrationNote`;
- visualization, persistence, result adaptation, and calibration-state updates
  remain outside the Task;
- `ExperimentRuntime` exposes restricted execution capabilities instead of the
  complete `ExperimentContext`;
- the same Task implementation serves interactive and external workflows; and
- unresolved API details can be decided incrementally without weakening these
  boundaries.

## Follow-up work after acceptance

1. Select one reference experiment and define its exact parameter and result
   models.
2. Define the minimum `ExperimentRuntime` capability protocols required by the
   reference experiment.
3. Add contract tests for construction, pre-execution validation, state
   isolation, and cleanup on failure.
4. Implement Service delegation while preserving the current facade and
   `Result` behavior.
5. Validate the explicit construction path with an external workflow consumer.
6. Use the reference slice to refine the common contracts before migrating
   additional experiments.
