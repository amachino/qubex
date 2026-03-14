# `system` module

`qubex.system` owns configuration loading and the software-side representation
of one concrete instrument setup. It also coordinates synchronization between
the in-memory `ExperimentSystem` and backend controller state.

This page sits under [Low-level APIs](../low-level-apis/index.md).

## Use `system` when

- You want to load configuration files directly or inspect the resulting models
- `ExperimentSystem`, `QuantumSystem`, `ControlSystem`, targets, or control parameters should be first-class objects
- You need to compare or synchronize software state and hardware/controller state

## Key objects

- `ConfigLoader`: loads one selected system and builds an `ExperimentSystem`
- `ExperimentSystem`, `QuantumSystem`, and `ControlSystem`: software-side models for chip, wiring, ports, channels, and parameters
- `SystemManager`: singleton that owns the active experiment-system state, backend controller, and backend settings
- `Quel1SystemSynchronizer` and `Quel3SystemSynchronizer`: backend-specific synchronizers used by `SystemManager`

## Relationship to the other modules

- [`measurement`](../measurement/index.md): consumes the loaded system state to
  build sessions, schedules, and execution flows
- [`backend`](../backend/index.md): provides the controller state and runtime
  endpoints that `SystemManager` synchronizes against

## Recommended path

1. Prepare your configuration files: [System configuration](../getting-started/system-configuration.md)
2. Read the section overview: [Low-level APIs](../low-level-apis/index.md)
3. Start with curated notebooks: [`system` example workflows](examples.md)
4. Move to [`measurement`](../measurement/index.md) when sessions or schedules are the primary abstraction
5. Move to [`backend`](../backend/index.md) when controller implementations are the primary abstraction

## Choose `Experiment` instead when

- You want the recommended workflow that hides direct system-model handling
- You do not need to inspect configuration loading or synchronization details
- You prefer one facade for setup, execution, and analysis
