# Low-level APIs

The low-level API section is for developers who need module-level control
beneath `Experiment`. It is organized around three modules:
`measurement`, `system`, and `backend`.

Most Qubex users should start with [`Experiment`](../experiment/index.md) on
real hardware or [`QuantumSimulator`](../simulator/index.md) for offline work.
Come here when sessions, system models, or backend controllers need to be the
primary abstraction.

## Module map

| Module | Responsibility | Start here when |
| --- | --- | --- |
| [`measurement`](../measurement/index.md) | Sessions, schedules, capture/readout, sweeps, and result conversion | You want to build or run measurement-centric execution flows |
| [`system`](../system/index.md) | Configuration loading, in-memory system models, and software/hardware synchronization | You want to inspect one system definition or coordinate runtime state |
| [`backend`](../backend/index.md) | Backend controller contracts and QuEL-specific implementations | You want to work with controller-level execution or backend-specific payloads |

## How the modules fit together

1. [`system`](../system/index.md) loads configuration files and assembles the
   software-side `ExperimentSystem`.
2. [`measurement`](../measurement/index.md) builds sessions, schedules,
   capture/readout, and sweep flows on top of that state.
3. [`backend`](../backend/index.md) controllers execute the prepared requests
   on concrete QuEL runtimes.

## Recommended paths

- [`measurement`](../measurement/index.md): start here for session lifecycle,
  `MeasurementSchedule`, capture/readout, and sweeps. Then continue with
  [`measurement` example workflows](../measurement/examples.md).
- [`system`](../system/index.md): start here for `ConfigLoader`,
  `ExperimentSystem`, `SystemManager`, and synchronization. Then continue with
  [`system` example workflows](../system/examples.md).
- [`backend`](../backend/index.md): start here for `BackendController`,
  backend kinds, and QuEL-specific implementations. Then continue with
  [`backend` example workflows](../backend/examples.md).

## Choose `Experiment` instead when

- You want the recommended user-facing workflow for most hardware-backed experiments
- You want built-in characterization, calibration, and benchmarking routines
- You prefer one facade for setup, execution, and analysis

## Related documentation

- [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
- [Examples](../../examples/index.md)
- [API reference](../../api-reference/qubex/index.md)
