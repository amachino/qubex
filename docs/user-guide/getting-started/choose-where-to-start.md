# Choose where to start

Qubex documentation is organized around two recommended workflows, one shared
concept area, and one specialized API area.
For most users, the right place to start is either `Experiment` or `Simulator`.

## Start with one of these

| Start here | Use it when you want to | Next step |
| --- | --- | --- |
| `Experiment` | Run hardware-backed experiments through the recommended user-facing workflow, from setup through analysis | [Experiment overview](../experiment/index.md) |
| `Simulator` | Study pulse-level dynamics offline without connecting to hardware | [Simulator overview](../simulator/index.md) |

## Learn the shared pulse-sequence model

Choose `Pulse sequences` when you want to learn how Qubex builds pulse-level
sequences before deciding whether to execute them on hardware or use them in
offline studies.

Recommended path:

- [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
- [Pulse tutorial notebook](../../examples/pulse/tutorial.ipynb)

## Use Low-level APIs when needed

Choose `Low-level APIs` when sessions, schedules, capture/readout behavior,
sweeps, or backend execution details need to be your main abstraction.
This section is centered on `Measurement`, which underpins session lifecycle
and measurement execution in `Experiment`.

Recommended path:

- [Low-level APIs overview](../low-level-apis/index.md)
- [Measurement API overview](../measurement/index.md)
- [Measurement example workflows](../measurement/examples.md)

## Setup notes

- Install Qubex first: [Installation](installation.md)
- `Experiment` and hardware-backed `Low-level APIs` require configuration and parameter files: [System configuration](system-configuration.md)
- `Simulator` does not require hardware configuration files

## Contributing instead of using Qubex

If you want to extend Qubex or work on the codebase itself, start with [Contributing](../../CONTRIBUTING.md) and then continue with the [Developer guide](../../developer-guide/index.md).
