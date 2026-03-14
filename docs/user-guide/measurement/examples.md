# Measurement example workflows

This page highlights the main notebook entry points for the `Measurement` workflow.
Use these notebooks when you want a measurement-centric workflow built around sessions, schedules, capture/readout, sweeps, or backend-facing execution details.

## Recommended starting points

- [Measurement config](../../examples/measurement/measurement_config.ipynb): Build measurement configuration objects, including an offline example with a mocked system.
- [Measurement session](../../examples/measurement/measurement_client.ipynb): Create a measurement session, connect to hardware, and run basic measurements.
- [Loopback capture](../../examples/measurement/capture_loopback.ipynb): Inspect capture behavior without a full experiment workflow.

## Sweep and execution workflows

- [Measurement sweep builder](../../examples/measurement/sweep_measurement_builder.ipynb): Build `PulseSchedule` objects from sweep configuration models.
- [Sweep measurement executor](../../examples/measurement/sweep_measurement_executor.ipynb): Execute structured sweep workflows from measurement-side abstractions.
- [QuEL-3 sequencer builder flow](../../examples/measurement/quel3_sequencer_builder_flow.ipynb): Work with a backend-specific sequencer path.

## Related pages

- [Low-level APIs](../low-level-apis/index.md)
- [Measurement overview](index.md)
- [Full examples index](../../examples/index.md)
