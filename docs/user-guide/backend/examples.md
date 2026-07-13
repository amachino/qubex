# `backend` example workflows

This page highlights notebooks for backend-specific execution paths and runtime
validation.

Backend-level tooling is backend-family specific. QuEL-1 CW output is provided
as a `Quel1BackendController` debugging workflow, not as a common
`BackendController` operation.

For direct QuEL-1 CW output API details, use
[QuEL-1 continuous-wave output](continuous-wave.md).

## Recommended starting points

- [PulseSchedule to QuEL-3 Sequencer Flow](../../examples/measurement/quel3_sequencer_builder_flow.ipynb): Follow how a `PulseSchedule` becomes a QuEL-3 sequencer plan.
- [QuEL-3 Deploy Check](../../examples/system/quel3_deploy_check.ipynb): Validate deployment and runtime connectivity for a QuEL-3 environment.
- [QuEL-1 continuous-wave output](continuous-wave.md): Start and stop CW output directly through `Quel1BackendController`.
- [QuEL-1 continuous-wave notebook](../../examples/backend/quel1_continuous_wave.ipynb): Run the guarded notebook workflow for QuEL-1 CW debugging.

## Related pages

- [Low-level APIs](../low-level-apis/index.md)
- [`backend` module](index.md)
- [`measurement` module](../measurement/index.md)
- [`system` module](../system/index.md)
- [Full examples index](../../examples/index.md)
