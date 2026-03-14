# `backend` module

`qubex.backend` defines the shared controller contract and the concrete
QuEL-1/QuEL-3 implementations that drive hardware-backed execution. It is the
lowest layer in the low-level stack and is mainly for integrators, runtime
validation, and backend-specific execution paths.

This page sits under [Low-level APIs](../low-level-apis/index.md).

## Use `backend` when

- You are implementing or validating a backend controller
- You need `BackendExecutionRequest`, backend result payloads, or backend kinds directly
- You are working on QuEL-specific deployment, sequencer, or execution paths

## Key objects

- `BackendController`, `BackendExecutionRequest`, and `BackendKind`: the shared controller contract
- `Quel1BackendController` and `Quel3BackendController`: concrete implementations for supported backend families
- Backend-specific models and builders such as `Quel1ExecutionPayload`, `Quel3ExecutionPayload`, and `Quel3SequencerBuilder`
- `qubex.measurement.adapters`: the bridge from measurement schedules/configs to backend requests

## Direct use is advanced

Most hardware-backed workflows should start from `Experiment` or
[`measurement`](../measurement/index.md). Use `backend` directly only when
controller-level behavior itself is the subject.

## Recommended path

1. Read the section overview: [Low-level APIs](../low-level-apis/index.md)
2. Read [`measurement`](../measurement/index.md) first if your work starts from schedules or results
3. Continue with [`backend` example workflows](examples.md)
4. Use the [API reference](../../api-reference/qubex/backend/index.md) for concrete controller details

## Choose another module instead when

- [`system`](../system/index.md): configuration loading, in-memory models, and synchronization are the main concern
- [`measurement`](../measurement/index.md): sessions, schedules, capture/readout, and sweeps are the main concern

## Choose `Experiment` instead when

- You want the recommended workflow for running hardware-backed experiments
- You do not need to inspect controller-level execution details
- You prefer one facade for setup, execution, and analysis
