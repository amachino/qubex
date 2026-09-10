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
- QuEL-1 optional controller capabilities such as
  [`start_continuous_wave()`](continuous-wave.md) for hardware-level CW output

## Direct use is advanced

Most hardware-backed workflows should start from `Experiment` or
[`measurement`](../measurement/index.md). Use `backend` directly only when
controller-level behavior itself is the subject.

## QuEL-3 execution sessions

Execution logs record the session ID captured on opening and the request attempt
number at `INFO`. Retries and cleanup failures are logged at `WARNING`; a final
request failure is logged at `ERROR` with its traceback and session ID.

Add your own possible causes to `QUELWARE_EXCEPTION_HINTS` in
`qubex.backend.quel3.managers.session_workarounds`. Keys are module-qualified
exception class names (for example, `quelware_client.core.exceptions.LockConflictError`);
values are the messages to display. The mapping is initially empty. Registered
hints appear as `possible cause` in failure logs, including for subclasses and
explicitly chained causes. They do not change retry decisions or exceptions.

Session creation retries known resource or unit availability failures up to four
times with backoff. A separate loop retries each payload up to four times after
an `Exception`, recreating the client and session. Each outer attempt retains its
own session creation budget; cancellation is not retried. A failure after trigger
can therefore execute the same payload again. Healthy clients are reused within
a batch. Final cleanup failures are logged without replacing the result or error.

## Recommended path

1. Read the section overview: [Low-level APIs](../low-level-apis/index.md)
2. Read [`measurement`](../measurement/index.md) first if your work starts from schedules or results
3. For QuEL-1 CW checks, read [QuEL-1 continuous-wave output](continuous-wave.md)
4. Continue with [`backend` example workflows](examples.md)
5. Use the [API reference](../../api-reference/qubex/backend/index.md) for concrete controller details

## Choose another module instead when

- [`system`](../system/index.md): configuration loading, in-memory models, and synchronization are the main concern
- [`measurement`](../measurement/index.md): `MeasurementSchedule`, capture/readout, sweeps, and measurement execution flows are the main concern

## Choose `Experiment` instead when

- You want the recommended workflow for running hardware-backed experiments
- You do not need to inspect controller-level execution details
- You prefer one facade for setup, execution, and analysis
