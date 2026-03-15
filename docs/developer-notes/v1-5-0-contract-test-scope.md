# v1.5.0 Internal Beta Contract Test Scope

## Purpose

Freeze the minimum compatibility test scope required for the internal beta sign-off path.
The compatibility scope itself is still the one frozen for `v1.5.0b1`.
Last status update: `2026-03-15`.

## Scope policy

- Primary contract surface: `Measurement` on QuEL-1 for existing users
- Delegation smoke surface: `Experiment` core lifecycle and measurement delegation on QuEL-1
- Required execution modes: regular path and `mock_mode=True` compatibility path
- QuEL-3 coverage remains valuable, but it is not a blocking gate for internal `v1.5.0b1`

## Measurement contract matrix

| Contract item | Status | Evidence |
| --- | --- | --- |
| Primary facade import stability (`Measurement`) | covered | `tests/measurement/test_measurement_alias.py` |
| QuEL-1 backend-kind forwarding (`load(..., backend_kind=\"quel1\")`) | covered | `tests/measurement/test_backend_kind_selection.py` |
| Legacy API delegation (`execute`, `measure` -> schedule execution) | covered | `tests/measurement/test_measurement_api_delegation.py` |
| Schedule execution custom-factory hook path | covered | `tests/measurement/test_measurement_api_delegation.py` |
| Default executor backend-kind selection for existing controller path | covered | `tests/measurement/test_measurement_schedule_runner.py` |
| QuEL-1 controller / qubecalib compatibility contract | covered | `tests/backend/test_qubecalib_compat_contract.py`, `tests/backend/test_quel1_backend_controller_qubecalib_delegation.py` |
| QuEL-1 parallel execution / resource cleanup behavior | covered | `tests/backend/test_quel1_backend_controller_parallel_execution.py`, `tests/backend/test_quel1_backend_controller_resource_cleanup.py` |
| Sampling-period source contract (`dt`) | covered | `tests/measurement/test_sampling_period_source.py` |

## Experiment delegation smoke matrix

| Contract item | Status | Evidence |
| --- | --- | --- |
| Delegation of selected facade APIs (calibration/benchmarking/context methods) | covered | `tests/experiment/test_experiment_facade_delegation.py` |
| `mock_mode=True` forwarding at Experiment creation | covered | `tests/experiment/test_experiment_mock_mode_compat.py` |
| Registry-based resolution path for execute/measure/measure_state | covered | `tests/experiment/test_measurement_service_registry_resolution.py` |
| Core lifecycle smoke: `connect`, `reload`, `run` | covered | `tests/experiment/test_experiment_facade_delegation.py` |
| Core delegation smoke: `measure_idle_states` | covered | `tests/experiment/test_experiment_facade_delegation.py` |

## QuEL-3 follow-up matrix (non-blocking for internal `v1.5.0b1`)

| Contract item | Status | Evidence |
| --- | --- | --- |
| QuEL-3 adapter payload conversion and alias resolution | covered | `tests/measurement/test_quel3_measurement_backend_adapter.py` |
| QuEL-3 controller-to-manager execution contract | covered | `tests/backend/test_quel3_backend_controller.py` |
| QuEL-3 integration compatibility with current `quelware-client` resolver API (`InstrumentResolver`) | partial | `tests/backend/test_quel3_backend_controller.py` (resolver-path payload resolution with resolver-compatible doubles); TODO: add direct import/usage regression against local quelware package layout. |
| QuEL-3 compatibility fallback for legacy reset and backend-settings override requests | covered | `tests/experiment/test_experiment_context_skew_file.py`, `tests/experiment/test_session_service.py`, `tests/backend/test_system_manager.py` |
| Target-to-alias auto-resolution by wiring/port with fail-fast errors on unresolved/ambiguous cases | covered | `tests/backend/test_quel3_backend_controller.py`, `tests/measurement/test_quel3_measurement_backend_adapter.py` |
| Multi-instrument synchronized trigger including cross-unit execution | gap | TODO (new backend execution tests + hardware gate evidence) |
| Capture-mode contract (`avg`=`AVERAGED_VALUE`, `single`=`VALUES_PER_ITER`, waveform inspection=`AVERAGED_WAVEFORM`) | partial | Source implementation in `src/qubex/backend/quel3/managers/execution_manager.py`; TODO: add explicit execution/result contract tests including waveform inspection path. |

## Beta gate requirement

1. All currently covered QuEL-1 items stay green.
2. Any compatibility-surface change must include this scope update in the same PR.
3. QuEL-3 `gap` / `partial` items above are tracked separately and do not block internal `v1.5.0b1` sign-off.
