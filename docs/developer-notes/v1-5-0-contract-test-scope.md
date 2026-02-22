# v1.5.0 Beta Contract Test Scope

## Purpose

Freeze the minimum compatibility test scope required for v1.5.0 beta sign-off.

## Scope policy

- Primary contract surface: `Measurement`
- Delegation smoke surface: `Experiment` core lifecycle and measurement delegation
- Required execution modes: regular path and `mock_mode=True` compatibility path

## Measurement contract matrix

| Contract item | Status | Evidence |
| --- | --- | --- |
| Primary facade import stability (`Measurement`) | covered | `tests/measurement/test_measurement_alias.py` |
| Backend-kind forwarding (`load(..., backend_kind=...)`) | covered | `tests/measurement/test_backend_kind_selection.py` |
| Legacy API delegation (`execute`, `measure` -> schedule execution) | covered | `tests/measurement/test_measurement_api_delegation.py` |
| Schedule execution custom-factory hook path | covered | `tests/measurement/test_measurement_api_delegation.py` |
| Default executor backend-kind selection (`quel1` / `quel3`) | covered | `tests/measurement/test_measurement_schedule_runner.py` |
| QuEL-3 adapter payload conversion and alias resolution | covered | `tests/measurement/test_quel3_measurement_backend_adapter.py` |
| QuEL-3 backend executor hook contract | covered | `tests/measurement/test_quel3_backend_executor.py` |
| Sampling-period source contract (`dt`) | covered | `tests/measurement/test_sampling_period_source.py` |

## Experiment delegation smoke matrix

| Contract item | Status | Evidence |
| --- | --- | --- |
| Delegation of selected facade APIs (calibration/benchmarking/context methods) | covered | `tests/experiment/test_experiment_facade_delegation.py` |
| `mock_mode=True` forwarding at Experiment creation | covered | `tests/experiment/test_experiment_mock_mode_compat.py` |
| Registry-based resolution path for execute/measure/measure_state | covered | `tests/experiment/test_measurement_service_registry_resolution.py` |
| Core lifecycle smoke: `connect`, `reload`, `run` | covered | `tests/experiment/test_experiment_facade_delegation.py` |
| Core delegation smoke: `measure_idle_states` | covered | `tests/experiment/test_experiment_facade_delegation.py` |

## Beta gate requirement

1. All currently covered items stay green.
2. Any compatibility-surface change must include this scope update in the same PR.
